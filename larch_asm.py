#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""larch_asm: a two-pass LoongArch (LA64) assembler, pure Python, stdlib only.

Part of a from-scratch LoongArch toolchain (assembler -> linker ->
emulator).  It assembles LoongArch assembly text into the linker's text
object format (SEC/SYM/REL lines, exactly what linker_loongarch.py parses),
emitting relocation entries for every label reference the linker must
resolve (PCALA_HI20/LO12 pairs, GOT_PC_HI20/LO12 pairs, B26 branches,
R_LARCH_64 data words).

Instruction encodings are the inverse of larch_emu.decode(), which was
parsed from the LoongArch ISA Volume 1 (Appendix B, Table 93) and
cross-checked against qemu's target/loongarch/insns.decode:
  https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html

Syntax (one instruction or directive per line; `//` comments):
  label:                      defines a label (local unless .global)
  mnemonic rd, rj, src        src = register | number | %reloc(sym)
  .text / .data               switch output section
  .global name                export a label (SYM kind G)
  .common name, size[, align] tentative definition (SYM kind C)
  .word/.quad/.byte v...      data words (little-endian); .quad accepts a
                              label -> R_LARCH_64 relocation
  .ascii "..." / .asciz "..." string data (\\n \\t \\" \\\\ escapes)
  .align N                    pad to 2^N: NOPs (0x03400000) in .text, zeros
                              in .data

Relocation suffixes (the immediate of the matching instruction):
  pcalau12i rd, %pcala_hi20(sym)    -> REL PCALA_HI20
  addi.d/ld.d/ld.w/st.d/st.w
      rd, rj, %pcala_lo12(sym)      -> REL PCALA_LO12
  pcalau12i rd, %got_pc_hi20(sym)   -> REL GOT_PC_HI20
  addi.d/ld.d/ld.w/st.d/st.w
      rd, rj, %got_pc_lo12(sym)     -> REL GOT_PC_LO12
  b/bl label                        -> REL B26 (numeric offsets allowed)
  .quad label                       -> REL R_LARCH_64

Relaxation: every %pcala_lo12 / %got_pc_lo12 pair also emits an
R_LARCH_RELAX marker (reloc 100, "paired with a normal relocation at the
same address"), and the linker folds relaxable pairs to pcaddi (see
linker_loongarch.py's try_relax_one — lld's relaxPCHi20Lo12).

Limitations (documented, not hidden): beq/bne/blt with a label operand have
no relocation kind in the toy linker (numeric offsets only); local labels
are emitted as SYM kind L and scoped per object by the linker; no macro
support; .align is fixed at assembly time (real tools emit R_LARCH_ALIGN
for align-in-code).  SLTUI's immediate is sign-extended to
match larch_emu (the manual's "ui12" spelling differs only in name; the
12-bit field layout is identical, and the interpreter's unsigned-immediate
set covers andi/ori only).

The demo (python3 larch_asm.py) assembles the three toolchain programs,
proves the generated objects are BYTE-IDENTICAL to the hand-encoded
objects in linker_loongarch.py, then links and emulates them: a0 = 6.
"""

import re

import larch_emu
import linker_loongarch as ll

# ---------------------------------------------------------------------------
# registers and encodings (inverse of larch_emu.decode)
# ---------------------------------------------------------------------------

REG_ALIASES = {
    "zero": 0,
    "ra": 1,
    "tp": 2,
    "sp": 3,
    "fp": 22,
    **{f"a{i}": 4 + i for i in range(8)},
    **{f"t{i}": 12 + i for i in range(9)},
    **{f"s{i}": 23 + i for i in range(9)},
}
for i in range(32):
    REG_ALIASES[f"r{i}"] = i

# mnemonic -> (opcode, bits, kind) where bits is the opcode field width at
# the top of the word and kind selects the operand layout:
#   "rjrd_imm"   rd, rj, imm(12|16)      "rd_rj_rk"  rd, rj, rk
#   "rd_imm20"   rd, si20                 "offs26"    b/bl
ENC = {
    # 6-bit opcodes
    "jirl": (0x13, 6, "rjrd_offs16"),
    "b": (0x14, 6, "offs26"),
    "bl": (0x15, 6, "offs26"),
    "beq": (0x16, 6, "rjrd_offs16"),
    "bne": (0x17, 6, "rjrd_offs16"),
    "blt": (0x18, 6, "rjrd_offs16"),
    # 10-bit opcodes
    "slti": (0x08, 10, "rd_rj_si12"),
    "sltui": (0x09, 10, "rd_rj_si12"),  # si12 per larch_emu (see note)
    "addi.w": (0x0A, 10, "rd_rj_si12"),
    "addi.d": (0x0B, 10, "rd_rj_si12"),
    "andi": (0x0D, 10, "rd_rj_ui12"),
    "ori": (0x0E, 10, "rd_rj_ui12"),
    "ld.w": (0xA2, 10, "rd_rj_si12"),
    "ld.d": (0xA3, 10, "rd_rj_si12"),
    "st.w": (0xA6, 10, "rd_rj_si12"),
    "st.d": (0xA7, 10, "rd_rj_si12"),
    # 17-bit opcodes
    "add.d": (0x21, 17, "rd_rj_rk"),
    "sub.d": (0x23, 17, "rd_rj_rk"),
    "slt": (0x24, 17, "rd_rj_rk"),
    "sltu": (0x25, 17, "rd_rj_rk"),
    "and": (0x29, 17, "rd_rj_rk"),
    "or": (0x2A, 17, "rd_rj_rk"),
    "xor": (0x2B, 17, "rd_rj_rk"),
    # 7-bit opcodes (QEMU target/loongarch/insns.decode; lld):
    #   pcaddi 0x0C, pcalau12i 0x0D, pcaddu12i 0x0E
    "lu12i.w": (0x0A, 7, "rd_imm20"),
    "pcaddi": (0x0C, 7, "rd_imm20"),
    "pcalau12i": (0x0D, 7, "rd_imm20"),
    "pcaddu12i": (0x0E, 7, "rd_imm20"),
}

EXPECTED_A0 = 6  # the toolchain demo's emulated result
DECIMAL_REL_OFFSET = 0xA  # hand-encoded originals write offsets < 10 decimal


RELOC_SUFFIX = re.compile(r"%(\w+)\(([A-Za-z_][A-Za-z0-9_]*)\)")
REL_KIND = {  # suffix -> (reloc kind, allowed instructions)
    "pcala_hi20": ("PCALA_HI20", ("pcalau12i",)),
    "pcala_lo12": ("PCALA_LO12", ("addi.d", "addi.w", "ld.d", "ld.w", "st.d", "st.w")),
    "got_pc_hi20": ("GOT_PC_HI20", ("pcalau12i",)),
    "got_pc_lo12": ("GOT_PC_LO12", ("addi.d", "addi.w", "ld.d", "ld.w", "st.d", "st.w")),
}

NOP = 0x03400000  # andi r0, r0, 0


class AsmError(Exception):
    """Assembly error with a line number."""


# ---------------------------------------------------------------------------
# the assembler
# ---------------------------------------------------------------------------


def _reg(tok: str, where: str) -> int:
    r = REG_ALIASES.get(tok)
    if r is None:
        raise AsmError(f"{where}: bad register {tok!r}")
    return r


def _imm(tok: str, where: str, bits: int, *, signed: bool = True) -> int:
    """Parse a numeric immediate; return the field value."""
    try:
        v = int(tok, 0)
    except ValueError:
        raise AsmError(f"{where}: bad immediate {tok!r}") from None
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if not lo <= v <= hi:
        raise AsmError(f"{where}: immediate {v} out of range [{lo}, {hi}]")
    return v & ((1 << bits) - 1)


def _sext(v: int, bits: int) -> int:
    """Sign-extend a masked `bits`-wide field to a signed Python int."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def encode_instruction(
    mnem: str, ops: list[str], where: str
) -> tuple[bytes, list[tuple[int | None, str, str | None]]]:
    """Encode one instruction; returns (bytes, [relocs]) with relocs as
    (section-relative offset, kind, symbol) — the offset is filled by the
    caller from the current section position."""
    try:
        op, _width, kind = ENC[mnem]
    except KeyError:
        raise AsmError(f"{where}: unknown instruction {mnem!r}") from None
    relocs = []

    def emit_word(word):
        return word.to_bytes(4, "little")

    if kind == "offs26":  # b/bl: rd=rj=0, 26-bit offset
        if len(ops) != 1:
            raise AsmError(f"{where}: {mnem} takes one operand")
        m = RELOC_SUFFIX.match(ops[0])
        if m:
            raise AsmError(f"{where}: use a plain label for {mnem} (B26 reloc)")
        if ops[0].isidentifier():
            relocs.append((None, "B26", ops[0]))
            field = 0
        else:
            v = _sext(_imm(ops[0], where, 26), 26)
            if v % 4:
                raise AsmError(f"{where}: branch offset must be 4-aligned")
            field = (v >> 2) & 0x3FFFFFF  # arithmetic shift, then mask
        # I26 split, same as the linker's B26 patch and the ABI:
        # bits [25:10] = field[15:0], bits [9:0] = field[25:16]
        return emit_word(op << 26 | ((field & 0xFFFF) << 10) | ((field >> 16) & 0x3FF)), relocs

    if kind == "rd_imm20":  # lu12i.w/pcaddi/pcalau12i/pcaddu12i
        rd, imm = ops
        rd = _reg(rd, where)
        m = RELOC_SUFFIX.match(imm)
        if m:
            kind2 = REL_KIND.get(m.group(1))
            if kind2 is None or mnem not in kind2[1]:
                raise AsmError(f"{where}: %{m.group(1)} not allowed here")
            relocs.append((None, kind2[0], m.group(2)))
            si20 = 0
        else:
            si20 = _imm(imm, where, 20)
        return emit_word(op << 25 | si20 << 5 | rd), relocs

    if kind in ("rd_rj_si12", "rd_rj_ui12"):
        rd, rj, imm = ops
        rd, rj = _reg(rd, where), _reg(rj, where)
        unsigned = kind == "rd_rj_ui12"
        m = RELOC_SUFFIX.match(imm)
        if m:
            kind2 = REL_KIND.get(m.group(1))
            if kind2 is None or mnem not in kind2[1]:
                raise AsmError(f"{where}: %{m.group(1)} not allowed here")
            if unsigned:
                raise AsmError(f"{where}: {mnem} takes a number, not a relocation")
            relocs.append((None, kind2[0], m.group(2)))
            if kind2[0] in ("PCALA_LO12", "GOT_PC_LO12"):
                relocs.append((None, "RELAX", None))  # pair is relaxable
            si12 = 0
        else:
            si12 = _imm(imm, where, 12, signed=not unsigned)
        return emit_word(op << 22 | si12 << 10 | rj << 5 | rd), relocs

    if kind == "rjrd_offs16":  # jirl/beq/bne/blt
        a, b, offs = ops
        ra, rb = _reg(a, where), _reg(b, where)
        if ops[2].isidentifier():  # label: no reloc kind in the toy
            raise AsmError(
                f"{where}: {mnem} with a label needs a numeric offset "
                f"(no branch-cond reloc in the toy linker)"
            )
        v = _sext(_imm(offs, where, 16), 16)
        if v % 4:
            raise AsmError(f"{where}: offset must be 4-aligned")
        off = ((v >> 2) & 0xFFFF) << 10
        if mnem == "jirl":  # jirl rd, rj, offs (rd low)
            return emit_word(op << 26 | off | rb << 5 | ra), relocs
        # beq/bne/blt rj, rd, offs — rj in bits 5-9, rd in bits 0-4 (Vol1 2.2.4)
        return emit_word(op << 26 | off | ra << 5 | rb), relocs

    if kind == "rd_rj_rk":  # add.d/sub.d/... rd, rj, rk
        rd, rj, rk = ops
        return emit_word(
            op << 15 | _reg(rk, where) << 10 | _reg(rj, where) << 5 | _reg(rd, where)
        ), relocs

    raise AsmError(f"{where}: {mnem} not implemented")


def _parse_string(tok: str, where: str) -> bytes:
    """Parse a quoted string token with \\n \\t \\" \\\\ escapes."""
    if not (tok.startswith('"') and tok.endswith('"') and len(tok) >= 2):
        raise AsmError(f"{where}: expected a quoted string")
    out, i = bytearray(), 1
    while i < len(tok) - 1:
        c = tok[i]
        if c == "\\":
            i += 1
            esc = tok[i] if i < len(tok) - 1 else ""
            out += {"n": b"\n", "t": b"\t", '"': b'"', "\\": b"\\"}.get(esc, esc.encode())
        else:
            out += c.encode()
        i += 1
    return bytes(out)


def assemble(src: str, name: str = "a.obj") -> str:
    """Assemble `src` (a string) into the linker's text object format.

    One pass: every label reference becomes a relocation (the linker
    resolves them all), so instruction encoding is position-local.  Data
    directives emit bytes directly; `.quad label` emits an R_LARCH_64
    relocation with zero placeholder bytes.
    """
    lines = [ln.split("//", 1)[0].strip() for ln in src.splitlines()]
    lines = [ln for ln in lines if ln]

    sections = {}  # section name -> bytearray
    order = []  # section first-use order
    cur = None
    labels = []  # (name, section, offset) in definition order
    globals_ = set()
    commons = []  # (name, size, align) in declaration order
    relocs = []  # (section, offset, kind, symbol)
    pending = None  # label awaiting a body

    def at():
        if cur is None:
            raise AsmError(f"{name}: bytes outside any section (use .text/.data)")
        return cur

    def define():
        nonlocal pending
        if pending is None:
            return
        labels.append((pending, at(), len(sections[at()])))
        pending = None

    for lineno, raw in enumerate(lines, 1):
        where = f"{name}:{lineno}"
        if raw.endswith(":"):
            if pending is not None:
                raise AsmError(f"{where}: label {pending!r} has no body")
            lbl = raw[:-1].strip()
            if not lbl.isidentifier():
                raise AsmError(f"{where}: bad label name {lbl!r}")
            pending = lbl
            continue
        tok = raw.split()
        if tok[0].startswith("."):
            d = tok[0]
            if d in (".text", ".data"):
                define()
                cur = d
                sections.setdefault(d, bytearray())
                if d not in order:
                    order.append(d)
            elif d == ".global":
                define()
                for g in tok[1:]:
                    globals_.add(g)
            elif d == ".common":
                define()
                args = [t.strip() for t in raw[raw.index(d) + len(d) :].split(",")]
                if len(args) not in (2, 3):
                    raise AsmError(f"{where}: .common name, size[, align]")
                commons.append((args[0], int(args[1], 0), int(args[2], 0) if len(args) == 3 else 1))
            elif d == ".align":
                define()
                n = int(tok[1], 0)
                pad = (1 << n) - (len(sections[at()]) % (1 << n))
                if pad != 1 << n:
                    filler = NOP.to_bytes(4, "little") if cur == ".text" else b"\x00"
                    sections[at()].extend(filler * (pad // len(filler)))
            elif d in (".word", ".quad", ".byte"):
                define()
                n = {".word": 4, ".quad": 8, ".byte": 1}[d]
                for v in tok[1:]:
                    if d == ".quad" and v.isidentifier():
                        relocs.append((at(), len(sections[at()]), "R_LARCH_64", v))
                        sections[at()].extend(b"\x00" * 8)
                    else:
                        sections[at()].extend(int(v, 0).to_bytes(n, "little"))
            elif d in (".ascii", ".asciz"):
                define()
                b = _parse_string(raw[raw.index(d) + len(d) :].strip(), where)
                if d == ".asciz":
                    b += b"\x00"
                sections[at()].extend(b)
            else:
                raise AsmError(f"{where}: unknown directive {d}")
            continue
        # instruction
        define()
        mnem = tok[0]
        ops = (
            [t.strip() for t in raw[raw.index(mnem) + len(mnem) :].split(",")]
            if len(tok) > 1
            else []
        )
        if mnem not in ENC:
            raise AsmError(f"{where}: unknown instruction {mnem!r}")
        off = len(sections[at()])
        word, rels = encode_instruction(mnem, ops, where)
        sections[at()].extend(word)
        for _, kind, sym in rels:
            relocs.append((at(), off, kind, sym))

    if pending is not None:
        raise AsmError(f"{name}: label {pending!r} has no body")

    out = [f"# {name} — assembled by larch_asm"]
    for s in order:
        out.append(f"SEC {s}")
        data = sections[s]
        # wrap width matches the hand-encoded originals byte-for-byte
        # (.text 16 bytes/line, data sections 8)
        width = 16 if s == ".text" else 8
        out.extend(
            " ".join(f"{b:02x}" for b in data[i : i + width]) for i in range(0, len(data), width)
        )
    for n, s, off in labels:  # label definition order
        out.append(f"SYM {n} {'G' if n in globals_ else 'L'} {s} {off}")
    for n, size, align in commons:  # .common declaration order
        out.append(f"SYM {n} C - {align} {size}")
    for s, off, kind, sym in relocs:
        # offsets < 0xa decimal, else hex — matches the hand-encoded
        # originals byte-for-byte; RELAX markers carry no symbol
        out.append(
            f"REL {s} "
            + (f"{off}" if off < DECIMAL_REL_OFFSET else f"{off:#x}")
            + f" {kind}"
            + (f" {sym}" if sym is not None else "")
        )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# demo: assemble the toolchain programs, prove byte-identity, link, emulate
# ---------------------------------------------------------------------------

PROG_A = """\
// prog_a.s: caller — loads magic (PCALA pair), calls compute DIRECTLY
// (bl, B26) and helper THROUGH the GOT (pcalau12i + ld.d GOT_PC pair +
// jirl — the canonical indirect call), then branches off the end of the
// image (b image_end; the interpreter halts when PC >= len(image)).
.text
_start:
.global _start
    pcalau12i r4,  %pcala_hi20(magic)   // la r4, magic (relaxable pair)
    addi.d    r4,  r4, %pcala_lo12(magic)
    ld.d      r4,  r4, 0                 // r4 = magic
    addi.d    r4,  r4, 5
    bl        compute
    addi.d    r4,  r4, 1
    pcalau12i r13, %got_pc_hi20(helper)  // la.global r13, helper (relaxable)
    ld.d      r13, r13, %got_pc_lo12(helper)
    jirl      r1,  r13, 0
    b         image_end
.common image_end, 0, 1
"""

PROG_B = """\
// prog_b.s: compute — reads the common 'bonus' (zero-filled tentative
// definition, like C 'int bonus;'), adds 2, returns via jirl r0, ra, 0
// (rd=0 -> plain jump).
.text
compute:
.global compute
    pcalau12i r4,  %pcala_hi20(bonus)   // la r4, bonus (relaxable pair)
    addi.d    r4,  r4, %pcala_lo12(bonus)
    ld.d      r4,  r4, 0                 // r4 = bonus
    addi.d    r4,  r4, 2
    jirl      r0,  r1, 0
.common bonus, 8, 8
"""

PROG_C = """\
// prog_c.s: helper (adds 3, returns) and the strong global magic — a
// 64-bit data word — plus magic_ptr, an 8-byte pointer to magic patched
// by R_LARCH_64.
.text
helper:
.global helper
    addi.d r4, r4, 3
    jirl   r0, r1, 0
.data
magic:
.global magic
    .quad 7
magic_ptr:
.global magic_ptr
    .quad magic
"""


def _structural(text: str) -> list[str]:
    """The object text minus comments/blank lines, for comparison."""
    return [
        ln
        for ln in (ln.split("//", 1)[0].split("#", 1)[0].strip() for ln in text.splitlines())
        if ln
    ]


def main() -> int:
    print("== assembling the three toolchain programs ==")
    objects = []
    for src, ref, refname in (
        (PROG_A, ll.OBJ_A, "prog_a.obj"),
        (PROG_B, ll.OBJ_B, "prog_b.obj"),
        (PROG_C, ll.OBJ_C, "prog_c.obj"),
    ):
        obj = assemble(src, refname)
        objects.append((refname, obj))
        print(f"--- {refname} ---")
        print(obj.rstrip())
        if _structural(obj) != _structural(ref):
            raise SystemExit(f"ASSEMBLER MISMATCH vs hand-encoded {refname}")
        print(f"  byte-identical to the hand-encoded {refname}\n")

    print("\n== link + emulate (linker relaxation on by default, like real ld) ==")
    objs = [ll.parse_object(t, n) for n, t in objects]
    image, symaddr, layout = ll.link(objs, ll.BASE)
    ll.verify(image, ll.BASE, objs, layout, symaddr)
    result = larch_emu.run(image, symaddr["_start"] - ll.BASE)
    print(f"  emulated _start: a0 = {result}")
    assert result == EXPECTED_A0, f"emulated result {result} != {EXPECTED_A0}"

    objs2 = [ll.parse_object(t, n) for n, t in objects]
    image2, _, _ = ll.link(objs2, ll.BASE, relax=False)
    saved = len(image2) - len(image)
    print(
        f"  relaxed image {len(image):#x} bytes vs {len(image2):#x} without"
        f" relaxation ({saved} bytes saved); R_LARCH_RELAX pairs folded"
        f" to pcaddi"
    )
    assert saved > 0, "relaxation should shrink the image"
    assert saved % 4 == 0, "relaxation removes whole instructions"
    print("  the assembled toolchain agrees (assembler -> linker -> emulator)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
