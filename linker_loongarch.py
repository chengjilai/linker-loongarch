#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""linker_loongarch.py: a small LoongArch static linker, pure Python, stdlib only.

The linker half of a from-scratch LoongArch toolchain (assembler ->
linker -> emulator).  The pipeline is the classic one — parse -> resolve
strong>weak>common -> merge sections -> synthesize .plt/.got/.common ->
apply relocations -> verify -> emulate — with every formula taken
verbatim from the ABI and every optimization decision traced to a
production linker (lld, binutils).

Grounding (spec, all formulas verbatim from the sources):
  * System V ABI (gABI), ch. 4 "Relocation": "Relocation is the process of
    connecting symbolic references with symbolic definitions."
      https://www.sco.com/developers/gabi/latest/ch4.reloc.html
  * LoongArch ELF ABI, table 2-17 "Relocation types" (the six kinds this
    toy implements; S = symbol value, A = addend, P = place of the storage
    unit being relocated, GP = the global pointer, G = offset into the GOT
    at which the symbol's address resides):
      R_LARCH_64          = 2    "Runtime address resolving
                                 *(int64_t *) PC = RtAddr + A"  (static: S+A)
      R_LARCH_PCALA_HI20  = 71   "(*(uint32_t *) PC) [24 ... 5] =
                                 (((S+A) & ~0xfff) - (PC & ~0xfff)) [31 ... 12]
                                 Note: The lower 12 bits are not included
                                 when calculating the PC-relative offset."
      R_LARCH_PCALA_LO12  = 72   "(*(uint32_t *) PC) [21 ... 10] = (S+A) [11 ... 0]"
      R_LARCH_GOT_PC_HI20 = 75   "(*(uint32_t *) PC) [24 ... 5] =
                                 (((GP+G) & ~0xfff) - (PC & ~0xfff)) [31 ... 12]"
      R_LARCH_GOT_PC_LO12 = 76   "(*(uint32_t *) PC) [21 ... 10] = (GP+G) [11 ... 0]"
      R_LARCH_B26         = 66   "(*(uint32_t *) PC) [9 ... 0] = (S+A-PC) [27 ... 18],
                                 (*(uint32_t *) PC) [25 ... 10] = (S+A-PC) [17 ... 2]
                                 with check 28-bit signed overflow and 4-bit
                                 aligned"
      R_LARCH_RELAX       = 100  "Instruction can be relaxed, paired with a
                                 normal relocation at the same address"
      R_LARCH_ALIGN       = 102  "Alignment directive; delete NOPs emitted
                                 for .align" (toy addend form: alignment bytes,
                                 optional max NOP bytes)
      R_LARCH_PCREL20_S2  = 103  "22-bit PC-relative offset, %pcrel_20(symbol):
                                 (*(uint32_t *) PC) [24 ... 5] = (S+A-PC) [21 ... 2]"
                                 (the relocation a folded pair leaves behind)
  * lld/ELF/Arch/LoongArch.cpp, relaxPCHi20Lo12: the fold itself —
    "pcalau12i $a0, %pc_hi20(sym) | %got_pc_hi20(sym); addi.w/d $a0, $a0,
    %pc_lo12(sym) | %got_pc_lo12(sym)" becomes one "pcaddi $a0" when the
    registers are canonical, the delta is 4-aligned and fits 22 bits; GOT
    pairs fold to the symbol's own address (getVA), so the GOT slot dies
    with the pair.  Relaxation runs to a fixpoint (layout -> apply ->
    fold -> re-layout) like real ld.
      https://loongson.github.io/LoongArch-Documentation/LoongArch-ELF-ABI-EN.html
    GP = GOT base, per the x86-64 psABI's definitions of the classic
    symbols: "G Represents the offset into the global offset table at
    which the relocation entry's symbol will reside during execution",
    "GOT Represents the address of the global offset table" (and gABI's
    GP is the same base in the LoongArch ABI's GP+G formulas).
      https://gitlab.com/x86-psABIs/x86-64-ABI/-/blob/master/x86-64-ABI/object-files.tex
  * LoongArch ISA Volume 1, "2.2.1.7. PCADDI, PCADDU12I, PCADDU18I,
    PCALAU12I": "The PCALAU12I instruction performs the operation that
    splice the 12-bit 0 behind the lowest bit of the 20-bit immediate data
    si20 and sign extension; the resultant data plus the PC of the
    instruction; then the lowest 12 bits of the addition result are erased
    and written into the general register rd.  PCALAU12I: tmp = PC +
    SignExtend({si20, 12'b0}, GRLEN); GR[rd] = {tmp[GRLEN-1:12], 12'b0}"
    — erasing the low 12 bits of PC is why the ABI's HI20 formula masks
    both operands with ~0xfff. "2.2.4.5. JIRL": "JIRL: GR[rd] = PC + 4;
    PC = GR[rj] + SignExtend({offs16, 2'b0}, GRLEN)" ("When rd is equal
    to 0, the function of JIRL is a common non-call indirect jump");
    "BL: GR[1] = PC + 4; PC = PC + SignExtend({offs26, 2'b0}, GRLEN)".
      https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html
  * binutils bfd/elfnn-loongarch.c, loongarch_make_plt_entry (the real
    LoongArch PLT stub; hi/lo decomposition and registers verbatim):
        hi = ((pcrel + 0x800) >> 12) & 0xfffff;
        lo = pcrel & 0xfff;
        entry[0] = 0x1c00000f | (hi & 0xfffff) << 5;
        entry[1] = ((GOT_ENTRY_SIZE == 8 ? 0x28c001ef : 0x288001ef)
                    | (lo & 0xfff) << 10);
        entry[2] = 0x4c0001ed;  /* jirl $r13, $15, 0 */
        entry[3] = 0x03400000;  /* nop */
    i.e. `pcaddu12i $t3, %hi(%pcrel(.got.plt))` (note: PCADDU12I, full PC,
    not the pcalau12i of the psABI pairs), `ld.d $t3, $t3,
    %lo(%pcrel(.got.plt))`, `jirl $t1, $t3, 0` — registers $t3 (r15) and
    $t1 (r13) per the ABI's table ($r12-$r20 = $t0-$t8). LLD agrees
    (lld/ELF/Arch/LoongArch.cpp, writePltEntry): "pcaddu12i $t3,
    %pcrel_hi20(f@.got.plt) ... jirl $t1, $t3, 0". This toy keeps the
    three instructions (12 bytes) and drops the trailing nop.
      https://github.com/gnutools/binutils-gdb/blob/master/bfd/elfnn-loongarch.c
  * binutils bfd/elfnn-loongarch.c, RELOCATE_CALC_PC32_HI20 (used for both
    PCALA_HI20 and GOT_PC_HI20; the ABI formula plus the rounding the
    hardware needs when the LO12's sign bit is set; LLD's
    getLoongArchPageDelta does the same: "if (dest & 0x800) result +=
    0x1000"):
        bfd_vma __lo = (relocation) & ((bfd_vma)0xfff);
        relocation = (relocation & ~(bfd_vma)0xfff)
                      - (pc & ~(bfd_vma)0xfff);
        if (__lo > 0x7ff)
            relocation += 0x1000;
  * Ian Lance Taylor, "Linkers" parts 2/5/6 — resolution rules, common
    symbols, section merging, PLT/GOT synthesis.
      https://www.airs.com/blog/archives/39
      https://www.airs.com/blog/archives/42
      https://www.airs.com/blog/archives/43

Toy object format — one directive or hex bytes per line:
  SEC name            start a section (same-named sections are merged)
  SYM name K sec off [size]
                      K = G global | L local | W weak | C common.
                      sec '-' = undefined reference; for C, off = alignment
                      and size = size (like SHN_COMMON: value 0 + st_size).
  REL sec off KIND sym
                      relocation: KIND = PCALA_HI20 | PCALA_LO12 |
                      GOT_PC_HI20 | GOT_PC_LO12 | B26 | R_LARCH_64; off =
                      r_offset, the address of the instruction (or word)
                      being patched. The LoongArch psABI is Rela-based;
                      here the addend is implicit and always 0 (the
                      object's immediate fields are zero), so e.g. a
                      pcalau12i+ld.d pair carries the whole 32-bit address
                      via its two relocations.

Instruction fields (per the ABI table and binutils opcodes):
  pcalau12i/pcaddu12i si20 -> bits [24:5]; ld.d/addi.d si12 -> [21:10];
  jirl offs16 (x4) -> [25:10]; b/bl offs26 (x4) -> [25:0].

The GOT/PLT policy (static link): one .got slot per
GOT_PC-referenced symbol and one 12-byte .plt entry per such symbol,
patched by the linker with the binutils stub above against the shared
slot. Slots hold the symbol's runtime address = S - BASE: the toy's
emulator (larch_emu) maps the image at virtual address 0 (its PC is an
image offset and memory starts as bytearray(image)), so a jump target
stored in a slot must be S - BASE — exactly as a dynamic linker fills a
slot with S + load_bias. The linker's absolute layout (base 0x400000) is
translation-invariant: every patched immediate (PCALA/GOT_PC fields, B26)
is identical in either space, since BASE is 12-bit aligned.

Pipeline: parse -> resolve -> layout -> apply -> verify -> emulate the
image on larch_emu.py (a module of this toolchain exposing
`run(image: bytes, entry: int) -> int`, returning the final r4/a0; memory
= bytearray(image), entry is an offset into the image, sp/r3 = 0x800000,
execution ends when PC >= len(image)).

Run:  python3 -B linker_loongarch.py
      (tests: python3 -B -m unittest -v test_linker_loongarch)
"""

import sys
from dataclasses import dataclass

import larch_emu  # the emulator; see the run() contract in the docstring

BASE, ALIGN, MASK = 0x400000, 16, 0xFFFFFFFFFFFFFFFF  # base, align, 64-bit

# opcode bases, binutils opcodes/loongarch-opc.c (verbatim):
#   pcaddi 0x18000000, pcalau12i 0x1a000000, pcaddu12i 0x1c000000,
#   ld.d 0x28c00000, jirl 0x4c000000, nop 0x03400000
OP_PCADDI, OP_PCALAU12I = 0x18000000, 0x1A000000
OP_PCADDU12I, OP_LD_D = 0x1C000000, 0x28C00000
OP_JIRL, OP_NOP = 0x4C000000, 0x03400000


class LinkError(Exception):
    pass


# --- 1. object files: sections + symbol tables + relocations ---------------


Symbol = tuple[str, str | None, int, int]  # (kind, section|None, value, size)
Reloc = tuple[str, int, str, str | None, int]  # (section, offset, kind, symbol, addend)


@dataclass
class Align:
    """One R_LARCH_ALIGN request.

    `offset` points at the first byte of the max-NOP run the assembler
    emitted (`align - 4` bytes). `max_bytes` is the maximum number of NOPs
    the sequence is willing to keep; 0 means "always align" (binutils'
    symbol-index-0 form)."""

    section: str
    offset: int
    align: int  # byte alignment, a power of two >= 4
    max_bytes: int  # max NOPs to keep; 0 = unlimited


@dataclass
class AlignStep:
    """One R_LARCH_ALIGN decision made during a layout pass."""

    pass_no: int
    obj: str
    section: str
    offset: int
    align: int
    max_bytes: int
    site: int  # address of the first NOP in this layout
    needed: int  # NOP bytes needed to reach the boundary from `site`
    removed: int  # NOP bytes deleted in this layout
    abandoned: bool  # max_bytes was exceeded: all NOPs removed


@dataclass
class Object:
    name: str
    sections: dict[str, bytearray]  # section name -> bytes
    symbols: dict[str, Symbol]  # kind 'G'|'L'|'W'|'C'
    relocs: list[Reloc]
    aligns: list[Align]  # R_LARCH_ALIGN requests, sorted by offset


def parse_object(text: str, name: str) -> Object:
    """Parse the text object format above into an Object."""
    obj = Object(name, {}, {}, [], [])
    cur = None
    for ln, raw in enumerate(text.splitlines(), 1):
        tok = raw.strip().split()
        if not tok or tok[0].startswith("#"):
            continue
        if tok[0] == "SEC":
            cur = tok[1]
            obj.sections[cur] = bytearray()
        elif tok[0] == "SYM":
            sec = None if tok[3] == "-" else tok[3]
            if tok[2] not in "GLWC":
                raise LinkError(f"{name}:{ln}: bad symbol kind {tok[2]!r}")
            size = int(tok[5], 0) if len(tok) > 5 else 0
            obj.symbols[tok[1]] = (tok[2], sec, int(tok[4], 0), size)
        elif tok[0] == "REL":
            # REL sec off KIND sym [addend]; RELAX marker may omit the symbol
            addend = int(tok[5], 0) if len(tok) > 5 else 0
            obj.relocs.append(
                (tok[1], int(tok[2], 0), tok[3], tok[4] if len(tok) > 4 else None, addend)
            )
        elif tok[0] == "ALIGN":
            # ALIGN sec off alignment [max_bytes]
            sec, off = tok[1], int(tok[2], 0)
            align, max_bytes = int(tok[3], 0), int(tok[4], 0) if len(tok) > 4 else 0
            if align < 4 or align & (align - 1):
                raise LinkError(f"{name}:{ln}: alignment must be a power of two >= 4")
            if off % 4 or max_bytes % 4 or not 0 <= max_bytes < align:
                raise LinkError(
                    f"{name}:{ln}: bad R_LARCH_ALIGN off/max "
                    f"({off:#x}, {max_bytes:#x}) for alignment {align:#x}"
                )
            obj.aligns.append(Align(sec, off, align, max_bytes))
        elif cur is not None:
            obj.sections[cur].extend(int(b, 16) for b in tok)
        else:
            raise LinkError(f"{name}:{ln}: bytes outside any section")
    return obj


# --- 2. symbol resolution --------------------------------------------------


def resolve(objects):
    """Resolve globals: strong (G) beats weak (W) beats common (C); two
    strongs -> error; two weaks -> first in link order wins; commons merge
    (largest size, then largest alignment); undefined refs are errors
    unless weak, which resolves to 0. Returns (defs, commons)."""
    defs, commons, undef = {}, {}, {}
    locs = {}
    for o in objects:
        for n, (kind, sec, val, size) in o.symbols.items():
            if kind == "L":  # object-local: own namespace
                if sec is None:
                    raise LinkError(f"{o.name}: local symbol '{n}' must be defined")
                locs.setdefault(o.name, {})[n] = (sec, val)
            elif kind == "C":
                if sec is not None:
                    raise LinkError(f"{o.name}: common symbol '{n}' must be undefined (sec '-')")
                if n not in commons or (size, val) > (commons[n][0], commons[n][1]):
                    commons[n] = (size, val)
            elif sec is None:  # undefined reference
                if kind == "W":
                    continue  # weak: resolves to 0
                undef[n] = o.name
            elif kind == "W":
                defs.setdefault(n, (kind, o.name, sec, val))  # first weak wins
            else:  # G: strong
                if n in defs and defs[n][0] == "G":
                    raise LinkError(f"multiple definition of global '{n}'")
                defs[n] = (kind, o.name, sec, val)
    for n in list(commons):  # any definition beats a common
        if n in defs:
            del commons[n]
    if missing := sorted(n for n in undef if n not in defs and n not in commons):
        raise LinkError("undefined references: " + ", ".join(f"{n} ({undef[n]})" for n in missing))
    return defs, commons, locs


# --- 3+4. layout and relocation application --------------------------------


Applied = tuple[str, str, int, str, str | None, int, int, int]
GotSlot = tuple[str, int, int]


@dataclass
class Layout:
    base: int
    sec_addr: dict[str, int]  # output section name -> address
    offs: dict[tuple[str, str], int]  # (obj, section) -> address
    plt: dict[str, int]  # symbol -> PLT entry address
    got: dict[str, int]  # symbol -> GOT slot address
    common: dict[str, int]  # symbol -> .common address
    applied: list[Applied]  # (obj, sec, roff, kind, sym, site, target, field)
    slots: list[GotSlot]  # (sym, slot, value) — patched GOT slots
    plt_addr: int
    got_addr: int
    common_addr: int
    common_size: int
    # (obj, section) -> [(align offset, bytes removed)] in offset order.
    # Used to shift later reloc/symbol offsets in the output image.
    align_delta: dict[tuple[str, str], list[tuple[int, int]]]


def pcrel_page_field(target: int, pc: int) -> int:
    """R_LARCH_PCALA_HI20 / R_LARCH_GOT_PC_HI20 field value: the ABI's
    (((S+A) & ~0xfff) - (PC & ~0xfff)) [31:12] with binutils'
    RELOCATE_CALC_PC32_HI20 rounding (add 0x1000 when the LO12's sign bit
    is set — pcalau12i erases PC's low 12 bits, so the hi must carry the
    page of the *rounded-up* target)."""
    lo = target & 0xFFF
    hi = (target & ~0xFFF) - (pc & ~0xFFF)
    if lo > 0x7FF:
        hi += 0x1000
    return hi >> 12  # exact: hi is a multiple of 0x1000


def align_shift(plan, offset):
    """Bytes deleted before `offset` by one section's ALIGN plan."""
    return sum(removed for off, removed in plan if off < offset)


def _make_align_plan(o, sec, start, *, pass_no=0, align_log=None):
    """Compute R_LARCH_ALIGN deletions for one contribution at `start`.

    The input section holds the assembler's maximum NOP run at each ALIGN
    record (`align - 4` bytes). For each request, in offset order, compute
    how many NOPs the current address actually needs and delete the rest;
    later requests see the already-shrunk section. `max_bytes` abandons the
    alignment (delete all NOPs) when the needed run would exceed it, per
    binutils `loongarch_relax_align` / lld's R_LARCH_ALIGN handling.
    """
    plan = []
    removed_before = 0
    for al in sorted((a for a in o.aligns if a.section == sec), key=lambda a: a.offset):
        site = start + al.offset - removed_before
        off = site & (al.align - 1)
        needed = 0 if off == 0 else al.align - off
        all_bytes = al.align - 4
        if al.max_bytes and needed > al.max_bytes:
            removed, keep, abandoned = all_bytes, 0, True
        else:
            removed, keep, abandoned = all_bytes - needed, needed, False
        if removed < 0:
            raise LinkError(
                f"{o.name}: R_LARCH_ALIGN at {sec}+{al.offset:#x} has "
                f"{all_bytes} NOP bytes but needs {needed} at {site:#x}"
            )
        if align_log is not None:
            align_log.append(
                AlignStep(
                    pass_no,
                    o.name,
                    sec,
                    al.offset,
                    al.align,
                    al.max_bytes,
                    site,
                    needed,
                    removed,
                    abandoned,
                )
            )
        # Keep the first `keep` NOPs; the deletion starts after them.
        plan.append((al.offset + keep, removed))
        removed_before += removed
    return plan


def _shrink_section(data, plan):
    """Copy `data` minus the byte ranges recorded in `plan`."""
    out = bytearray()
    prev = 0
    for off, removed in plan:
        if off < prev or off + removed > len(data):
            raise LinkError(f"ALIGN deletion {off:#x}+{removed:#x} outside section")
        out += data[prev:off]
        prev = off + removed
    out += data[prev:]
    return out


@dataclass
class RelaxStep:
    """One relaxation decision (fold or skip) in the layout/apply/relax loop."""

    pass_no: int
    obj: str
    section: str
    offset: int  # LO12 offset in the section, before the fold
    kind: str  # PCALA_LO12 | GOT_PC_LO12
    sym: str | None
    site: int  # LO12 address in the current layout
    target: int | None  # address the folded pcaddi would reach
    delta: int | None  # target - (site - 4), the folded pcaddi's delta
    decision: str  # "fold" | "skip"
    reason: str | None = None  # why a skip happened; None for folds


def try_relax_one(objects, layout, symaddr, *, pass_no=0, log=None):
    """Fold one relaxable pcalau12i+addi/ld pair (R_LARCH_RELAX semantics).

    The transformation is lld's (lld/ELF/Arch/LoongArch.cpp,
    relaxPCHi20Lo12):

      pcalau12i $a0, %pc_hi20(sym) | %got_pc_hi20(sym)
      addi.w/d  $a0, $a0, %pc_lo12(sym) | %got_pc_lo12(sym)
      ->
      pcaddi    $a0, (sym - PC) >> 2

    Conditions (same source): the pair must be canonical (pcalau12i's rd ==
    the second instruction's rj == its rd), the delta must be 4-aligned and
    fit pcaddi's signed 22-bit range, and GOT pairs fold only for symbols
    that are link-time defined (lld's `!rHi20.sym->isDefined()` check).
    Both PCALA and GOT folds target the symbol's own address (lld computes
    dest = sym->getVA()), never the GOT slot: the original GOT pair loads
    the slot's value, while pcaddi yields the slot's address.

    On a fold the HI20 instruction (4 bytes earlier) is deleted, the LO12
    word is rewritten as pcaddi, the HI20 + RELAX relocations are removed,
    and the LO12 relocation is REWRITTEN as PCREL20_S2 — it stays in the
    object so every later layout pass re-applies the delta with the final
    addresses (lld keeps the relocation as R_LARCH_PCREL20_S2; values are
    never frozen).

    Returns True when something changed; the caller re-lays-out and
    re-checks (relaxation is a fixpoint: each fold shrinks the image, and
    later folds may become possible). If `log` is a list, every candidate
    seen before this pass's first fold is appended as a RelaxStep.
    """
    defined = {n for o in objects for n, (_, sec, _, _) in o.symbols.items() if sec is not None}

    def note(decision, target, delta, reason=None):
        if log is not None:
            log.append(
                RelaxStep(
                    pass_no,
                    o.name,
                    sec,
                    roff,
                    kind,
                    sym,
                    pc,
                    target,
                    delta,
                    decision,
                    reason,
                )
            )

    for o in objects:
        for sec, roff, kind, sym, lo_add in list(o.relocs):
            if kind not in ("PCALA_LO12", "GOT_PC_LO12"):
                continue
            if not any(s == sec and r == roff and k == "RELAX" for s, r, k, _, _ in o.relocs):
                continue  # not marked relaxable
            if not any(
                s == sec and r == roff - 4 and k in ("PCALA_HI20", "GOT_PC_HI20")
                for s, r, k, _, _ in o.relocs
            ):
                raise LinkError(
                    f"{o.name}: RELAX at {sec}+{roff:#x} without "
                    f"a paired HI20 at {sec}+{roff - 4:#x}"
                )
            hi_add = next(
                a
                for s, r, k, _, a in o.relocs
                if s == sec and r == roff - 4 and k in ("PCALA_HI20", "GOT_PC_HI20")
            )
            if hi_add != 0 or lo_add != 0:
                note("skip", None, None, "non-zero addend in relaxable pair")
                continue
            pc = (
                layout.offs[(o.name, sec)]
                + roff
                - align_shift(layout.align_delta[(o.name, sec)], roff)
            )
            if kind == "GOT_PC_LO12":
                if sym not in defined:
                    note("skip", None, None, "GOT target is not link-time defined")
                    continue
                target = symaddr.get(sym)
            else:
                target = symaddr.get(sym)
            if target is None:
                raise LinkError(f"{o.name}: RELAX for undefined symbol '{sym}'")
            delta = target - (pc - 4)  # the folded pcaddi's own PC
            if (delta & 3) or not -(1 << 21) <= delta < (1 << 21):
                note("skip", target, delta, "outside pcaddi's 22-bit range")
                continue
            # lld's checks: pcalau12i rd == insn rj == insn rd, and the
            # opcode shape (PCALA pairs must end in addi.w/d; GOT pairs in
            # ld.w/d — an address-take vs a load)
            hi = int.from_bytes(o.sections[sec][roff - 4 : roff], "little")
            lo = int.from_bytes(o.sections[sec][roff : roff + 4], "little")
            if hi & 0x1F != (lo >> 5) & 0x1F or (lo >> 5) & 0x1F != lo & 0x1F:
                note("skip", target, delta, "non-canonical registers")
                continue
            want = (0x0A, 0x0B) if kind == "PCALA_LO12" else (0xA2, 0xA3)
            if (lo >> 22) not in want:
                note("skip", target, delta, "wrong opcode shape for the pair")
                continue
            note("fold", target, delta)
            _apply_relax(o, sec, roff)
            return True
    return False


def _apply_relax(o, sec, roff):
    """Delete the HI20 word at roff-4 and rewrite the LO12 word (now at
    roff-4) as pcaddi rd (si20 placeholder; the surviving PCREL20_S2
    relocation patches it in every later pass).  The HI20 and RELAX
    relocations are dropped, the LO12 one is rewritten as
    PCREL20_S2 / PCREL20_S2_GOT, and every later reloc/symbol shifts -4."""
    data = o.sections[sec]
    word = int.from_bytes(data[roff : roff + 4], "little")  # the LO12 instruction
    rd = word & 0x1F
    relaxed = OP_PCADDI | rd  # pcaddi, si20 = 0 (0x0C << 25, not pcaddu12i)
    data[roff - 4 : roff + 4] = relaxed.to_bytes(4, "little")
    # both PCALA and GOT folds target the symbol's own address (lld's
    # relaxPCHi20Lo12 computes dest = sym->getVA(); the GOT slot is only
    # valid when the symbol is link-time known — always true here — so the
    # surviving relocation is plain PCREL20_S2 and the slot can die).
    new_kind = "PCREL20_S2"
    out = []
    for s, r, k, sy, add in o.relocs:
        if s == sec and (r == roff or r == roff - 4):
            if r == roff and k in ("PCALA_LO12", "GOT_PC_LO12"):
                out.append((s, r - 4, new_kind, sy, add))  # survives, rewritten
            continue  # HI20 / RELAX dropped
        out.append((s, r - 4 if s == sec and r > roff else r, k, sy, add))
    o.relocs = out
    o.symbols = {
        n: (k2, s2, v - 4 if s2 == sec and v >= roff else v, z)
        for n, (k2, s2, v, z) in o.symbols.items()
    }
    o.aligns = [
        Align(
            a.section,
            a.offset - 4 if a.section == sec and a.offset >= roff else a.offset,
            a.align,
            a.max_bytes,
        )
        for a in o.aligns
    ]


def link(objects, base=BASE, *, relax=True, relax_log=None, align_log=None):
    """Link objects into one image. Returns (image, symaddr, layout).

    With relax=True (the default, like real ld) the layout/apply/relax
    cycle repeats to a fixpoint: each fold of a relaxable pair shrinks a
    section, addresses shift, and further folds may become possible.

    R_LARCH_ALIGN is handled inside every `link_once` layout: the full NOP
    run stays in the input object, and the layout computes how much of it
    survives at the current addresses. Pair folds and alignment deletion
    therefore feed each other through the same re-layout loop.

    If `relax_log` is a list, the relaxation loop appends a RelaxStep for
    every candidate considered (folds and skips, with the skip reason).
    If `align_log` is a list, every layout pass appends an AlignStep per
    R_LARCH_ALIGN request.
    """
    defs, commons, locs = resolve(objects)
    image, symaddr, layout = link_once(
        objects, base, defs, commons, locs, align_pass=0, align_log=align_log
    )
    if not relax:
        return image, symaddr, layout
    pass_no = 0
    while True:
        pass_no += 1
        if not try_relax_one(objects, layout, symaddr, pass_no=pass_no, log=relax_log):
            break
        defs, commons, locs = resolve(objects)  # symbol offsets shifted by the fold
        image, symaddr, layout = link_once(
            objects, base, defs, commons, locs, align_pass=pass_no, align_log=align_log
        )
    return image, symaddr, layout


def link_once(objects, base, defs, commons, locs, *, align_pass=0, align_log=None):
    """One layout + apply round (used by the relaxation fixpoint).

    R_LARCH_ALIGN is evaluated here from the current contribution
    addresses: the object still contains the full max-NOP run, and the
    layout decides how much survives. This keeps alignment decisions
    recomputable after every pair fold instead of committing a deletion
    that a later fold could invalidate.
    """
    # --- 3. layout --------------------------------------------------------
    # Merge same-named input sections in link order (Taylor part 2), then
    # synthesize .plt (one 12-byte stub per GOT_PC-referenced symbol),
    # .got (one 8-byte slot per GOT_PC-referenced symbol; the PLT stubs
    # share the slots), and .common (zero-filled, per merged common).
    merged = {}  # section name -> [(obj, bytes)]
    by_name = {o.name: o for o in objects}
    for o in objects:
        for sec, data in o.sections.items():
            merged.setdefault(sec, []).append((o.name, data))
    got_ord, seen_g = [], set()
    for o in objects:  # GOT/PLT slots in first-use order
        for _, _, kind, sym, _ in o.relocs:
            if kind in ("GOT_PC_HI20", "GOT_PC_LO12") and sym not in seen_g:
                seen_g.add(sym)
                got_ord.append(sym)
    addr = base
    sec_addr, offs = {}, {}  # output section addr; contribution addr
    align_delta = {}  # (obj, section) -> ALIGN deletion plan
    shrunk_size = {}  # (obj, section) -> contribution length after ALIGN
    for sec, parts in merged.items():
        addr = (addr + ALIGN - 1) & ~(ALIGN - 1)
        sec_addr[sec] = addr
        for oname, data in parts:
            addr = (addr + ALIGN - 1) & ~(ALIGN - 1)  # align each contribution
            offs[(oname, sec)] = addr
            o = by_name[oname]
            plan = _make_align_plan(o, sec, addr, pass_no=align_pass, align_log=align_log)
            align_delta[(oname, sec)] = plan
            total_removed = sum(removed for _, removed in plan)
            shrunk_size[(oname, sec)] = len(data) - total_removed
            addr += shrunk_size[(oname, sec)]
    addr = (addr + ALIGN - 1) & ~(ALIGN - 1)
    plt_addr = addr
    plt = {s: addr + 12 * i for i, s in enumerate(got_ord)}  # 3 insns x 4 B
    addr += 12 * len(got_ord)
    addr = (addr + 7) & ~7  # GOT entries are 8 bytes
    got_addr = addr
    got = {s: addr + 8 * i for i, s in enumerate(got_ord)}
    addr += 8 * len(got_ord)
    addr = (addr + ALIGN - 1) & ~(ALIGN - 1)
    common_addr = addr
    common = {}
    for n, (size, al) in sorted(commons.items()):  # deterministic order
        addr = (addr + al - 1) & ~(al - 1)
        common[n] = addr
        addr += size
    common_size = addr - common_addr
    image = bytearray(addr - base)
    for sec, parts in merged.items():
        for oname, data in parts:
            at = offs[(oname, sec)] - base
            out = _shrink_section(data, align_delta[(oname, sec)])
            image[at : at + len(out)] = out
    # symbol addresses come from the *resolved* table (strong-over-weak,
    # first-weak, common slots), with ALIGN deletions before the symbol
    # offset applied; undefined weaks get 0.
    symaddr = {}
    for n, (_, o, sec, val) in defs.items():
        symaddr[n] = offs[(o, sec)] + val - align_shift(align_delta[(o, sec)], val)
    symaddr.update(common)
    for o in objects:
        for n, (kind, sec, _, _) in o.symbols.items():
            if sec is None and kind == "W" and n not in symaddr:
                symaddr[n] = 0
    # --- 4. relocations ----------------------------------------------------
    applied, slots = [], []
    for o in objects:
        olocs = locs.get(o.name, {})
        for sec, roff, kind, sym, addend in o.relocs:
            shift = align_shift(align_delta[(o.name, sec)], roff)
            site = offs[(o.name, sec)] + roff - shift  # r_offset after ALIGN
            if kind in ("GOT_PC_HI20", "GOT_PC_LO12"):
                target = got.get(sym)  # GP+G: the slot's address
            elif sym in olocs:
                ls, lv = olocs[sym]
                target = offs[(o.name, ls)] + lv - align_shift(align_delta[(o.name, ls)], lv)
            else:
                target = symaddr.get(sym)
            if kind == "RELAX":
                continue  # marker: consumed by the relax pass
            if kind not in (
                "GOT_PC_HI20",
                "GOT_PC_LO12",
                "PCALA_HI20",
                "PCALA_LO12",
                "B26",
                "R_LARCH_64",
                "R_LARCH_32",
                "PCREL20_S2",
            ):
                raise LinkError(f"{o.name}: unknown relocation kind {kind}")
            if target is None:
                raise LinkError(f"{o.name}: {kind} for undefined symbol '{sym}'")
            target += addend
            img_off = site - base
            if kind == "PCREL20_S2":
                disp = target - site
                if disp % 4:
                    raise LinkError(f"PCREL20_S2 misaligned for '{sym}' at {site:#x}")
                if not -(1 << 21) <= disp < (1 << 21):
                    raise LinkError(f"PCREL20_S2 overflow for '{sym}' at {site:#x}")
                insn = int.from_bytes(image[img_off : img_off + 4], "little")
                insn = (insn & ~(0xFFFFF << 5)) | (((disp >> 2) & 0xFFFFF) << 5)
                image[img_off : img_off + 4] = insn.to_bytes(4, "little")
                field = (disp >> 2) & 0xFFFFF
            elif kind in ("PCALA_HI20", "GOT_PC_HI20"):
                field = pcrel_page_field(target, site)
                if not -(1 << 19) <= field < (1 << 19):
                    raise LinkError(f"{kind} overflow for '{sym}' at {site:#x}")
                insn = int.from_bytes(image[img_off : img_off + 4], "little")
                insn |= (field & 0xFFFFF) << 5  # si20 -> bits [24:5]
                image[img_off : img_off + 4] = insn.to_bytes(4, "little")
            elif kind in ("PCALA_LO12", "GOT_PC_LO12"):
                field = target & 0xFFF  # si12 -> bits [21:10]
                insn = int.from_bytes(image[img_off : img_off + 4], "little")
                insn |= field << 10
                image[img_off : img_off + 4] = insn.to_bytes(4, "little")
            elif kind == "B26":
                disp = target - site  # S + A - PC, A = 0
                if disp % 4:
                    raise LinkError(f"B26 misaligned for '{sym}' at {site:#x}")
                if not -(1 << 27) <= disp < (1 << 27):
                    raise LinkError(f"B26 overflow for '{sym}' at {site:#x}")
                # offs26 = (S+A-PC)[27:2]; the I26 format splits the field:
                # bits [25:10] = D[15:0] and bits [9:0] = D[25:16] (ABI:
                # "(*(uint32_t *) PC) [9 ... 0] = (S+A-PC) [27 ... 18],
                #  (*(uint32_t *) PC) [25 ... 10] = (S+A-PC) [17 ... 2]").
                field = (disp >> 2) & 0x3FFFFFF
                insn = int.from_bytes(image[img_off : img_off + 4], "little")
                insn |= ((field & 0xFFFF) << 10) | ((field >> 16) & 0x3FF)
                image[img_off : img_off + 4] = insn.to_bytes(4, "little")
            elif kind == "R_LARCH_32":
                if not -(1 << 31) <= target < (1 << 32):
                    raise LinkError(f"R_LARCH_32 overflow for '{sym}' at {site:#x}")
                field = target & 0xFFFFFFFF
                image[img_off : img_off + 4] = field.to_bytes(4, "little")
            else:  # R_LARCH_64: word64 S + A
                field = target & MASK
                image[img_off : img_off + 8] = field.to_bytes(8, "little")
            applied.append((o.name, sec, roff, kind, sym, site, target, field))
    for sym, at in plt.items():  # PLT stubs: the binutils
        p = at - base  # loongarch_make_plt_entry
        pcrel = got[sym] - at  # sequence, minus the nop:
        hi = ((pcrel + 0x800) >> 12) & 0xFFFFF  # pcaddu12i $t3, hi
        lo = pcrel & 0xFFF  # ld.d $t3, $t3, lo
        if pcrel + 0x80000800 > 0xFFFFFFFF:  # binutils' range check
            raise LinkError(f"PLT stub for '{sym}' out of range")
        stub = (
            OP_PCADDU12I | 0x0F | (hi << 5),
            OP_LD_D | 0x1EF | (lo << 10),  # rd=rj=r15 ($t3)
            OP_JIRL | 0x1ED,
        )  # rd=r13 ($t1), rj=r15
        for i, w in enumerate(stub):
            image[p + 4 * i : p + 4 * i + 4] = w.to_bytes(4, "little")
        applied.append(
            (".plt", ".plt", at - plt_addr, "PLT-STUB", sym, at, got[sym], (hi << 12) + lo)
        )
    for sym, at in got.items():  # static link: slots filled here
        # runtime address in the emulated space: larch_emu maps the image
        # at virtual address 0 (PC is an image offset), so a slot's value
        # is S - BASE — what a dynamic linker would store as S + load_bias
        # (the linker's own base 0x400000 is a link-time constant, and
        # every patched immediate is translation-invariant).
        loc = next(((o, ls[sym]) for o, ls in locs.items() if sym in ls), None)
        if loc is not None:  # GOT slot for an object-local
            o, s = loc
            value = (
                offs[(o, s[0])] + s[1] - align_shift(align_delta[(o, s[0])], s[1]) - base
            ) & MASK
        else:
            value = (symaddr.get(sym, 0) - base) & MASK
        p = at - base
        image[p : p + 8] = value.to_bytes(8, "little")
        slots.append((sym, at, value))
    return (
        image,
        symaddr,
        Layout(
            base,
            sec_addr,
            offs,
            plt,
            got,
            common,
            applied,
            slots,
            plt_addr,
            got_addr,
            common_addr,
            common_size,
            align_delta,
        ),
    )


# --- 5. verification: re-derive patches, then emulate the image ------------


def verify(image, base, objects, layout, symaddr):
    """Independently re-derive every patch from the layout (contribution
    offsets, GOT/PLT tables) and check the image bytes. Raises
    AssertionError on the first mismatch."""

    def field_at(site, lo, hi):
        w = int.from_bytes(image[site - base : site - base + 4], "little")
        return (w >> lo) & ((1 << (hi - lo + 1)) - 1)

    locs = {}
    for o in objects:
        for n, (kind, sec, val, _) in o.symbols.items():
            if kind == "L" and sec is not None:
                locs.setdefault(o.name, {})[n] = (sec, val)

    for o in objects:
        for sec, roff, kind, sym, addend in o.relocs:
            site = (
                layout.offs[(o.name, sec)]
                + roff
                - align_shift(layout.align_delta[(o.name, sec)], roff)
            )
            if kind in ("PCALA_HI20", "GOT_PC_HI20"):
                target = symaddr[sym] if kind == "PCALA_HI20" else layout.got[sym]
                target += addend
                assert field_at(site, 5, 24) == pcrel_page_field(target, site) & 0xFFFFF, (
                    f"patch for '{sym}' ({kind}) at {site:#x}"
                )
            elif kind in ("PCALA_LO12", "GOT_PC_LO12"):
                target = symaddr[sym] if kind == "PCALA_LO12" else layout.got[sym]
                target += addend
                assert field_at(site, 10, 21) == target & 0xFFF, (
                    f"patch for '{sym}' ({kind}) at {site:#x}"
                )
            elif kind == "B26":
                target = symaddr.get(sym)
                if target is None:
                    ls, lv = locs[o.name][sym]
                    target = (
                        layout.offs[(o.name, ls)]
                        + lv
                        - align_shift(layout.align_delta[(o.name, ls)], lv)
                    )
                target += addend
                disp = target - site
                field = (disp >> 2) & 0x3FFFFFF
                assert field_at(site, 0, 25) == ((field & 0xFFFF) << 10) | (
                    (field >> 16) & 0x3FF
                ), f"patch for '{sym}' (B26) at {site:#x}"
            elif kind == "PCREL20_S2":
                target = symaddr.get(sym)
                if target is None:
                    ls, lv = locs[o.name][sym]
                    target = (
                        layout.offs[(o.name, ls)]
                        + lv
                        - align_shift(layout.align_delta[(o.name, ls)], lv)
                    )
                target += addend
                assert field_at(site, 5, 24) == ((target - site) >> 2) & 0xFFFFF, (
                    f"patch for '{sym}' (PCREL20_S2) at {site:#x}"
                )
            elif kind == "R_LARCH_32":
                target = symaddr.get(sym)
                if target is None:
                    ls, lv = locs[o.name][sym]
                    target = (
                        layout.offs[(o.name, ls)]
                        + lv
                        - align_shift(layout.align_delta[(o.name, ls)], lv)
                    )
                target += addend
                w = int.from_bytes(image[site - base : site - base + 4], "little")
                assert w == target & 0xFFFFFFFF, f"patch for '{sym}' (R_LARCH_32) at {site:#x}"
            else:  # R_LARCH_64
                target = symaddr.get(sym)
                if target is None:
                    ls, lv = locs[o.name][sym]
                    target = (
                        layout.offs[(o.name, ls)]
                        + lv
                        - align_shift(layout.align_delta[(o.name, ls)], lv)
                    )
                target += addend
                w = int.from_bytes(image[site - base : site - base + 8], "little")
                assert w == target, f"patch for '{sym}' (R_LARCH_64) at {site:#x}"
    for sym, at in layout.plt.items():  # synthesized stubs
        p = at - base
        pcrel = layout.got[sym] - at
        hi = ((pcrel + 0x800) >> 12) & 0xFFFFF
        lo = pcrel & 0xFFF
        assert int.from_bytes(image[p : p + 4], "little") == OP_PCADDU12I | 0x0F | (hi << 5), (
            f"PLT stub for '{sym}' lost its pcaddu12i"
        )
        assert int.from_bytes(image[p + 4 : p + 8], "little") == OP_LD_D | 0x1EF | (lo << 10), (
            f"PLT stub for '{sym}' lost its ld.d"
        )
        assert int.from_bytes(image[p + 8 : p + 12], "little") == OP_JIRL | 0x1ED, (
            f"PLT stub for '{sym}' lost its jirl"
        )
    for sym, at, value in layout.slots:  # GOT slot contents
        got = int.from_bytes(image[at - base : at - base + 8], "little")
        assert got == value, f"GOT slot for '{sym}' at {at:#x}: {got:#x} != {value:#x}"


# --- the demo: three hand-written object files -----------------------------

OBJ_A = """\
# prog_a.obj: caller — _start loads magic (PCALA pair), calls compute
# DIRECTLY (bl, R_LARCH_B26) and helper THROUGH the GOT (pcalau12i + ld.d
# GOT_PC pair + jirl — the canonical indirect call), then branches off the
# end of the image (b image_end). image_end is a zero-size common symbol:
# commons are placed in name order, 'image_end' sorts after 'bonus', so
# its slot lands exactly at base + len(image), and the interpreter ends
# when PC >= len(image).
SEC .text
04 00 00 1a 84 00 c0 02 84 00 c0 28 84 14 c0 02
00 00 00 54 84 04 c0 02 0d 00 00 1a ad 01 c0 28
a1 01 00 4c 00 00 00 50
SYM _start G .text 0
SYM image_end C - 1 0
REL .text 0 PCALA_HI20 magic
REL .text 4 PCALA_LO12 magic
REL .text 4 RELAX
REL .text 0x10 B26 compute
REL .text 0x18 GOT_PC_HI20 helper
REL .text 0x1c GOT_PC_LO12 helper
REL .text 0x1c RELAX
REL .text 0x24 B26 image_end
"""
OBJ_B = """\
# prog_b.obj: compute — reads the common 'bonus' (zero-filled; tentative
# definition, like C 'int bonus;') into a0 and adds 2, then returns via
# jirl $r0, $ra, 0 (rd=0 -> plain jump, per Vol1). Its .text merges into
# the single output .text.
SEC .text
04 00 00 1a 84 00 c0 02 84 00 c0 28 84 08 c0 02
20 00 00 4c
SYM compute G .text 0
SYM bonus C - 8 8
REL .text 0 PCALA_HI20 bonus
REL .text 4 PCALA_LO12 bonus
REL .text 4 RELAX
"""
OBJ_C = """\
# prog_c.obj: helper (adds 3, returns) and the strong global magic — a
# 64-bit data word (ld.d is 8 bytes) — plus magic_ptr, an 8-byte pointer
# to magic patched by R_LARCH_64 (word64 S + A).
SEC .text
84 0c c0 02 20 00 00 4c
SEC .data
07 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
SYM helper G .text 0
SYM magic G .data 0
SYM magic_ptr G .data 8
REL .data 8 R_LARCH_64 magic
"""

ALIGN_TRACE_OBJ = """\
# align-demo.obj: one relaxable pair followed by .align 16, max 8 NOPs.
# Before the pair folds the ALIGN site is +8 (8 NOPs needed, within max);
# after the fold it is +4, alignment would need 12 > max, so the whole
# max-NOP run is deleted and the alignment is abandoned.
SEC .text
04 00 00 1a 84 00 c0 02
00 00 40 03 00 00 40 03 00 00 40 03
SYM x G .text 8
REL .text 0 PCALA_HI20 x
REL .text 4 PCALA_LO12 x
REL .text 4 RELAX
ALIGN .text 8 0x10 8
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv != ["--trace"]:
        raise SystemExit(f"unknown arguments: {' '.join(argv)}")
    trace = "--trace" in argv
    a, b, c = (
        parse_object(t, n)
        for t, n in ((OBJ_A, "prog_a.obj"), (OBJ_B, "prog_b.obj"), (OBJ_C, "prog_c.obj"))
    )

    print("== 1. symbol resolution ==")
    clash = parse_object("SEC .text\n20 00 c0 4c\nSYM compute G .text 0\n", "clash.obj")
    broken = parse_object("SEC .text\n20 00 c0 4c\nSYM missing G - 0\n", "broken.obj")
    for objs, label in [
        ([b, clash], "duplicate definition rejected"),
        ([broken], "undefined reference rejected"),
    ]:
        try:
            link(objs)
        except LinkError as e:
            print(f"  [ok] {label}: {e}")
        else:
            raise AssertionError(f"{label} accepted!")
    weak1 = parse_object("SEC .text\n20 00 c0 4c\nSYM w_even W .text 0\n", "weak1.obj")
    weak2 = parse_object("SEC .text\n20 00 c0 4c\nSYM w_even W .text 8\n", "weak2.obj")
    strong = parse_object("SEC .text\n20 00 c0 4c\nSYM w_even G .text 0\n", "strong.obj")
    _, ws, _ = link([weak1, weak2])
    print(f"  [ok] two weak definitions: first in link order wins, w_even -> {ws['w_even']:#x}")
    _, ss, _ = link([strong, weak1])
    print(f"  [ok] strong beats weak: no error, w_even -> {ss['w_even']:#x}")
    maybe = parse_object("SEC .text\n20 00 c0 4c\nSYM maybe W - 0\n", "maybe.obj")
    _, ms, _ = link([maybe])
    print(f"  [ok] undefined weak resolves to 0: maybe -> {ms['maybe']:#x}")
    far = parse_object(
        "SEC .text\n0c 00 00 1a\nSYM far G .text 0x200000000\nREL .text 0 PCALA_HI20 far\n",
        "far.obj",
    )
    try:
        link([far])
    except LinkError as e:
        print(f"  [ok] PCALA_HI20 overflow rejected: {e}")
    else:
        raise AssertionError("PCALA_HI20 overflow accepted!")
    odd = parse_object(
        "SEC .data\n00 00 00 00 00 00 00 00\nSYM odd G .data 2\nREL .data 0 B26 odd\n", "odd.obj"
    )
    try:
        link([odd])
    except LinkError as e:
        print(f"  [ok] B26 misalignment rejected: {e}")
    else:
        raise AssertionError("B26 misalignment accepted!")
    # GOT_PC pairs cannot overflow here: .got sits adjacent to the code
    # and the psABI's GOT offsets are 32-bit by construction.

    print("\n== 2. layout (3 objects, sections merged by name) ==")
    relax_log = [] if trace else None
    align_log = [] if trace else None
    image, symaddr, layout = link([a, b, c], relax_log=relax_log, align_log=align_log)
    for o in (a, b, c):
        for sec, data in o.sections.items():
            print(f"  {layout.offs[(o.name, sec)]:#08x}  {sec:<6} {len(data):>3} B  {o.name}")
    for sec, at, size in (
        (".plt", layout.plt_addr, 12 * len(layout.plt)),
        (".got", layout.got_addr, 8 * len(layout.got)),
        (".common", layout.common_addr, layout.common_size),
    ):
        print(f"  {at:#08x}  {sec:<6} {size:>3} B  <synthesized>")
    print(f"  image {BASE:#x}..{BASE + len(image):#x} ({len(image)} B)")
    print(
        "\n".join(
            f"  sym {symaddr[n]:#08x} {n:<10} {k} {sec if sec is not None else '-'} ({o.name})"
            for o in (a, b, c)
            for n, (k, sec, _, _) in o.symbols.items()
            if sec is not None or k == "C"
        )
    )

    print("\n== 3. relocations applied (PCALA / GOT_PC / B26 / R_LARCH_64) ==")
    for off in range(0, len(image), 16):
        print(f"  {BASE + off:#08x}: " + " ".join(f"{b:02x}" for b in image[off : off + 16]))
    for objn, sec, roff, kind, sym, site, target, field in layout.applied:
        if kind in ("PCALA_HI20", "GOT_PC_HI20"):
            print(
                f"  {objn} {sec}+{roff:#04x} {kind:<12} {sym:<8} "
                f"hi={field:#x} ({target:#x} vs PC {site:#x})  ok"
            )
        elif kind in ("PCALA_LO12", "GOT_PC_LO12"):
            print(
                f"  {objn} {sec}+{roff:#04x} {kind:<12} {sym:<8} "
                f"lo={field:#03x} (target {target:#x})  ok"
            )
        elif kind == "B26":
            print(
                f"  {objn} {sec}+{roff:#04x} {kind:<12} {sym:<8} "
                f"offs26={field:#x} ({target:#x} - {site:#x})  ok"
            )
        elif kind == "PCREL20_S2":
            print(
                f"  {objn} {sec}+{roff:#04x} {kind:<12} {sym:<8} "
                f"pcaddi -> {target:#x} (delta {target - site:#x})  ok"
            )
        elif kind == "R_LARCH_32":
            print(f"  {objn} {sec}+{roff:#04x} {kind:<12} {sym:<8} word32={field:#x}  ok")
        elif kind == "R_LARCH_64":
            print(f"  {objn} {sec}+{roff:#04x} {kind:<12} {sym:<8} word64={field:#x}  ok")
        else:  # PLT-STUB
            print(
                f"  {objn} {sec}+{roff:#04x} {kind:<12} {sym:<8} "
                f"stub -> slot {target:#x} (pcrel {field:#x})  ok"
            )
    print("  GOT slots (static link fills them at link time; a dynamic linker")
    print("  would apply R_LARCH_GLOB_DAT / R_LARCH_JUMP_SLOT here):")
    for sym, at, value in layout.slots:
        print(
            f"    [{at:#x}] {sym:<8} -> {value:#x}  (runtime addr = S - BASE;"
            f" S = {symaddr[sym]:#x})"
        )

    print("\n== 4. verification ==")
    verify(image, BASE, [a, b, c], layout, symaddr)
    print("  every patch (incl. synthesized .plt/.got) re-derived from the layout matches")
    result = larch_emu.run(image, symaddr["_start"] - BASE)
    assert result == 6, f"emulated result {result} != 6"
    print("  emulated _start (larch_emu): magic=7 -> +5 -> bl compute directly")
    print("    (+2, reads zero-filled common 'bonus') -> +1 -> helper through")
    print("    the GOT (+3) -> b image_end, PC runs off the image -> a0=6")
    print("  the emulated run agrees.")

    if trace and relax_log is not None:
        print("\n== 5. relaxation trace ==")
        if not relax_log:
            print("  no R_LARCH_RELAX candidate was considered")
        for s in relax_log:
            if s.decision == "fold":
                print(
                    f"  pass {s.pass_no}: FOLD {s.obj} {s.section}+{s.offset:#x} "
                    f"{s.kind} {s.sym} -> pcaddi to {s.target:#x} "
                    f"(delta {s.delta:#x})"
                )
            else:
                print(
                    f"  pass {s.pass_no}: SKIP {s.obj} {s.section}+{s.offset:#x} "
                    f"{s.kind} {s.sym}: {s.reason}"
                )

    if trace and align_log is not None:
        print("\n== 6. R_LARCH_ALIGN trace ==")
        if not align_log:
            print("  no R_LARCH_ALIGN request was considered")
        for a in align_log:
            how = "abandoned" if a.abandoned else "aligned"
            print(
                f"  pass {a.pass_no}: {how:9} {a.obj} {a.section}+{a.offset:#x} "
                f"align {a.align:#x} site {a.site:#x} -> keep {a.needed:#x} "
                f"NOP bytes, delete {a.removed:#x}"
            )

    if trace:
        print("\n== 7. ALIGN x pair-fold interaction ==")
        demo = parse_object(ALIGN_TRACE_OBJ, "align-demo.obj")
        demo_relax, demo_align = [], []
        demo_image, demo_sym, _ = link([demo], BASE, relax_log=demo_relax, align_log=demo_align)
        for s in demo_relax:
            print(
                f"  pass {s.pass_no}: FOLD {s.kind} {s.sym} (ALIGN site moves {s.offset:#x} -> 4)"
            )
        for a in demo_align:
            how = "abandoned" if a.abandoned else "aligned"
            print(
                f"  pass {a.pass_no}: {how:9} site {a.site:#x} "
                f"needs {a.needed:#x} NOPs, max {a.max_bytes:#x}, "
                f"delete {a.removed:#x}"
            )
        print(f"  final image {len(demo_image):#x} bytes; x -> {demo_sym['x']:#x} (section end)")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
