#!/usr/bin/env python3
"""larch_dis: a disassembler for the LoongArch toolchain, on top of
larch_emu.decode (which was parsed from the ISA Volume 1 and cross-checked
against qemu's insns.decode).

Disassembles a linked image word by word (linear sweep), rendering each
instruction in the assembler's syntax.  Branch targets resolve to symbol
names when the caller supplies the linker's symbol table; relocation
annotations (%pcala_hi20(sym) etc.) come from the linker's applied list,
so the disassembly reads like the original assembly.

    disasm(data, base=0, syms=None, annot=None) -> [line, ...]

    syms:  {address: name}   branch targets (b/bl/beq/bne/blt)
    annot: {address: (kind, sym)}  relocation annotations, from
           linker_loongarch.Layout.applied (site -> kind + symbol)

The demo links the three toolchain programs and disassembles the merged
.text with symbols and annotations: `nix run .#dis`.
"""

import larch_asm
import larch_emu
import linker_loongarch as ll

SUFFIX = {  # reloc kind -> assembler suffix
    "PCALA_HI20": "pcala_hi20",
    "PCALA_LO12": "pcala_lo12",
    "GOT_PC_HI20": "got_pc_hi20",
    "GOT_PC_LO12": "got_pc_lo12",
}
# which instruction families accept a %suffix annotation per operand slot
_HI20_OPS = frozenset(("lu12i.w", "pcalau12i"))
_LO12_OPS = frozenset(("addi.d", "addi.w", "ld.d", "ld.w", "st.d", "st.w"))


def _reg(i: int) -> str:
    return f"r{i}"


def disasm(
    data: bytes,
    base: int = 0,
    syms: dict[int, str] | None = None,
    annot: dict[int, tuple[str, str]] | None = None,
) -> list[str]:
    """Linear-sweep disassembly of `data` (bytes) into text lines."""
    syms = syms or {}
    annot = annot or {}
    out = []
    for off in range(0, len(data) - 3, 4):
        word = int.from_bytes(data[off : off + 4], "little")
        d = larch_emu.decode(word)
        addr = base + off
        if d is None:
            out.append(f"{addr:#08x}: {word:08x}  .word 0x{word:x}")
            continue
        mnem, rd, rj, rk, imm = d
        if mnem in ("b", "bl"):
            target = addr + imm
            name = syms.get(target)
            op = name or f"{target:#x}"
        elif mnem in ("beq", "bne", "blt"):
            target = addr + imm
            name = syms.get(target)
            op = f"{_reg(rj)}, {_reg(rd)}, " + (name or f"{target:#x}")
        elif mnem == "jirl":
            op = f"{_reg(rd)}, {_reg(rj)}, {imm:#x}"
        elif mnem in _HI20_OPS:
            a = annot.get(addr)
            if a and a[0] in SUFFIX:
                op = f"{_reg(rd)}, %{SUFFIX[a[0]]}({a[1]})"
            else:
                op = f"{_reg(rd)}, {imm:#x}"
        elif mnem == "pcaddi":  # the relaxed pair form
            target = addr + (imm << 2)
            op = f"{_reg(rd)}, " + (syms.get(target) or f"{target:#x}")
        elif mnem in _LO12_OPS:
            a = annot.get(addr)
            if a and a[0] in SUFFIX:
                op = f"{_reg(rd)}, {_reg(rj)}, %{SUFFIX[a[0]]}({a[1]})"
            else:
                op = f"{_reg(rd)}, {_reg(rj)}, {imm}"
        elif mnem in ("add.d", "sub.d", "slt", "sltu", "and", "or", "xor"):
            op = f"{_reg(rd)}, {_reg(rj)}, {_reg(rk)}"
        else:  # remaining 10-bit ops
            op = f"{_reg(rd)}, {_reg(rj)}, {imm}"
        out.append(f"{addr:#08x}: {word:08x}  {mnem:<10} {op}")
    return out


def main() -> int:

    print("== assemble -> link -> disassemble (with symbols + relocs) ==")
    objs = []
    for src, name in (
        (larch_asm.PROG_A, "prog_a.obj"),
        (larch_asm.PROG_B, "prog_b.obj"),
        (larch_asm.PROG_C, "prog_c.obj"),
    ):
        objs.append(ll.parse_object(larch_asm.assemble(src, name), name))
    image, symaddr, layout = ll.link(objs, ll.BASE)
    ll.verify(image, ll.BASE, objs, layout, symaddr)

    syms = {addr: n for n, addr in symaddr.items()}
    annot = {site: (kind, sym) for _, _, _, kind, sym, site, _, _ in layout.applied}
    lines = disasm(image, ll.BASE, syms, annot)

    for ln in lines:
        print("  " + ln)

    # sanity checks, anchored by address: the disassembly must read like
    # the original assembly — with the relaxable pairs FOLDED to pcaddi by
    # the linker (the magic/helper/bonus pairs all fit pcaddi's range)
    by_addr = {ln.split(":")[0]: ln for ln in lines}
    first = by_addr[f"{ll.BASE:#08x}"]
    assert "pcaddi" in first
    assert "magic" in first
    assert "compute" in by_addr[f"{ll.BASE + 0xC:#08x}"]
    assert "helper" in by_addr[f"{ll.BASE + 0x14:#08x}"]
    assert "image_end" in by_addr[f"{ll.BASE + 0x1C:#08x}"]
    assert "jirl" in by_addr[f"{ll.BASE + 0x18:#08x}"]
    print("\n  the disassembly resolves symbols and relocations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
