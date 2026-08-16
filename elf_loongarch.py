#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""elf_loongarch: real ELF64-loongarch relocatable object (.o) I/O, stdlib only.

Writes and reads REAL ET_REL ELF files for EM_LOONGARCH (258) — the same
format `gcc -c` / `ld -r` produce — so the toolchain's objects are real
ELF artifacts (validated structurally with `readelf -h -S -r`), not just
the toy text format.  The read side maps a .o back to the linker's
internal Object (sections, symbols, relocations), so the pipeline is:

    assembly -> (toy .obj) -> write .o -> read .o -> link -> emulate

and the read-back objects must link and emulate identically (a0 = 6).

Grounding (LoongArch ELF ABI, loongson.github.io):
  * EI_CLASS/e_flags[7:0] identify the ABI: "The ABI type of an ELF object
    is uniquely identified by EI_CLASS and e_flags[7:0] in its header";
    e_flags bits 7-6 = ABI version, bits 2-0 = base ABI (LP64D = 0x3),
    bit 6 set = OBJABI v1.  The spec's "ABI-related bits in e_flags"
    table: Bit 31-8 reserved | Bit 7-6 ABI version | Bit 5-3 ABI
    extension | Bit 2-0 Base ABI modifier.
  * Relocation type numbers (same source, table 2-17): R_LARCH_64 = 2,
    R_LARCH_B26 = 66, R_LARCH_PCALA_HI20 = 71, R_LARCH_PCALA_LO12 = 72,
    R_LARCH_GOT_PC_HI20 = 75, R_LARCH_GOT_PC_LO12 = 76.
  * gABI ch.4-5 for the ELF64 header / section / symbol / RELA layouts
    (SHT_RELA with r_info = (sym << 32) | type, explicit r_addend).

Limitations (honest): the writer emits the subset a toy compiler needs —
SHT_PROGBITS .text/.data, SHT_SYMTAB/.strtab, SHT_RELA per relocatable
section, section symbols, SHN_COMMON handling; no .rela.* for .symtab
itself, no debug sections, no .eh_frame.  The reader also accepts the
GNU as subset used by the committed fixture: SHT_NOBITS sections, non-zero
RELA addends, R_LARCH_32, R_LARCH_RELAX, and R_LARCH_ALIGN. Other real
sections are ignored; TLS/extreme-code-model relocations are not decoded.
"""

import pathlib
import struct
import subprocess
import tempfile

import larch_asm
import larch_emu
from linker_loongarch import BASE, Align, LinkError, Object, link, parse_object, verify

# ---------------------------------------------------------------------------
# ELF64 constants
# ---------------------------------------------------------------------------

EM_LOONGARCH = 258
EF_LARCH_ABI_LP64D = 0x3
EF_LARCH_OBJABI_V1 = 0x40

SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_NOBITS = 8

SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4

SHN_UNDEF = 0
SHN_COMMON = 0xFFF2
SHN_ABS = 0xFFF1

STB_LOCAL = 0
STB_GLOBAL = 1
STB_WEAK = 2
STT_NOTYPE = 0
STT_OBJECT = 1
STT_FUNC = 2
STT_SECTION = 3
STT_COMMON = 10

RELOC_NUM = {  # toy kind -> R_LARCH_* number (ABI table 2-17; verified)
    "R_LARCH_32": 1,
    "R_LARCH_64": 2,
    "B16": 64,
    "B26": 66,
    "ABS_HI20": 67,
    "ABS_LO12": 68,
    "PCALA_HI20": 71,
    "PCALA_LO12": 72,
    "GOT_PC_HI20": 75,
    "GOT_PC_LO12": 76,
    "RELAX": 100,
    "R_LARCH_ALIGN": 102,
    "PCREL20_S2": 103,
}
RELOC_NAME = {v: k for k, v in RELOC_NUM.items()}

EHDR = struct.Struct("<16sHHIQQQIHHHHHH")  # e_ident..e_shstrndx
SHDR = struct.Struct("<IIQQQQIIQQ")  # sh_name..sh_entsize
SYM = struct.Struct("<IBBHQQ")  # st_name, st_info, st_other, st_shndx, st_value, st_size
RELA = struct.Struct("<QQQ")  # r_offset, r_info, r_addend

IDENT = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\x00" * 8  # class64, data2lsb, version1, osabi0


class ElfError(Exception):
    pass


# ---------------------------------------------------------------------------
# writer: Object -> real .o file
# ---------------------------------------------------------------------------


def write_object(obj: Object, path: str) -> None:
    """Write an Object (toy format) as a real ELF64-loongarch .o file."""
    # --- sections: .text, .data first, then symtab/strtab/shstrtab/rela ---
    secs = [(n, d) for n, d in obj.sections.items()]
    names = [
        "",
        *(n for n, _ in secs),
        ".symtab",
        ".strtab",
        ".shstrtab",
        *(f".rela{n}" for n, _ in secs),
    ]
    shstr = b"\x00".join(n.encode() for n in names) + b"\x00"
    shstr_off = {n: shstr.index(n.encode() + b"\x00") for n in names}

    # --- symbols -----------------------------------------------------------
    # like a real assembler: section symbols first, then defined/undefined
    # object symbols, then any symbol a relocation references that is not
    # yet in the table (an external ref -> STB_GLOBAL, SHN_UNDEF).
    syms = [(None, 0, 0, 0, 0, 0)]  # symbol 0: the required null symbol
    for secname, _ in secs:  # STT_SECTION symbols, like real .o files
        syms.append((secname, STB_LOCAL, STT_SECTION, len(syms), 0, 0))
    sym_idx = {}
    for name, (kind, sec, val, size) in obj.symbols.items():
        if kind == "C":
            syms.append((name, STB_GLOBAL, STT_COMMON, SHN_COMMON, val, size))
        elif sec is None:
            bind = STB_WEAK if kind == "W" else STB_GLOBAL
            syms.append((name, bind, STT_NOTYPE, SHN_UNDEF, 0, 0))
        else:
            bind = {"G": STB_GLOBAL, "W": STB_WEAK, "L": STB_LOCAL}[kind]
            shndx = [n for n, _ in secs].index(sec) + 1
            syms.append((name, bind, STT_NOTYPE, shndx, val, size))
        sym_idx[name] = len(syms) - 1  # index of the symbol just added
    for _, _, _, sym, _ in obj.relocs:  # external refs -> UNDEF
        if sym is None:  # RELAX marker: no symbol
            continue
        if sym not in sym_idx:
            syms.append((sym, STB_GLOBAL, STT_NOTYPE, SHN_UNDEF, 0, 0))
            sym_idx[sym] = len(syms) - 1
    strtab = b"\x00"
    stroffs = []  # parallel to syms: stroffs[i] = strtab offset of syms[i]
    for s in syms:
        nm = s[0].encode() + b"\x00" if s[0] else b""
        off = strtab.find(nm)
        if off < 0:
            off = len(strtab)
            strtab += nm
        stroffs.append(off)
    first_global = next((i for i, s in enumerate(syms) if s[1] == STB_GLOBAL), len(syms))

    # --- relocations --------------------------------------------------------
    relas = {}  # secname -> list of RELA entries
    for sec, roff, kind, sym, addend in obj.relocs:
        if kind == "RELAX":
            relas.setdefault(sec, []).append(RELA.pack(roff, 100, 0))
            continue
        if sym not in sym_idx:
            raise LinkError(f"write_object: relocation symbol '{sym}' not in symtab")
        r = RELOC_NUM.get(kind)
        if r is None:
            raise LinkError(f"write_object: no ELF reloc number for {kind}")
        relas.setdefault(sec, []).append(RELA.pack(roff, (sym_idx[sym] << 32) | r, addend))
    for al in obj.aligns:
        # r_symndx == 0: addend = alignment - 4 (binutils' no-max form).
        # r_symndx  > 0: low byte = log2(alignment), high bytes = max NOPs.
        if al.max_bytes == 0:
            sym, addend = 0, al.align - 4
        else:
            sec_idx = next(i + 1 for i, (n, _) in enumerate(secs) if n == al.section)
            sym, addend = sec_idx, (al.max_bytes << 8) | (al.align.bit_length() - 1)
        relas.setdefault(al.section, []).append(
            RELA.pack(al.offset, (sym << 32) | RELOC_NUM["R_LARCH_ALIGN"], addend)
        )
    # --- layout --------------------------------------------------------------
    shdrs = []
    offset = EHDR.size
    data_blobs = []
    for secname, data in secs:
        align = 4 if secname == ".text" else 8
        offset = (offset + align - 1) & ~(align - 1)
        flags = SHF_ALLOC | (SHF_EXECINSTR if secname == ".text" else SHF_WRITE)
        shdrs.append(
            (shstr_off[secname], SHT_PROGBITS, flags, 0, offset, len(data), 0, 0, align, 0)
        )
        data_blobs.append((offset, data))
        offset += len(data)
    # .symtab
    symtab_data = b"".join(
        SYM.pack(stroffs[i], (s[1] << 4) | s[2], 0, s[3], s[4], s[5]) for i, s in enumerate(syms)
    )
    offset = (offset + 7) & ~7
    symtab_off = offset
    shdrs.append(
        (
            shstr_off[".symtab"],
            SHT_SYMTAB,
            0,
            0,
            offset,
            len(symtab_data),
            len(shdrs) + 2,
            first_global,
            8,
            SYM.size,
        )
    )
    offset += len(symtab_data)
    # .strtab
    offset = (offset + 7) & ~7
    strtab_off = offset
    shdrs.append((shstr_off[".strtab"], SHT_STRTAB, 0, 0, offset, len(strtab), 0, 0, 1, 0))
    offset += len(strtab)
    # .shstrtab
    offset = (offset + 7) & ~7
    shstrtab_off = offset
    shdrs.append((shstr_off[".shstrtab"], SHT_STRTAB, 0, 0, offset, len(shstr), 0, 0, 1, 0))
    offset += len(shstr)
    # .rela.*
    rela_sections = {}
    for secname, entries in relas.items():
        rela_data = b"".join(entries)
        offset = (offset + 7) & ~7
        rela_sections[secname] = (offset, rela_data)
        target = next(i + 1 for i, (n, _) in enumerate(secs) if n == secname)
        shdrs.append(
            (
                shstr_off[f".rela{secname}"],
                SHT_RELA,
                0,
                0,
                offset,
                len(rela_data),
                len(shdrs) + 1,
                target,
                8,
                RELA.size,
            )
        )
        offset += len(rela_data)
    # section header table
    shoff = offset
    shnum = len(shdrs) + 1
    shstrndx = names.index(".shstrtab")
    # sh_link values: symtab->strtab, rela->symtab (set by index position)
    # (fix up: symtab's sh_link is the strtab's index, rela's sh_link the
    #  symtab's index; computed from shdrs order below)
    e_ident = IDENT
    e_flags = EF_LARCH_ABI_LP64D | EF_LARCH_OBJABI_V1
    header = EHDR.pack(
        e_ident,
        1,
        EM_LOONGARCH,
        1,
        0,
        0,
        shoff,
        e_flags,
        EHDR.size,
        0,
        0,
        SHDR.size,
        shnum,
        shstrndx,
    )

    with open(path, "wb") as f:
        f.write(header)
        for off, blob in data_blobs:
            assert f.tell() <= off
            f.write(b"\x00" * (off - f.tell()))
            f.write(blob)
        for off, blob in [
            (symtab_off, symtab_data),
            (strtab_off, strtab),
            (shstrtab_off, shstr),
            *[(o, d) for o, d in rela_sections.values()],
        ]:
            assert f.tell() <= off
            f.write(b"\x00" * (off - f.tell()))
            f.write(blob)
        # section headers: index 0 = NULL
        f.write(b"\x00" * SHDR.size)
        strtab_idx = names.index(".strtab")
        symtab_idx = names.index(".symtab")
        for sh in shdrs:
            name, stype, _flags, _addr, off, size, link, info, align, entsize = sh
            if stype == SHT_SYMTAB:
                link = strtab_idx
            elif stype == SHT_RELA:
                link = symtab_idx
            f.write(SHDR.pack(name, stype, 0, 0, off, size, link, info, align, entsize))


# ---------------------------------------------------------------------------
# reader: real .o -> Object
# ---------------------------------------------------------------------------


def read_object(path: str) -> Object:
    """Parse a real ELF64-loongarch ET_REL file back into an Object."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < EHDR.size or data[:4] != b"\x7fELF":
        raise ElfError(f"{path}: not an ELF file")
    (
        ident,
        etype,
        emachine,
        _ver,
        _entry,
        _phoff,
        shoff,
        _eflags,
        _ehsize,
        _phentsize,
        _phnum,
        shentsize,
        shnum,
        shstrndx,
    ) = EHDR.unpack_from(data)
    if emachine != EM_LOONGARCH:
        raise ElfError(f"{path}: e_machine {emachine} != EM_LOONGARCH (258)")
    if etype != 1:
        raise ElfError(f"{path}: e_type {etype} != ET_REL (1)")
    if ident[4] != 2 or ident[5] != 1:
        raise ElfError(f"{path}: not ELF64 little-endian")

    shdrs = [SHDR.unpack_from(data, shoff + i * shentsize) for i in range(shnum)]

    shstr = data[shdrs[shstrndx][4] : shdrs[shstrndx][4] + shdrs[shstrndx][5]]

    def shname(i):
        off = shdrs[i][0]
        return shstr[off : shstr.index(b"\x00", off)].decode()

    strtab_data = None
    symtab = None
    for i, shdr in enumerate(shdrs):
        if shdr[1] == SHT_STRTAB and shname(i) == ".strtab":
            strtab_data = data[shdr[4] : shdr[4] + shdr[5]]
        if shdr[1] == SHT_SYMTAB:
            symtab = (i, shdr[4], shdr[5], shdr[6], shdr[7], shdr[9])
    if symtab is None or strtab_data is None:
        raise ElfError(f"{path}: no .symtab/.strtab")

    _symtab_idx, symoff, symsize, _strlink, _first_global, entsize = symtab
    nsyms = symsize // (entsize or SYM.size)
    syms = [SYM.unpack_from(data, symoff + i * SYM.size) for i in range(nsyms)]

    def sym_name(st_name):
        return strtab_data[st_name : strtab_data.index(b"\x00", st_name)].decode()

    # sections (PROGBITS carry bytes; NOBITS become zero-filled toy sections)
    sections = {}
    order = []
    for i, shdr in enumerate(shdrs):
        if shdr[1] == SHT_PROGBITS:
            sections[shname(i)] = bytearray(data[shdr[4] : shdr[4] + shdr[5]])
            order.append(shname(i))
        elif shdr[1] == SHT_NOBITS:
            sections[shname(i)] = bytearray(shdr[5])
            order.append(shname(i))

    # symbols -> Object symbols
    obj_syms = {}
    for st_name, st_info, _st_other, st_shndx, st_value, st_size in syms:
        name = sym_name(st_name)
        if not name:
            continue
        bind, typ = st_info >> 4, st_info & 0xF
        if typ == STT_SECTION:
            continue
        if st_shndx == SHN_COMMON:
            obj_syms[name] = ("C", None, st_value, st_size)
        elif st_shndx == SHN_UNDEF:
            obj_syms[name] = ("W" if bind == STB_WEAK else "G", None, 0, 0)
        else:
            kind = "L" if bind == STB_LOCAL else ("W" if bind == STB_WEAK else "G")
            obj_syms[name] = (kind, shname(st_shndx), st_value, st_size)

    # relocations and R_LARCH_ALIGN requests
    relocs = []
    aligns = []
    for shdr in shdrs:
        if shdr[1] != SHT_RELA:
            continue
        secname = shname(shdr[7])
        n = shdr[5] // RELA.size
        for j in range(n):
            r_offset, r_info, r_addend = RELA.unpack_from(data, shdr[4] + j * RELA.size)
            kind = RELOC_NAME.get(r_info & 0xFFFFFFFF)
            if kind is None:
                raise ElfError(f"{path}: unsupported relocation type {r_info & 0xFFFFFFFF:#x}")
            si = (r_info >> 32) & 0xFFFFFFFF
            if kind == "R_LARCH_ALIGN":
                if si == 0:
                    align, max_bytes = r_addend + 4, 0
                else:
                    align, max_bytes = 1 << (r_addend & 0xFF), r_addend >> 8
                if align < 4 or align & (align - 1) or r_offset % 4:
                    raise ElfError(f"{path}: bad R_LARCH_ALIGN at {secname}+{r_offset:#x}")
                aligns.append(Align(secname, r_offset, align, max_bytes))
                continue
            sym = None if (kind == "RELAX" or si == 0) else sym_name(syms[si][0])
            relocs.append((secname, r_offset, kind, sym, r_addend))

    return Object(path, sections, obj_syms, relocs, aligns)


# ---------------------------------------------------------------------------
# demo: write real .o files, read them back, link, emulate
# ---------------------------------------------------------------------------


def main():
    print("== the toolchain's objects as REAL ELF64-loongarch .o files ==")
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        objs = []
        for src, name in (
            (larch_asm.PROG_A, "prog_a"),
            (larch_asm.PROG_B, "prog_b"),
            (larch_asm.PROG_C, "prog_c"),
        ):
            toy = larch_asm.assemble(src, f"{name}.obj")
            obj = parse_object(toy, f"{name}.obj")
            path = d / f"{name}.o"
            write_object(obj, str(path))
            flags = EF_LARCH_ABI_LP64D | EF_LARCH_OBJABI_V1
            print(
                f"  wrote {path} ({path.stat().st_size} bytes, "
                f"EM_LOONGARCH={EM_LOONGARCH}, e_flags={flags:#x})"
            )
            read_back = read_object(str(path))
            objs.append(read_back)
            # read-back must equal the input object structurally — with one
            # intentional superset: the writer adds an UNDEF symbol for every
            # relocation target external to this object (as real assemblers
            # do), so the read-back symbol table carries those too.
            expected_syms = dict(obj.symbols)
            for _, _, _, sym, _ in obj.relocs:
                if sym is not None:  # RELAX markers carry no symbol
                    expected_syms.setdefault(sym, ("G", None, 0, 0))
            assert read_back.sections == obj.sections, f"{name}: sections changed"
            assert read_back.symbols == expected_syms, f"{name}: symbols changed"
            assert read_back.relocs == obj.relocs, f"{name}: relocs changed"
            assert read_back.aligns == obj.aligns, f"{name}: aligns changed"
            print("  read-back identical to the assembled object (+ UNDEF externals)")

        # validate with the real ELF tools when available
        for tool, args in (("readelf", ["-h", "-S", "-r"]), ("file", [])):
            exe = pathlib.Path("/run/current-system/sw/bin") / tool
            if not exe.exists():
                continue
            r = subprocess.run(
                [str(exe), *args, str(d / "prog_a.o")], capture_output=True, text=True, check=False
            )
            out = r.stdout + r.stderr
            assert "EM_LOONGARCH" in out or "LoongArch" in out or "258" in out, (
                f"{tool} did not recognize the machine"
            )
            line = next(
                ln for ln in out.splitlines() if "arch" in ln.lower() or "machine" in ln.lower()
            )
            print(f"  {tool}: {line}")

        print("\n== link + emulate the READ-BACK objects ==")
        image, symaddr, layout = link(objs, BASE)
        verify(image, BASE, objs, layout, symaddr)
        result = larch_emu.run(image, symaddr["_start"] - BASE)
        print(f"  emulated _start: a0 = {result}")
        assert result == 6
        print("  the ELF .o round-trip agrees (write -> read -> link -> emulate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
