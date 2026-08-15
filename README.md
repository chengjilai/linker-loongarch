# linker-loongarch

A small LoongArch (LA64) toolchain written from scratch in pure Python —
stdlib only, no dependencies. Assembler → linker → emulator, with a
disassembler and real ELF64-loongarch object-file I/O around them.

```
assembly text ──► larch_asm ──► object ──► linker_loongarch ──► image ──► larch_emu
                                      ▲                                      │
                                 (real .o, via                       a0 = exit code
                                  elf_loongarch)                     (PC walks off)
                                      │
                                    larch_dis ◄── disassembly with symbols + relocs
```

## What each piece does

| module | role |
|---|---|
| `larch_asm.py` | two-pass assembler: labels, `%pcala_hi20/lo12` and `%got_pc_hi20/lo12` relocation suffixes, `R_LARCH_RELAX` markers on every relaxable pair |
| `linker_loongarch.py` | static linker: strong/weak/common resolution, section merging, synthesized `.plt`/`.got`/`.common`, the six relocations the toolchain emits (R_LARCH_64, PCALA_HI20/LO12, GOT_PC_HI20/LO12, B26, PCREL20_S2), `verify()` re-derives every patch, and **R_LARCH_RELAX relaxation** folds `pcalau12i`+`addi/ld` pairs to `pcaddi` at a fixpoint |
| `larch_emu.py` | 26-instruction LA64 interpreter (encodings cross-checked against QEMU's `insns.decode`), with a per-instruction tracer |
| `larch_dis.py` | linear-sweep disassembler on the interpreter's decoder; branch targets and relaxed `pcaddi` forms resolve to symbols |
| `elf_loongarch.py` | real ELF64-loongarch ET_REL writer + reader (recognized by `file`/`readelf`), including R_LARCH_RELAX (100) and R_LARCH_PCREL20_S2 (103) |

Everything is grounded in the primary sources, cited verbatim in the
module docstrings: the [LoongArch ELF ABI](https://loongson.github.io/LoongArch-Documentation/LoongArch-ELF-ABI-EN.html),
the [LoongArch ISA Volume 1](https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html),
[lld's `relaxPCHi20Lo12`](https://github.com/llvm/llvm-project/blob/main/lld/ELF/Arch/LoongArch.cpp)
(the relaxation transformation), binutils'
[`loongarch_make_plt_entry`](https://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=bfd/elfnn-loongarch.c)
(the PLT stub), the gABI, and Ian Lance Taylor's [Linkers](https://www.airs.com/blog/archives/38)
series.

## Demo

```
nix run            # link three hand-written objects, verify, emulate → a0 = 6
nix run .#asm      # assemble → link (relaxation on) → emulate; the objects are
                   # proven byte-identical to the hand-encoded references, and
                   # the relaxed image is 0x58 bytes vs 0x98 without relaxation
nix run .#dis      # disassemble the linked image: pcaddi r4, magic, bl compute, ...
nix run .#elf      # write the objects as real .o files, read them back, link,
                   # emulate — validated by file/readelf
nix run .#test     # 64 stdlib-unittest tests across the five modules
nix run .#lint     # ruff (strict ALL)
nix run .#typecheck  # ty (strict rules; modules are annotated)
nix develop        # python3, ruff, ty
```

No nix? `python3 -B larch_asm.py`, `python3 -B linker_loongarch.py`, the
test suites via `python3 -B -m unittest -v test_linker_loongarch test_larch_emu
test_larch_asm test_larch_dis test_elf_loongarch`.

## The object format

A text format, one directive or hex bytes per line (the assembler emits
it; the linker parses it):

```
SEC .text
04 00 00 1a 84 00 c0 02
SYM _start G .text 0
REL .text 0 PCALA_HI20 magic
REL .text 4 PCALA_LO12 magic
REL .text 4 RELAX
```

`elf_loongarch.py` converts these to and from real ELF64-loongarch
relocatable files.

## Honest limitations

- Instruction subset: the base integer + LA64 subset the assembler emits
  (no SIMD, no privileged instructions, no floats).
- `.align` inside code is fixed at assembly time; real toolchains use
  `R_LARCH_ALIGN` for alignment during relaxation.
- The linker reads the toy text format (or the bundled ELF reader), not
  arbitrary toolchain output; relaxation covers the canonical
  `pcalau12i`+`addi/ld` pairs lld relaxes.
