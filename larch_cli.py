#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""larch_cli: tiny as/ld/run commands for the LoongArch toy toolchain.

Commands:
  as   INPUT.s -o OUT[.obj|.o]     assemble toy/GNU-as-compatible .s
  ld   INPUT... -o IMAGE -e SYM    link text objects or real ELF .o files
  run  IMAGE -e OFFSET             emulate and print a0

The assembler and linker libraries already understand both the toy text
object format and the ELF subset produced by GNU as. This driver only
adds file dispatch and output handling.
"""

import argparse
import pathlib
import sys

import elf_loongarch
import larch_asm
import larch_emu
import linker_loongarch as ll


def _input_objects(paths: list[str]):
    """Load object files: .s -> assemble, .o -> ELF, else toy text."""
    objects = []
    for path in paths:
        p = pathlib.Path(path)
        name = p.name
        if p.suffix == ".s":
            text = larch_asm.assemble(p.read_text(), name)
            objects.append(ll.parse_object(text, name))
        elif p.suffix == ".o":
            objects.append(elf_loongarch.read_object(path))
        else:
            objects.append(ll.parse_object(p.read_text(), name))
    return objects


def _cmd_as(argv):
    parser = argparse.ArgumentParser(prog="larch as")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("-o", dest="output", type=pathlib.Path)
    args = parser.parse_args(argv)
    source = args.input.read_text()
    text = larch_asm.assemble(source, args.input.name)
    output = args.output or args.input.with_suffix(".obj")
    if output.suffix == ".o":
        elf_loongarch.write_object(ll.parse_object(text, output.name), str(output))
    else:
        output.write_text(text)
    print(f"wrote {output}")


def _cmd_ld(argv):
    parser = argparse.ArgumentParser(prog="larch ld")
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("-o", dest="output", type=pathlib.Path, default=pathlib.Path("a.bin"))
    parser.add_argument("-e", dest="entry", default="_start")
    parser.add_argument("--no-relax", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--print-syms", action="store_true")
    args = parser.parse_args(argv)
    objects = _input_objects([str(p) for p in args.inputs])
    image, symaddr, layout = ll.link(objects, ll.BASE, relax=not args.no_relax)
    ll.verify(image, ll.BASE, objects, layout, symaddr)
    if args.entry not in symaddr:
        raise SystemExit(f"entry symbol {args.entry!r} not found")
    args.output.write_bytes(image)
    print(f"wrote {args.output} ({len(image)} bytes)")
    if args.print_syms:
        for name, addr in sorted(symaddr.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"  {addr:#08x} {name}")
    if args.run:
        result = larch_emu.run(image, symaddr[args.entry] - ll.BASE)
        print(f"a0 = {result}")


def _cmd_run(argv):
    parser = argparse.ArgumentParser(prog="larch run")
    parser.add_argument("image", type=pathlib.Path)
    parser.add_argument("-e", dest="entry", type=lambda s: int(s, 0), default=0)
    args = parser.parse_args(argv)
    result = larch_emu.run(args.image.read_bytes(), args.entry)
    print(f"a0 = {result}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("as", "ld", "run"):
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    {"as": _cmd_as, "ld": _cmd_ld, "run": _cmd_run}[cmd](rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
