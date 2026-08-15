# SPDX-License-Identifier: Apache-2.0
"""test_elf_loongarch.py — unittest suite for the ELF64-loongarch .o I/O.

Writer/reader round-trip for all three toolchain objects, ELF header
fields (machine/type/flags), readelf validation when available, error
paths, and the end-to-end read-back link + emulate.
"""

import contextlib
import importlib.util
import io
import os
import struct
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("elf_loongarch", "elf_loongarch.py")
assert spec is not None and spec.loader is not None
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
spec = importlib.util.spec_from_file_location("larch_asm", "larch_asm.py")
assert spec is not None and spec.loader is not None
asm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asm)
spec = importlib.util.spec_from_file_location("larch_emu", "larch_emu.py")
assert spec is not None and spec.loader is not None
emu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emu)

READELF = "/run/current-system/sw/bin/readelf"


def write_all(d):
    """Assemble the three programs and write their .o files into dir d."""
    paths = []
    for src, name in ((asm.PROG_A, "prog_a"), (asm.PROG_B, "prog_b"), (asm.PROG_C, "prog_c")):
        obj = E.parse_object(asm.assemble(src, f"{name}.obj"), f"{name}.obj")
        p = os.path.join(d, f"{name}.o")
        E.write_object(obj, p)
        paths.append(p)
    return paths


class TestHeader(unittest.TestCase):
    def test_header_fields(self):
        with tempfile.TemporaryDirectory() as d:
            (p,) = write_all(d)[:1]
            data = open(p, "rb").read()
            self.assertEqual(data[:4], b"\x7fELF")
            (
                ident,
                etype,
                emachine,
                _v,
                _entry,
                _phoff,
                _shoff,
                eflags,
                _ehsize,
                _phentsize,
                _phnum,
                _shentsize,
                _shnum,
                _shstrndx,
            ) = E.EHDR.unpack_from(data)
            self.assertEqual(ident[4], 2)  # ELFCLASS64
            self.assertEqual(ident[5], 1)  # ELFDATA2LSB
            self.assertEqual(etype, 1)  # ET_REL
            self.assertEqual(emachine, 258)  # EM_LOONGARCH
            self.assertEqual(eflags, 0x43)  # LP64D | OBJ-v1
            self.assertEqual(struct.unpack_from("<H", data, 56)[0], 0)  # e_phnum

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            for src, name in (
                (asm.PROG_A, "prog_a"),
                (asm.PROG_B, "prog_b"),
                (asm.PROG_C, "prog_c"),
            ):
                obj = E.parse_object(asm.assemble(src, f"{name}.obj"), f"{name}.obj")
                p = os.path.join(d, f"{name}.o")
                E.write_object(obj, p)
                rb = E.read_object(p)
                expected = dict(obj.symbols)
                for _, _, _, sym in obj.relocs:
                    if sym is not None:  # RELAX markers carry no symbol
                        expected.setdefault(sym, ("G", None, 0, 0))
                self.assertEqual(rb.sections, obj.sections)
                self.assertEqual(rb.symbols, expected)
                self.assertEqual(rb.relocs, obj.relocs)

    def test_readelf_validation(self):
        if not os.path.exists(READELF):
            self.skipTest("readelf not available")
        with tempfile.TemporaryDirectory() as d:
            (p,) = write_all(d)[:1]
            r = subprocess.run([READELF, "-h", p], capture_output=True, text=True)
            self.assertIn("LoongArch", r.stdout)
            self.assertIn("DOUBLE-FLOAT", r.stdout)
            r = subprocess.run([READELF, "-r", p], capture_output=True, text=True)
            # readelf truncates long type names to the column width
            for t in ("PCALA_HI", "PCALA_LO", "R_LARCH_B26", "GOT_PC_HI"):
                self.assertIn(t, r.stdout)


class TestErrors(unittest.TestCase):
    def test_not_elf(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.o")
            open(p, "wb").write(b"not an elf file at all")
            with self.assertRaises(E.ElfError):
                E.read_object(p)

    def test_wrong_machine(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.o")
            data = bytearray(E.EHDR.size)
            data[:4] = b"\x7fELF"
            data[4:6] = bytes([2, 1])
            data[18:20] = (62).to_bytes(2, "little")  # EM_X86_64
            data[16:18] = (1).to_bytes(2, "little")
            open(p, "wb").write(data)
            with self.assertRaises(E.ElfError):
                E.read_object(p)

    def test_not_relocatable(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.o")
            data = bytearray(E.EHDR.size)
            data[:4] = b"\x7fELF"
            data[4:6] = bytes([2, 1])
            data[16:18] = (2).to_bytes(2, "little")  # ET_EXEC
            data[18:20] = (258).to_bytes(2, "little")
            open(p, "wb").write(data)
            with self.assertRaises(E.ElfError):
                E.read_object(p)


class TestEndToEnd(unittest.TestCase):
    def test_read_back_link_emulate(self):
        with tempfile.TemporaryDirectory() as d:
            objs = [E.read_object(p) for p in write_all(d)]
            import linker_loongarch as ll

            image, symaddr, layout = ll.link(objs, ll.BASE)
            ll.verify(image, ll.BASE, objs, layout, symaddr)
            result = emu.run(image, symaddr["_start"] - ll.BASE)
            self.assertEqual(result, 6)

    def test_demo(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(E.main(), 0)
        self.assertIn("the ELF .o round-trip agrees", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
