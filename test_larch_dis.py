"""test_larch_dis.py — unittest suite for the LoongArch disassembler.

Formatting per instruction family, branch-target symbol resolution,
relocation annotation rendering, unknown-word handling, and the demo's
asserted output lines.
"""

import contextlib
import io
import importlib.util
import unittest

spec = importlib.util.spec_from_file_location("larch_dis", "larch_dis.py")
assert spec is not None and spec.loader is not None
dis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dis)
spec = importlib.util.spec_from_file_location("larch_asm", "larch_asm.py")
assert spec is not None and spec.loader is not None
asm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asm)


def words(*words):
    return b"".join(w.to_bytes(4, "little") for w in words)


class TestFormatting(unittest.TestCase):
    def test_families(self):
        # addi.d r4, r4, 5 ; add.d r4, r5, r6 ; lu12i.w r12, 0x12345
        lines = dis.disasm(words(0x02C01484, 0x1098A4, 0x142468AC))
        self.assertIn("addi.d     r4, r4, 5", lines[0])
        self.assertIn("add.d      r4, r5, r6", lines[1])
        self.assertIn("lu12i.w    r12, 0x12345", lines[2])

    def test_unknown_word(self):
        lines = dis.disasm(words(0xDEADBEEF))
        self.assertIn(".word 0xdeadbeef", lines[0])

    def test_branch_symbols(self):
        # bl +4 (to 0x400004); beq r4, r5, +8 (to 0x400008)
        # bl +4 (offs26 field 1 at bits 10-25); beq r4,r5,+8 (offs16 field 2)
        # bl +4 (offs26 field 1 at bits 10-25); beq r4, r5, +8 (rj=4, rd=5)
        prog = words(0x54000000 | (1 << 10), 0x58000000 | (2 << 10) | (4 << 5) | 5)
        syms = {0x400004: "next", 0x40000C: "loop"}
        lines = dis.disasm(prog, base=0x400000, syms=syms)
        self.assertIn("bl         next", lines[0])
        self.assertIn("beq        r4, r5, loop", lines[1])

    def test_reloc_annotations(self):
        prog = words(0x1A00000C, 0x28C00184, 0x1A00000D, 0x28C001AD)
        annot = {
            0x400000: ("PCALA_HI20", "magic"),
            0x400004: ("PCALA_LO12", "magic"),
            0x400008: ("GOT_PC_HI20", "helper"),
            0x40000C: ("GOT_PC_LO12", "helper"),
        }
        lines = dis.disasm(prog, base=0x400000, annot=annot)
        self.assertIn("%pcala_hi20(magic)", lines[0])
        self.assertIn("%pcala_lo12(magic)", lines[1])
        self.assertIn("%got_pc_hi20(helper)", lines[2])
        self.assertIn("%got_pc_lo12(helper)", lines[3])


class TestDemo(unittest.TestCase):
    def test_demo_output(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(dis.main(), 0)
        out = buf.getvalue()
        for want in (
            "bl         compute",
            "b          image_end",
            "pcaddi     r4, magic",
            "pcaddi     r13, helper",
            "pcaddi     r4, bonus",
            "jirl       r1, r13, 0x0",
        ):
            self.assertIn(want, out)


if __name__ == "__main__":
    unittest.main()
