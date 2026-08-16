# SPDX-License-Identifier: Apache-2.0
"""test_larch_asm.py — unittest suite for the LoongArch assembler.

Every encoder is round-tripped through larch_emu.decode (an independent,
QEMU-cross-checked decoder); label/relocation emission and error paths
are checked against the linker's object format; the demo's byte-identity
claim (assembled == hand-encoded objects) and the end-to-end run are
asserted.
"""

import importlib.util
import unittest

spec = importlib.util.spec_from_file_location("larch_asm", "larch_asm.py")
assert spec is not None and spec.loader is not None
asm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asm)
spec = importlib.util.spec_from_file_location("larch_emu", "larch_emu.py")
assert spec is not None and spec.loader is not None
emu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emu)


def enc(mnem, ops):
    """Encode one instruction (operands as a comma string); return bytes+rels."""
    word, rels = asm.encode_instruction(mnem, [o.strip() for o in ops.split(",")], "test")
    return word, rels


class TestRoundTrip(unittest.TestCase):
    """encode -> decode must come back identical (encoder vs decoder)."""

    CASES = [
        # (mnemonic, ops, expected decode tuple (m, rd, rj, rk, imm))
        # NB: for 10-bit-opcode instructions the decoder reports the
        # immediate's top 5 bits in the rk slot (rk = imm & 0x1F).
        ("addi.d", "r4, r4, 5", ("addi.d", 4, 4, 5, 5)),
        ("addi.d", "sp, sp, -16", ("addi.d", 3, 3, 16, -16)),
        ("addi.w", "r7, r8, 0x7ff", ("addi.w", 7, 8, 31, 0x7FF)),
        ("add.d", "r4, r5, r6", ("add.d", 4, 5, 6, 0)),
        ("sub.d", "r9, r9, r1", ("sub.d", 9, 9, 1, 0)),
        ("and", "r10, r11, r12", ("and", 10, 11, 12, 0)),
        ("or", "r4, r4, r20", ("or", 4, 4, 20, 0)),
        ("xor", "r4, r4, r4", ("xor", 4, 4, 4, 0)),
        ("slt", "r1, r2, r3", ("slt", 1, 2, 3, 0)),
        ("sltu", "r1, r2, r3", ("sltu", 1, 2, 3, 0)),
        ("slti", "r5, r6, -1", ("slti", 5, 6, 31, -1)),
        ("sltui", "r5, r6, 100", ("sltui", 5, 6, 4, 100)),
        ("ori", "r4, r4, 0xfff", ("ori", 4, 4, 31, 0xFFF)),
        ("andi", "r0, r0, 0", ("andi", 0, 0, 0, 0)),  # NOP
        ("lu12i.w", "r12, 0x12345", ("lu12i.w", 12, 0, 0, 0x12345)),
        ("pcaddi", "r4, 0x7ffff", ("pcaddi", 4, 0, 0, 0x7FFFF)),
        ("pcalau12i", "r13, 0x7ffff", ("pcalau12i", 13, 0, 0, 0x7FFFF)),
        ("pcaddu12i", "r5, -1", ("pcaddu12i", 5, 0, 0, -1)),
        ("ld.d", "r4, r12, 0", ("ld.d", 4, 12, 0, 0)),
        ("ld.w", "r5, r6, 12", ("ld.w", 5, 6, 12, 12)),
        ("st.d", "r4, r12, 8", ("st.d", 4, 12, 8, 8)),
        ("st.w", "r4, r12, -8", ("st.w", 4, 12, 24, -8)),
        ("jirl", "r0, r1, 0", ("jirl", 0, 1, 0, 0)),  # ret
        ("jirl", "r1, r13, 0", ("jirl", 1, 13, 0, 0)),  # call through reg
        ("beq", "r4, r5, 8", ("beq", 5, 4, 0, 8)),  # manual order rj, rd
        ("bne", "r4, r5, -8", ("bne", 5, 4, 0, -8)),
        ("blt", "r6, r7, 4", ("blt", 7, 6, 0, 4)),
        ("b", "8", ("b", 0, 0, 0, 8)),
        ("bl", "-4", ("bl", 0, 0, 0, -4)),
    ]

    def test_round_trip(self):
        for mnem, ops, want in self.CASES:
            word, rels = enc(mnem, ops)
            self.assertEqual(rels, [], f"{mnem} {ops}")
            got = emu.decode(int.from_bytes(word, "little"))
            self.assertIsNotNone(got, f"{mnem} {ops} did not decode")
            self.assertEqual(got, want, f"{mnem} {ops}")

    def test_golden_7bit_encodings(self):
        # Ground truth from QEMU target/loongarch/insns.decode and lld:
        # pcaddi 0001_100, pcalau12i 0001_101, pcaddu12i 0001_110.
        word, _ = enc("pcaddi", "r4, 0")
        self.assertEqual(int.from_bytes(word, "little"), 0x18000004)
        word, _ = enc("pcalau12i", "r4, 0")
        self.assertEqual(int.from_bytes(word, "little"), 0x1A000004)
        word, _ = enc("pcaddu12i", "r4, 0")
        self.assertEqual(int.from_bytes(word, "little"), 0x1C000004)

    def test_register_aliases(self):
        word, _ = enc("addi.d", "sp, sp, 0")
        self.assertEqual(emu.decode(int.from_bytes(word, "little"))[1:3], (3, 3))
        word, _ = enc("addi.d", "a0, t0, 0")
        self.assertEqual(emu.decode(int.from_bytes(word, "little"))[1:3], (4, 12))

    def test_branch_field_placement(self):
        # b to +8: offs26 field = 8>>2 = 2, low 16 bits at [25:10]
        word, _ = enc("b", "8")
        w = int.from_bytes(word, "little")
        self.assertEqual((w >> 10) & 0xFFFF, 2)
        # bl to -4: field = -1 = 0x3FFFFFF, bits [25:10] = 0xFFFF, [9:0] = 0x3FF
        word, _ = enc("bl", "-4")
        w = int.from_bytes(word, "little")
        self.assertEqual((w >> 10) & 0xFFFF, 0xFFFF)
        self.assertEqual(w & 0x3FF, 0x3FF)
        # jirl to +4: offs16 = 1 at bits [25:10]
        word, _ = enc("jirl", "r1, r13, 4")
        w = int.from_bytes(word, "little")
        self.assertEqual((w >> 10) & 0xFFFF, 1)


class TestRelocs(unittest.TestCase):
    def _relocs(self, text):
        obj = asm.assemble(text, "t.obj")
        return [ln for ln in obj.splitlines() if ln.startswith("REL")]

    def test_pcala_pair(self):
        rels = self._relocs(
            ".text\npcalau12i r12, %pcala_hi20(magic)\nld.d r4, r12, %pcala_lo12(magic)\n"
        )
        self.assertEqual(
            rels,
            [
                "REL .text 0 PCALA_HI20 magic",
                "REL .text 4 PCALA_LO12 magic",
                "REL .text 4 RELAX",
            ],
        )

    def test_la_macros_expand(self):
        obj = asm.assemble(".text\nla.local r4, x\nla.global r13, y\n", "t.obj")
        lines = obj.splitlines()
        self.assertEqual(lines[1], "SEC .text")
        self.assertEqual(
            lines[2].split(),
            [
                "04",
                "00",
                "00",
                "1a",
                "84",
                "00",
                "c0",
                "02",
                "0d",
                "00",
                "00",
                "1a",
                "ad",
                "01",
                "c0",
                "28",
            ],
        )
        self.assertIn("REL .text 0 PCALA_HI20 x", obj)
        self.assertIn("REL .text 4 PCALA_LO12 x", obj)
        self.assertIn("REL .text 4 RELAX", obj)
        self.assertIn("REL .text 8 GOT_PC_HI20 y", obj)
        self.assertIn("REL .text 0xc GOT_PC_LO12 y", obj)
        self.assertIn("REL .text 0xc RELAX", obj)

    def test_got_pair_and_branch(self):
        rels = self._relocs(
            ".text\n"
            "pcalau12i r13, %got_pc_hi20(helper)\n"
            "ld.d r13, r13, %got_pc_lo12(helper)\n"
            "b image_end\n"
        )
        self.assertEqual(
            rels,
            [
                "REL .text 0 GOT_PC_HI20 helper",
                "REL .text 4 GOT_PC_LO12 helper",
                "REL .text 4 RELAX",
                "REL .text 8 B26 image_end",
            ],
        )

    def test_word_quad_addend_expressions(self):
        obj = asm.assemble(
            ".data\n.global magic\nmagic:\n.quad 7\n.quad magic+8\n.word magic\n.word magic-4\n",
            "t.obj",
        )
        self.assertIn("SYM magic G .data 0", obj)
        self.assertIn("REL .data 8 R_LARCH_64 magic 8", obj)
        self.assertIn("REL .data 0x10 R_LARCH_32 magic", obj)
        self.assertIn("REL .data 0x14 R_LARCH_32 magic -4", obj)

    def test_quad_label(self):
        obj = asm.assemble(".data\n.quad magic\n", "t.obj")
        self.assertIn("REL .data 0 R_LARCH_64 magic", obj)
        self.assertIn("SEC .data", obj)

    def test_symbols_and_commons(self):
        obj = asm.assemble(
            ".text\n"
            "global_fn:\n"
            ".global global_fn\n"
            "addi.d r4, r4, 0\n"
            "local_fn:\n"
            "addi.d r4, r4, 1\n"
            ".common big, 8, 8\n",
            "t.obj",
        )
        self.assertIn("SYM global_fn G .text 0", obj)
        self.assertIn("SYM local_fn L .text 4", obj)
        self.assertIn("SYM big C - 8 8", obj)

    def test_section_switch_and_align(self):
        obj = asm.assemble(
            ".text\nnop1:\nandi r0, r0, 0\n.align 4\n.data\n.word 7\n",
            "t.obj",
        )
        # .align 4 in .text emits the full max run (12 NOP bytes) and an
        # R_LARCH_ALIGN request for the linker.
        self.assertIn("SEC .text", obj)
        text = obj.split("SEC .text")[1].split("SEC .data")[0]
        toks = text.split()
        self.assertEqual(len(toks), 16)  # 4-byte instr + 12 NOP bytes
        self.assertEqual(toks[4:], ["00", "00", "40", "03"] * 3)  # NOP, LE bytes
        self.assertIn("ALIGN .text 4 0x10", obj)


class TestErrors(unittest.TestCase):
    def test_bad_instruction(self):
        with self.assertRaises(asm.AsmError):
            asm.assemble(".text\nfrob r4, r4, 0\n", "t.obj")

    def test_bad_register(self):
        with self.assertRaises(asm.AsmError):
            asm.assemble(".text\naddi.d r99, r4, 0\n", "t.obj")

    def test_branch_cond_label_emits_b16(self):
        obj = asm.assemble(".text\nbeq r4, r5, somewhere\n", "t.obj")
        self.assertIn("REL .text 0 B16 somewhere", obj)

    def test_jirl_label_rejected(self):
        with self.assertRaises(asm.AsmError):
            asm.assemble(".text\njirl r1, r4, somewhere\n", "t.obj")

    def test_label_without_body(self):
        with self.assertRaises(asm.AsmError):
            asm.assemble(".text\norphan:\n", "t.obj")

    def test_bytes_outside_section(self):
        with self.assertRaises(asm.AsmError):
            asm.assemble("addi.d r4, r4, 0\n", "t.obj")

    def test_wrong_reloc_suffix(self):
        with self.assertRaises(asm.AsmError):
            asm.assemble(".text\nlu12i.w r4, %pcala_lo12(x)\n", "t.obj")


class TestDemo(unittest.TestCase):
    def test_byte_identity_and_run(self):
        import contextlib
        import io
        import linker_loongarch as ll

        for src, ref, name in (
            (asm.PROG_A, ll.OBJ_A, "a"),
            (asm.PROG_B, ll.OBJ_B, "b"),
            (asm.PROG_C, ll.OBJ_C, "c"),
        ):
            obj = asm.assemble(src, name)
            self.assertEqual(
                asm._structural(obj), asm._structural(ref), f"{name} not byte-identical"
            )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(asm.main(), 0)
        self.assertIn("the assembled toolchain agrees", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
