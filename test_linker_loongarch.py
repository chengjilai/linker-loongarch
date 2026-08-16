# SPDX-License-Identifier: Apache-2.0
"""test_linker_loongarch.py — unittest suite for the LoongArch linker.

Covers: object parsing, symbol resolution (dup/undef errors), every
relocation formula against a hand-computed layout, GOT/PLT synthesis,
the R_LARCH_64 data reference, and the end-to-end emulated run.
"""

import contextlib
import io
import unittest

import linker_loongarch as ll


def link_texts(*texts, relax_log=None, align_log=None):
    """Parse + link the given object texts; return (image, symaddr, layout)."""
    objects = [ll.parse_object(t, f"obj{i}") for i, t in enumerate(texts)]
    return ll.link(objects, ll.BASE, relax_log=relax_log, align_log=align_log)


class TestParse(unittest.TestCase):
    def test_sections_and_symbols(self):
        obj = ll.parse_object(
            "SEC .text\n04 08 c0 02\nSYM f G .text 0\nREL .text 4 PCALA_LO12 x\n",
            "a",
        )
        self.assertEqual(obj.sections[".text"], bytearray.fromhex("0408c002"))
        self.assertEqual(obj.symbols["f"], ("G", ".text", 0, 0))

    def test_bad_symbol_kind(self):
        with self.assertRaises(ll.LinkError):
            ll.parse_object("SYM f X .text 0\n", "a")

    def test_undefined_and_duplicate(self):
        with self.assertRaises(ll.LinkError):
            ll.resolve([ll.parse_object("SYM f G - 0\n", "a")])
        with self.assertRaises(ll.LinkError):
            ll.resolve(
                [
                    ll.parse_object("SYM f G .text 0\n", "a"),
                    ll.parse_object("SYM f G .text 0\n", "b"),
                ]
            )


class TestRelocations(unittest.TestCase):
    def test_pcala_pair(self):
        # hi20 at +0, lo12 at +4, both for the same symbol at .text+8
        image, symaddr, _ = link_texts(
            "SEC .text\n00 00 00 00\n00 00 00 00\nSYM target G .text 8\n"
            "REL .text 0 PCALA_HI20 target\nREL .text 4 PCALA_LO12 target\n"
        )
        site = ll.BASE
        target = symaddr["target"]
        self.assertEqual(target, ll.BASE + 8)
        word_hi = int.from_bytes(image[0:4], "little")
        word_lo = int.from_bytes(image[4:8], "little")
        # independent check of the ABI formula, with the LO12 sign rounding
        lo = target & 0xFFF
        hi = ((target & ~0xFFF) - (site & ~0xFFF)) >> 12
        if lo > 0x7FF:
            hi += 1
        self.assertEqual((word_hi >> 5) & 0xFFFFF, hi & 0xFFFFF)
        self.assertEqual((word_lo >> 10) & 0xFFF, lo)

    def test_b26_branch(self):
        # b at 0x400000 to 0x400008: offs26 = 8 >> 2 = 2
        image, _, _ = link_texts(
            "SEC .text\n00 00 00 00\nSYM here G .text 0\nSYM there G .text 8\n"
            "REL .text 0 B26 there\n"
        )
        word = int.from_bytes(image[0:4], "little")
        # ABI: [25:10] = offs[17:2], [9:0] = offs[27:18] — for small offsets the
        # low 16 bits of the field sit at bits 10..25
        self.assertEqual((word >> 10) & 0xFFFF, 2)

    def test_b16_branch(self):
        # beq at BASE to BASE+8: offs16 field = 8 >> 2 = 2, bits [25:10]
        image, symaddr, _ = link_texts(
            "SEC .text\n00 00 00 00\n00 00 00 00\n"
            "SYM here G .text 0\nSYM there G .text 4\n"
            "REL .text 0 B16 there\n"
        )
        word = int.from_bytes(image[0:4], "little")
        disp = symaddr["there"] - ll.BASE
        self.assertEqual((word >> 10) & 0xFFFF, disp >> 2)

    def test_b16_local_target(self):
        image, _, layout = link_texts(
            "SEC .text\n00 00 00 00\n00 00 00 00\n"
            "SYM here L .text 0\nSYM there L .text 4\n"
            "REL .text 0 B16 there\n"
        )
        word = int.from_bytes(image[0:4], "little")
        self.assertEqual((word >> 10) & 0xFFFF, 1)
        self.assertTrue(any(k == "B16" for _, _, _, k, _, _, _, _ in layout.applied))

    def test_got_pair(self):
        # an undefined 'f' forces a GOT slot; the pair must agree on it
        image, _, layout = link_texts(
            "SEC .text\n00 00 00 00\n00 00 00 00\nSYM f G - 0\n"
            "REL .text 0 GOT_PC_HI20 f\nREL .text 4 GOT_PC_LO12 f\n",
            "SEC .text\n00 00 00 00\nSYM f G .text 0\n",
        )
        self.assertIn("f", layout.got)
        slot = layout.got["f"]  # absolute address of the GOT slot
        site = ll.BASE
        lo = slot & 0xFFF
        hi = ((slot & ~0xFFF) - (site & ~0xFFF)) >> 12
        if lo > 0x7FF:
            hi += 1
        word_hi = int.from_bytes(image[0:4], "little")
        word_lo = int.from_bytes(image[4:8], "little")
        self.assertEqual((word_hi >> 5) & 0xFFFFF, hi & 0xFFFFF)
        self.assertEqual((word_lo >> 10) & 0xFFF, lo)

    def test_r64_absolute(self):
        image, symaddr, _ = link_texts(
            "SEC .data\n00 00 00 00 00 00 00 00\nREL .data 0 R_LARCH_64 y\n",
            "SEC .data\n00 00 00 00\nSYM y G .data 0\n",
        )
        self.assertEqual(int.from_bytes(image[0:8], "little"), symaddr["y"])

    def test_plt_stub(self):
        # helper called via GOT+PLT: the stub is binutils' 12-byte sequence
        image, _, layout = link_texts(
            "SEC .text\n00 00 00 00\nSYM helper G - 0\n"
            "REL .text 0 GOT_PC_HI20 helper\nREL .text 4 GOT_PC_LO12 helper\n",
            "SEC .text\n00 00 00 00\nSYM helper G .text 0\n",
        )
        self.assertIn("helper", layout.plt)
        stub = layout.plt["helper"] - ll.BASE
        pcaddu12i = int.from_bytes(image[stub : stub + 4], "little")
        ld_d = int.from_bytes(image[stub + 4 : stub + 8], "little")
        jirl = int.from_bytes(image[stub + 8 : stub + 12], "little")
        self.assertEqual(pcaddu12i & 0xFC000000, 0x1C000000)  # pcaddu12i
        self.assertEqual(ld_d & 0xFC000000, 0x28000000)  # ld.d
        self.assertEqual(jirl & 0xFC000000, 0x4C000000)  # jirl


class TestRelaxation(unittest.TestCase):
    """R_LARCH_RELAX pair folding (lld's relaxPCHi20Lo12 semantics)."""

    # real instruction bytes: pcalau12i r4 = 1a000004; addi.d r4,r4,0 = 02c00084
    RELAX_PAIR = (
        "SEC .text\n"
        "04 00 00 1a\n"  # pcalau12i r4
        "84 00 c0 02\n"  # addi.d r4, r4, 0
        "SYM x G .text 8\n"
        "REL .text 0 PCALA_HI20 x\n"
        "REL .text 4 PCALA_LO12 x\n"
        "REL .text 4 RELAX\n"
    )

    def _word(self, image, off):
        return int.from_bytes(image[off : off + 4], "little")

    def test_pcala_pair_folds(self):
        image, _, _ = link_texts(self.RELAX_PAIR)
        word = self._word(image, 0)
        self.assertEqual(word >> 25, 0x0C)  # pcaddi
        self.assertEqual(word & 0x1F, 4)  # same rd
        # x shifted with the fold (8 -> 4, the section end), so delta is 4
        self.assertEqual((word >> 5) & 0xFFFFF, 1)
        self.assertEqual(self._word(image, 4), 0)  # the pair is gone

    def test_no_fold_without_relax(self):
        image, _, _ = link_texts(self.RELAX_PAIR.replace("REL .text 4 RELAX\n", ""))
        self.assertEqual(self._word(image, 0) >> 25, 0x0D)  # pcalau12i intact
        self.assertEqual(self._word(image, 4) >> 22, 0x0B)  # addi.d intact

    def test_far_target_not_folded(self):
        # x 0x300000 bytes away: outside pcaddi's 22-bit range
        obj = (
            "SEC .text\n04 00 00 1a\n84 00 c0 02\n"
            "SEC .big\n" + "00\n" * 0x300000 + "\n"
            "SYM x G .big 0x300000\n"
            "REL .text 0 PCALA_HI20 x\n"
            "REL .text 4 PCALA_LO12 x\n"
            "REL .text 4 RELAX\n"
        )
        image, _, _ = link_texts(obj)
        self.assertEqual(self._word(image, 0) >> 25, 0x0D)  # not folded

    def test_noncanonical_registers_not_folded(self):
        # pcalau12i r12 (1a00000c); addi.d r4, r12 (02c00104) — lld rejects
        obj = (
            "SEC .text\n0c 00 00 1a\n04 01 c0 02\n"
            "SYM x G .text 8\n"
            "REL .text 0 PCALA_HI20 x\n"
            "REL .text 4 PCALA_LO12 x\n"
            "REL .text 4 RELAX\n"
        )
        image, _, _ = link_texts(obj)
        self.assertEqual(self._word(image, 0) >> 25, 0x0D)

    def test_pcala_load_pair_not_folded(self):
        # pcalau12i r4 + ld.d r4,r4,0 (28c00084): lld requires addi for PCALA
        obj = (
            "SEC .text\n04 00 00 1a\n84 00 c0 28\n"
            "SYM x G .text 8\n"
            "REL .text 0 PCALA_HI20 x\n"
            "REL .text 4 PCALA_LO12 x\n"
            "REL .text 4 RELAX\n"
        )
        image, _, _ = link_texts(obj)
        self.assertEqual(self._word(image, 0) >> 25, 0x0D)

    def test_got_pair_folds_to_symbol_address(self):
        # la.global: pcalau12i r4 + ld.d r4,r4,0 GOT pair folds to pcaddi
        obj_a = (
            "SEC .text\n04 00 00 1a\n84 00 c0 28\n"
            "SYM f G - 0\n"
            "REL .text 0 GOT_PC_HI20 f\n"
            "REL .text 4 GOT_PC_LO12 f\n"
            "REL .text 4 RELAX\n"
        )
        obj_b = "SEC .text\n00 00 00 00\nSYM f G .text 0\n"
        image, symaddr, layout = link_texts(obj_a, obj_b)
        self.assertEqual(self._word(image, 0) >> 25, 0x0C)  # folded pcaddi
        self.assertEqual(layout.got, {})  # slot died with the pair
        self.assertEqual((self._word(image, 0) >> 5) & 0xFFFFF, (symaddr["f"] - ll.BASE) >> 2)

    def test_demo_relaxed_end_to_end(self):
        # the full demo: relaxed image runs a0 = 6 and is smaller
        import io, contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            import larch_asm

            objects = [
                (n, larch_asm.assemble(s, n))
                for s, n in (
                    (larch_asm.PROG_A, "a.obj"),
                    (larch_asm.PROG_B, "b.obj"),
                    (larch_asm.PROG_C, "c.obj"),
                )
            ]
            objs = [ll.parse_object(t, n) for n, t in objects]
            img, sym, _ = ll.link(objs, ll.BASE)
            objs2 = [ll.parse_object(t, n) for n, t in objects]
            img2, sym2, _ = ll.link(objs2, ll.BASE, relax=False)
            import larch_emu

            self.assertEqual(larch_emu.run(img, sym["_start"] - ll.BASE), 6)
            self.assertEqual(larch_emu.run(img2, sym2["_start"] - ll.BASE), 6)
            self.assertLess(len(img), len(img2))


class TestRelaxationTrace(unittest.TestCase):
    def test_fold_is_recorded(self):
        log = []
        image, _, _ = link_texts(TestRelaxation.RELAX_PAIR, relax_log=log)
        self.assertEqual(len(log), 1)
        step = log[0]
        self.assertEqual(step.pass_no, 1)
        self.assertEqual(step.obj, "obj0")
        self.assertEqual(step.section, ".text")
        self.assertEqual(step.offset, 4)
        self.assertEqual(step.kind, "PCALA_LO12")
        self.assertEqual(step.sym, "x")
        self.assertEqual(step.decision, "fold")
        self.assertIsNone(step.reason)
        self.assertEqual(step.delta, 8)
        # folded word is pcaddi; the surviving relocation is re-applied
        self.assertEqual((int.from_bytes(image[0:4], "little") >> 25), 0x0C)

    def test_skip_reasons_are_recorded(self):
        # out of range
        log = []
        link_texts(
            "SEC .text\n04 00 00 1a\n84 00 c0 02\n"
            "SEC .big\n" + "00\n" * 0x300000 + "\n"
            "SYM x G .big 0x300000\n"
            "REL .text 0 PCALA_HI20 x\n"
            "REL .text 4 PCALA_LO12 x\n"
            "REL .text 4 RELAX\n",
            relax_log=log,
        )
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0].decision, "skip")
        self.assertIn("range", log[0].reason)

        # non-canonical registers
        log = []
        link_texts(
            "SEC .text\n0c 00 00 1a\n04 01 c0 02\n"
            "SYM x G .text 8\n"
            "REL .text 0 PCALA_HI20 x\n"
            "REL .text 4 PCALA_LO12 x\n"
            "REL .text 4 RELAX\n",
            relax_log=log,
        )
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0].decision, "skip")
        self.assertIn("register", log[0].reason)

        # wrong opcode shape for a PCALA pair
        log = []
        link_texts(
            "SEC .text\n04 00 00 1a\n84 00 c0 28\n"
            "SYM x G .text 8\n"
            "REL .text 0 PCALA_HI20 x\n"
            "REL .text 4 PCALA_LO12 x\n"
            "REL .text 4 RELAX\n",
            relax_log=log,
        )
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0].decision, "skip")
        self.assertIn("shape", log[0].reason)

    def test_demo_trace_folds_three_pairs(self):
        import larch_asm

        objects = [
            (n, larch_asm.assemble(s, n))
            for s, n in (
                (larch_asm.PROG_A, "a.obj"),
                (larch_asm.PROG_B, "b.obj"),
                (larch_asm.PROG_C, "c.obj"),
            )
        ]
        objs = [ll.parse_object(t, n) for n, t in objects]
        log = []
        ll.link(objs, ll.BASE, relax_log=log)
        folds = [s for s in log if s.decision == "fold"]
        self.assertEqual(len(folds), 3)
        self.assertEqual([s.pass_no for s in folds], [1, 2, 3])
        self.assertEqual(
            {(s.obj, s.kind) for s in folds},
            {
                ("a.obj", "PCALA_LO12"),
                ("a.obj", "GOT_PC_LO12"),
                ("b.obj", "PCALA_LO12"),
            },
        )
        self.assertTrue(all(s.target is not None for s in folds))

    def test_got_trace_targets_symbol_not_slot(self):
        log = []
        obj_a = (
            "SEC .text\n04 00 00 1a\n84 00 c0 28\n"
            "SYM f G - 0\n"
            "REL .text 0 GOT_PC_HI20 f\n"
            "REL .text 4 GOT_PC_LO12 f\n"
            "REL .text 4 RELAX\n"
        )
        obj_b = "SEC .text\n00 00 00 00\nSYM f G .text 0\n"
        _, symaddr, layout = link_texts(obj_a, obj_b, relax_log=log)
        self.assertEqual(len(log), 1)
        step = log[0]
        self.assertEqual(step.decision, "fold")
        self.assertEqual(step.kind, "GOT_PC_LO12")
        self.assertEqual(step.target, symaddr["f"])
        self.assertNotEqual(step.target, layout.got.get("f"))
        self.assertNotIn("f", layout.got)  # the slot died with the pair


class TestAlignRelaxation(unittest.TestCase):
    NOP = "00 00 40 03"

    def test_excess_nops_are_deleted(self):
        # 12 bytes of real code, then the assembler's 12-byte max NOP run.
        # At BASE+0xc an align-16 request needs only 4 NOPs, so 8 die.
        obj = "SEC .text\n" + f"{self.NOP}\n" * 3 + f"{self.NOP}\n" * 3 + "ALIGN .text 0xc 0x10 0\n"
        image, _, layout = link_texts(obj)
        self.assertEqual(len(image), 0x10)
        self.assertEqual(layout.align_delta[("obj0", ".text")], [(16, 8)])
        self.assertEqual(image[:16], bytes.fromhex("00004003" * 4))

    def test_align_recomputed_after_pair_fold(self):
        # A relaxable pair followed by .align 16 with max 8 NOPs.
        # Pass 0: the unrelaxed pair leaves the ALIGN site at +8; it needs
        # 8 NOPs, within max, so 4 are deleted.
        # Pair fold shifts the ALIGN site to +4; now alignment would need
        # 12 NOPs, above max, so the alignment is abandoned and all 12 die.
        obj = (
            "SEC .text\n04 00 00 1a\n84 00 c0 02\n" + f"{self.NOP}\n" * 3 + "SYM x G .text 8\n"
            "REL .text 0 PCALA_HI20 x\n"
            "REL .text 4 PCALA_LO12 x\n"
            "REL .text 4 RELAX\n"
            "ALIGN .text 8 0x10 8\n"
        )
        relax_log, align_log = [], []
        image, symaddr, layout = link_texts(obj, relax_log=relax_log, align_log=align_log)
        self.assertEqual(len(image), 0x10)
        self.assertEqual(symaddr["x"], ll.BASE + 4)  # end of the folded section
        self.assertEqual([s.decision for s in relax_log], ["fold"])
        self.assertEqual(len(align_log), 2)
        self.assertEqual(align_log[0].pass_no, 0)
        self.assertFalse(align_log[0].abandoned)
        self.assertEqual(align_log[0].needed, 8)
        self.assertEqual(align_log[0].removed, 4)
        self.assertEqual(align_log[1].pass_no, 1)
        self.assertTrue(align_log[1].abandoned)
        self.assertEqual(align_log[1].needed, 12)
        self.assertEqual(align_log[1].removed, 12)
        self.assertEqual(layout.align_delta[("obj0", ".text")], [(4, 12)])

    def test_align_plan_shifts_following_relocations(self):
        # A B26 at +12 must be patched at its shrunk address, not its
        # object-relative offset; here align deletes 8 NOPs before it.
        obj = (
            "SEC .text\n" + f"{self.NOP}\n" * 3 + f"{self.NOP}\n" * 3 + "00 00 00 50\n"
            "SYM there G .text 0x18\n"
            "REL .text 0x18 B26 there\n"
            "ALIGN .text 0xc 0x10 0\n"
        )
        image, symaddr, layout = link_texts(obj)
        # section start + reloc offset 0x18 - deleted 8 = BASE + 0x10
        site = ll.BASE + 0x10
        self.assertEqual(layout.applied[-1][5], site)
        # the branch target also shifts past the deleted NOPs
        self.assertEqual(symaddr["there"], ll.BASE + 0x18 - 8)
        disp = symaddr["there"] - site
        word = int.from_bytes(image[site - ll.BASE : site - ll.BASE + 4], "little")
        self.assertEqual((word >> 10) & 0xFFFF, (disp >> 2) & 0xFFFF)


class TestEndToEnd(unittest.TestCase):
    def test_demo_run(self):
        """The full demo pipeline: link, verify, emulate -> a0 == 6."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ll.main([])
        out = buf.getvalue()
        self.assertIn("the emulated run agrees", out)
        self.assertIn("a0=6", out)


if __name__ == "__main__":
    unittest.main()
