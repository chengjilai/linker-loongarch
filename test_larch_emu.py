#!/usr/bin/env python3
"""Tests for larch_emu.py -- LoongArch LA64 interpreter.

Every instruction word below is hand-encoded from the bit layouts of the
LoongSon ISA manual, Appendix B "Table of Instruction Encoding" (and the
format definitions of section 1.2):

  LoongArch Volume 1 (Basic Architecture), EN edition
  https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html

The packers in this file are an INDEPENDENT implementation of those bit
layouts (they do not import anything from larch_emu), so a bug in the
emulator's decoder cannot hide behind a bug in the test's encoder.

Layouts used (MSB first, per Appendix B / section 1.2):
  3R      opcode[31:15] rk[14:10] rj[9:5] rd[4:0]
  2RI12   opcode[31:22] imm[21:10] rj[9:5] rd[4:0]
  2RI16   opcode[31:26] offs[25:10] rj[9:5] rd[4:0]
  2RI20   opcode[31:25] imm[24:5] rd[4:0]
  I26     opcode[31:26] offs[25:10] offs[9:0]   (offs[15:0] then offs[25:16])

Opcodes (Appendix B):
  6-bit:  jirl=0x13 b=0x14 bl=0x15 beq=0x16 bne=0x17 blt=0x18
  10-bit: slti=0x08 sltui=0x09 addi.w=0x0A addi.d=0x0B andi=0x0D ori=0x0E
          ld.w=0xA2 ld.d=0xA3 st.w=0xA6 st.d=0xA7
  17-bit: add.d=0x21 sub.d=0x23 slt=0x24 sltu=0x25 and=0x29 or=0x2A xor=0x2B
  7-bit:  lu12i.w=0x0A pcalau12i=0x0D

(Cross-checked against qemu target/loongarch/insns.decode -- identical.)

Semantics notes from the manual used by these tests:
  - branches/jirl/b/bl: offset field << 2, sign-extended, added to the PC
    of the instruction itself (sections 2.2.4.1 / 2.2.4.3-5)
  - bl links into r1 (section 2.2.4.4)
  - addi.w / lu12i.w sign-extend their 32-bit results (2.2.1.2 / 2.2.1.4)
  - sltui compares rj unsigned against the SIGN-extended immediate (2.2.1.6)
  - ld.w sign-extends; st.w stores GR[rd][31:0] (2.2.5.1)
  - little-endian memory (2.1.6)
  - sp = r3 = 0x800000, stack grows down
  - execution ends when PC >= len(image); r0 is hardwired zero
"""

import contextlib
import io
import unittest

from larch_emu import LA64, run, trace

# ---------------------------------------------------------------------------
# Independent encoders (doc-derived bit layouts, MSB first)
# ---------------------------------------------------------------------------
OP = {
    # 7-bit opcodes
    "lu12i.w": 0x0A,
    "pcalau12i": 0x0D,
    # 10-bit opcodes
    "slti": 0x08,
    "sltui": 0x09,
    "addi.w": 0x0A,
    "addi.d": 0x0B,
    "andi": 0x0D,
    "ori": 0x0E,
    "ld.w": 0xA2,
    "ld.d": 0xA3,
    "st.w": 0xA6,
    "st.d": 0xA7,
    # 17-bit opcodes
    "add.d": 0x21,
    "sub.d": 0x23,
    "slt": 0x24,
    "sltu": 0x25,
    "and": 0x29,
    "or": 0x2A,
    "xor": 0x2B,
    # 6-bit opcodes
    "jirl": 0x13,
    "b": 0x14,
    "bl": 0x15,
    "beq": 0x16,
    "bne": 0x17,
    "blt": 0x18,
}


def ri12(op, imm, rj, rd):
    """2RI12: opcode[31:22] imm[21:10] rj[9:5] rd[4:0]."""
    return (op << 22) | ((imm & 0xFFF) << 10) | (rj << 5) | rd


def rr3(op, rk, rj, rd):
    """3R: opcode[31:15] rk[14:10] rj[9:5] rd[4:0]."""
    return (op << 15) | (rk << 10) | (rj << 5) | rd


def ri16(op, offs_words, rj, rd):
    """2RI16: opcode[31:26] offs[25:10] rj[9:5] rd[4:0]; offs in WORDS
    (the field is << 2 by the hardware, manual 2.2.4.1)."""
    return (op << 26) | ((offs_words & 0xFFFF) << 10) | (rj << 5) | rd


def offs26(op, offs_words):
    """I26 (b/bl): opcode[31:26] offs[15:0][25:10] offs[25:16][9:0]."""
    offs_words &= 0x3FFFFFF
    return (op << 26) | ((offs_words & 0xFFFF) << 10) | ((offs_words >> 16) & 0x3FF)


def ri20(op, imm, rd):
    """2RI20: opcode[31:25] imm[24:5] rd[4:0]."""
    return (op << 25) | ((imm & 0xFFFFF) << 5) | rd


def prog(*words):
    """Pack instruction words little-endian into an image."""
    return b"".join(w.to_bytes(4, "little") for w in words)


NOP = ri12(OP["andi"], 0, 0, 0)  # andi r0, r0, 0 == 0x03400000


# ---------------------------------------------------------------------------
# Hand-encoded programs.  Each word carries the mnemonic it was encoded from.
# ---------------------------------------------------------------------------
# addi.d/add.d computing 2 + 3 = 5 into a0 (r4) -- literal raw bytes.
#   0x02C00804  addi.d r4, r0, 2     0000001011 000000000010 00000 00100
#   0x02C00C05  addi.d r5, r0, 3     0000001011 000000000011 00000 00101
#   0x00109484  add.d  r4, r4, r5    00000000000100001 00101 00100 00100
RAW_ADD = prog(0x02C00804, 0x02C00C05, 0x00109484)


def prog_lu12i_ori():
    # lu12i.w r4, 0x12345   -> r4 = 0x12345000
    # ori     r4, r4, 0x678 -> r4 = 0x12345678
    return prog(ri20(OP["lu12i.w"], 0x12345, 4), ri12(OP["ori"], 0x678, 4, 4))


def prog_pcalau12i():
    # nop x4 (padding so the pcalau12i sits at pc = 0x10)
    # pcalau12i r4, 0x10 -> (0x10 + 0x10000) & ~0xfff = 0x10000
    # ori r4, r4, 0x2ff  -> 0x102ff
    return prog(NOP, NOP, NOP, NOP, ri20(OP["pcalau12i"], 0x10, 4), ri12(OP["ori"], 0x2FF, 4, 4))


def prog_branches():
    # a beq/bne/blt branch dance: r4 = 0..3 in a loop, then equality checks
    # 0x00 addi.d r4, r0, 0
    # 0x04 addi.d r5, r0, 3
    # 0x08 addi.d r4, r4, 1        # loop body
    # 0x0c blt r4, r5, -1          # back to 0x08 while r4 < 3
    # 0x10 beq r4, r0, +2          # 3 != 0: NOT taken (fall-through tested)
    # 0x14 beq r4, r5, +2          # 3 == 3: taken -> 0x1c
    # 0x18 addi.d r4, r0, 100      # skipped
    # 0x1c addi.d r4, r4, 1        # 4
    return prog(
        ri12(OP["addi.d"], 0, 0, 4),
        ri12(OP["addi.d"], 3, 0, 5),
        ri12(OP["addi.d"], 1, 4, 4),
        ri16(OP["blt"], -1, 4, 5),
        ri16(OP["beq"], 2, 4, 0),
        ri16(OP["beq"], 2, 4, 5),
        ri12(OP["addi.d"], 100, 0, 4),
        ri12(OP["addi.d"], 1, 4, 4),
    )


def prog_call_return():
    # bl call / jirl return with the stack; main falls off the end.
    # The subroutine leaves its result in r6 (the caller restores r4 from
    # the stack), so the result flows out to a0.
    # 0x00 addi.d r3, r3, -16      # sp -= 16
    # 0x04 st.d  r1, r3, 0         # save ra
    # 0x08 st.d  r4, r3, 8         # save a0
    # 0x0c bl    +0x18             # call sub (0x24); r1 = 0x10
    # 0x10 ld.d  r4, r3, 8         # restore a0
    # 0x14 ld.d  r1, r3, 0         # restore ra
    # 0x18 addi.d r3, r3, 16       # sp += 16
    # 0x1c add.d  r4, r4, r6       # a0 = 0 + 40 = 40
    # 0x20 b     +0x0c             # skip the subroutine -> pc = len -> halt
    # 0x24 addi.d r6, r0, 40       # sub: r6 = 40
    # 0x28 jirl  r0, r1, 0         # return: pc = r1
    return prog(
        ri12(OP["addi.d"], -16, 3, 3),
        ri12(OP["st.d"], 0, 3, 1),
        ri12(OP["st.d"], 8, 3, 4),
        offs26(OP["bl"], 6),
        ri12(OP["ld.d"], 8, 3, 4),
        ri12(OP["ld.d"], 0, 3, 1),
        ri12(OP["addi.d"], 16, 3, 3),
        rr3(OP["add.d"], 6, 4, 4),
        offs26(OP["b"], 3),
        ri12(OP["addi.d"], 40, 0, 6),
        ri16(OP["jirl"], 0, 1, 0),
    )


def prog_ld_st():
    # ld/st round trip with sign extension, using the stack region.
    # 0x00 addi.d r5, r0, -2      # 0xFFFFFFFFFFFFFFFE
    # 0x04 st.w  r5, r3, -8       # M[0x7ffff8] = low 32 bits: 0xFFFFFFFE
    # 0x08 ld.w  r4, r3, -8       # r4 = sext32(0xFFFFFFFE) = -2
    # 0x0c st.d  r4, r3, -16      # M[0x7ffff0] = -2
    # 0x10 ld.d  r4, r3, -16      # r4 = -2
    return prog(
        ri12(OP["addi.d"], -2, 0, 5),
        ri12(OP["st.w"], -8, 3, 5),
        ri12(OP["ld.w"], -8, 3, 4),
        ri12(OP["st.d"], -16, 3, 4),
        ri12(OP["ld.d"], -16, 3, 4),
    )


def prog_unaligned():
    # unaligned st.d/ld.d at an odd address (0x7ffffb) -- allowed.
    # 0x00 addi.d r4, r0, -1
    # 0x04 st.d  r4, r3, -5
    # 0x08 ld.d  r4, r3, -5
    # 0x0c addi.d r4, r4, 3       # -1 + 3 = 2
    return prog(
        ri12(OP["addi.d"], -1, 0, 4),
        ri12(OP["st.d"], -5, 3, 4),
        ri12(OP["ld.d"], -5, 3, 4),
        ri12(OP["addi.d"], 3, 4, 4),
    )


def prog_logic():
    # sub.d / and / or / xor / andi / ori chain -> 42.
    # 0x00 addi.d r4, r0, 0x3c    # 60
    # 0x04 addi.d r5, r0, 0x1e    # 30
    # 0x08 sub.d r6, r4, r5       # 30
    # 0x0c and   r4, r6, r5       # 30 & 30 = 30
    # 0x10 or    r4, r4, r5       # 30
    # 0x14 xor   r4, r4, r5       # 0
    # 0x18 andi  r4, r4, 0xff     # 0
    # 0x1c ori   r4, r4, 42       # 42
    return prog(
        ri12(OP["addi.d"], 0x3C, 0, 4),
        ri12(OP["addi.d"], 0x1E, 0, 5),
        rr3(OP["sub.d"], 5, 4, 6),
        rr3(OP["and"], 5, 6, 4),
        rr3(OP["or"], 5, 4, 4),
        rr3(OP["xor"], 5, 4, 4),
        ri12(OP["andi"], 0xFF, 4, 4),
        ri12(OP["ori"], 42, 4, 4),
    )


def prog_slt():
    # slt/sltu/slti/sltui: signed vs unsigned comparisons -> 3.
    # 0x00 addi.d r4, r0, -1      # 0xFFFFFFFFFFFFFFFF
    # 0x04 addi.d r5, r0, 1
    # 0x08 sltu  r6, r4, r5       # unsigned 0xFF..FF < 1?  no -> 0
    # 0x0c slt   r7, r4, r5       # signed   -1 < 1?        yes -> 1
    # 0x10 sltu  r4, r5, r4       # unsigned 1 < 0xFF..FF?   yes -> 1
    # 0x14 slti  r4, r4, 2        # 1 < 2 -> 1
    # 0x18 sltui r4, r4, -1       # unsigned 1 < 0xFF..FF (sext imm) -> 1
    # 0x1c sltui r5, r5, -1       # -> 1
    # 0x20 add.d r4, r4, r5       # 2
    # 0x24 add.d r4, r4, r6       # 2 + 0
    # 0x28 or    r4, r4, r7       # 2 | 1 = 3
    return prog(
        ri12(OP["addi.d"], -1, 0, 4),
        ri12(OP["addi.d"], 1, 0, 5),
        rr3(OP["sltu"], 5, 4, 6),
        rr3(OP["slt"], 5, 4, 7),
        rr3(OP["sltu"], 4, 5, 4),
        ri12(OP["slti"], 2, 4, 4),
        ri12(OP["sltui"], -1, 4, 4),
        ri12(OP["sltui"], -1, 5, 5),
        rr3(OP["add.d"], 5, 4, 4),
        rr3(OP["add.d"], 6, 4, 4),
        rr3(OP["or"], 7, 4, 4),
    )


def prog_addi_w():
    # addi.w 32-bit wraparound + sign extension, sub.d -> 0x7FFFFFFF.
    # 0x00 lu12i.w r4, 0x80000    # {si20,12'b0} = 0x80000000, sext -> 0xFFFFFFFF80000000
    # 0x04 addi.w  r4, r4, 1      # rj[31:0]+1 = 0x80000001 -> sext -> 0xFFFFFFFF80000001
    # 0x08 addi.w  r5, r0, -1     # 0xFFFFFFFFFFFFFFFF
    # 0x0c sub.d   r4, r5, r4     # 0xFFFFFFFFFFFFFFFF - 0xFFFFFFFF80000001
    #                              = 0x000000007FFFFFFE
    # 0x10 addi.d  r4, r4, 1      # 0x7FFFFFFF
    return prog(
        ri20(OP["lu12i.w"], 0x80000, 4),
        ri12(OP["addi.w"], 1, 4, 4),
        ri12(OP["addi.w"], -1, 0, 5),
        rr3(OP["sub.d"], 4, 5, 4),
        ri12(OP["addi.d"], 1, 4, 4),
    )


def prog_b_forward():
    # 0x00 b     +0x08            # skip the 99
    # 0x04 addi.d r4, r0, 99
    # 0x08 addi.d r4, r0, 7
    return prog(offs26(OP["b"], 2), ri12(OP["addi.d"], 99, 0, 4), ri12(OP["addi.d"], 7, 0, 4))


def prog_b_backward():
    # backward b (negative 26-bit offset), reachable from two entries.
    # 0x00 addi.d r4, r0, 7       # <- target of the backward b at 0x0c
    # 0x04 b     +0x10            # skip the dead zone -> 0x14
    # 0x08 addi.d r4, r0, 99
    # 0x0c b     -0x0c            # backward: 0x0c - 12 = 0x00
    # 0x10 addi.d r4, r0, 98
    # 0x14 addi.d r4, r4, 1       # 7 + 1 = 8
    return prog(
        ri12(OP["addi.d"], 7, 0, 4),
        offs26(OP["b"], 4),
        ri12(OP["addi.d"], 99, 0, 4),
        offs26(OP["b"], -3),
        ri12(OP["addi.d"], 98, 0, 4),
        ri12(OP["addi.d"], 1, 4, 4),
    )


def prog_r0_hardwired():
    # writes to r0 must be dropped.
    # 0x00 addi.d r0, r0, 99
    # 0x04 addi.d r4, r0, 0
    return prog(ri12(OP["addi.d"], 99, 0, 0), ri12(OP["addi.d"], 0, 0, 4))


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------
class TestRun(unittest.TestCase):
    """run(image, entry) == r4 (a0) when the PC falls off the end."""

    def test_addi_d_add_d_two_plus_three(self):
        # literal raw bytes, cross-checked against the independent encoders
        self.assertEqual(
            RAW_ADD,
            prog(
                ri12(OP["addi.d"], 2, 0, 4), ri12(OP["addi.d"], 3, 0, 5), rr3(OP["add.d"], 5, 4, 4)
            ),
        )
        self.assertEqual(run(RAW_ADD, 0), 5)

    def test_lu12i_w_ori_constant(self):
        self.assertEqual(run(prog_lu12i_ori(), 0), 0x12345678)

    def test_pcalau12i_ori_constant(self):
        self.assertEqual(run(prog_pcalau12i(), 0), 0x102FF)

    def test_branches_beq_bne_blt(self):
        self.assertEqual(run(prog_branches(), 0), 4)

    def test_b_bl_jirl_call_return_with_stack(self):
        self.assertEqual(run(prog_call_return(), 0), 40)

    def test_ld_st_roundtrip_sign_extension(self):
        self.assertEqual(run(prog_ld_st(), 0), 0xFFFFFFFFFFFFFFFE)

    def test_unaligned_ld_st_allowed(self):
        self.assertEqual(run(prog_unaligned(), 0), 2)

    def test_sub_and_or_xor_andi_ori(self):
        self.assertEqual(run(prog_logic(), 0), 42)

    def test_slt_sltu_slti_sltui(self):
        self.assertEqual(run(prog_slt(), 0), 3)

    def test_addi_w_wraparound_sign_extension(self):
        self.assertEqual(run(prog_addi_w(), 0), 0x7FFFFFFF)

    def test_b_unconditional(self):
        self.assertEqual(run(prog_b_forward(), 0), 7)
        self.assertEqual(run(prog_b_backward(), 0), 8)  # entry 0
        self.assertEqual(run(prog_b_backward(), 0x0C), 8)  # backward b taken

    def test_r0_hardwired_zero(self):
        self.assertEqual(run(prog_r0_hardwired(), 0), 0)

    def test_sp_init_0x800000(self):
        self.assertEqual(run(prog(ri12(OP["addi.d"], 0, 3, 4)), 0), 0x800000)

    def test_falls_off_end_terminates(self):
        # empty image: nothing to execute
        self.assertEqual(run(b"", 0), 0)
        # one NOP, then pc = 4 == len -> halt
        self.assertEqual(run(prog(NOP), 0), 0)
        # entry past the end: immediate halt
        self.assertEqual(run(prog(NOP, NOP), 8), 0)
        self.assertEqual(run(prog(NOP, NOP), 100), 0)
        # wild branch below address 0 also ends execution
        self.assertEqual(run(prog(NOP, offs26(OP["b"], -4)), 4), 0)

    def test_unknown_instruction_raises(self):
        # 0x00000000 is not a valid encoding; padding must be NOPs
        with self.assertRaises(RuntimeError):
            run(b"\x00\x00\x00\x00", 0)

    def test_machine_state_after_run(self):
        m = LA64(prog_call_return(), 0)
        result = m.run()
        self.assertEqual(result, 40)
        self.assertEqual(m.pc, len(m.image))  # halted off the end
        self.assertEqual(m.regs[3], 0x800000)  # sp restored
        self.assertEqual(m.regs[1], 0)  # ra restored
        self.assertEqual(m.regs[4], 40)  # a0
        self.assertEqual(m.regs[6], 40)  # the subroutine's result


class TestTrace(unittest.TestCase):
    """trace() prints one line per step and returns them."""

    def test_trace_lines(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lines = trace(RAW_ADD, 0)
        self.assertEqual(len(lines), 3)
        self.assertIn("0x00000000", lines[0])
        self.assertIn("addi.d", lines[0])
        self.assertIn("r4, r0, 2", lines[0])
        self.assertIn("r0=0x0", lines[0])
        self.assertIn("-> r4 <- 0x5", lines[2])
        # the old r4 value, not the result, is traced as the source
        self.assertIn("r4=0x2", lines[2])
        # trace() prints exactly the lines it returns
        self.assertEqual(out.getvalue(), "".join(line + "\n" for line in lines))

    def test_trace_limit(self):
        lines = trace(prog_branches(), 0, limit=2)
        # two step lines plus the truncation note
        self.assertEqual(len(lines), 3)
        self.assertIn("truncated at 2 steps", lines[-1])

    def test_trace_branch_targets(self):
        lines = trace(prog_branches(), 0)
        # the blt line shows the static target 0x8 and that it was taken
        self.assertTrue(any("blt" in ln and "taken -> 0x8" in ln for ln in lines))
        # the untaken beq shows fall-through
        self.assertTrue(any("beq" in ln and "not taken" in ln for ln in lines))

    def test_trace_stops_at_end(self):
        lines = trace(prog(NOP), 0)
        self.assertEqual(len(lines), 1)
        self.assertIn("andi", lines[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
