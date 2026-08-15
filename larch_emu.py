#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LoongArch (LA64, basic integer subset) single-cycle interpreter, pure Python.

Part of a from-scratch LoongArch toolchain (assembler -> linker ->
emulator).  Executes a linked image and returns r4 (a0) when the PC runs
off the end of the image — there are no syscalls and no halt instruction:
"the machine halts when the PC walks off the end of the program".

Instruction encodings and semantics come from the Loongson ISA manual:

  LoongArch Volume 1 (Basic Architecture), EN edition
  https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html

  - section 1.2            instruction formats (2R/3R/2RI12/2RI16/1RI21/I26:
                           register fields low, opcode high, immediates between)
  - section 2.2.1          integer arithmetic/logic (ADDI.W/D, LU12I.W,
                           SLT[U][I], PCALAU12I, AND/OR/XOR, ANDI/ORI)
  - section 2.2.4          branches (BEQ/BNE/BLT, B, BL, JIRL): offsets are
                           the 16/26-bit field << 2, sign extended, added to
                           the PC of the instruction itself; BL links into r1
  - section 2.2.5.1        LD.W/D, ST.W/D: si12 added to rj, LD.W sign-extends,
                           ST.W stores GR[rd][31:0], ST.D stores GR[rd][63:0]
  - section 2.1.6          little-endian ("LoongArch bit designations are
                           always little-endian")
  - Appendix B (Table 93)  exact bit encodings, e.g.
                           addi.d = 0000001011 si12 rj rd
                           beq    = 010110 offs16 rj rd
                           b      = 010100 offs16 offs10
                           add.d  = 00000000000100001 rk rj rd

Every opcode below was parsed from Appendix B and cross-checked against
qemu's target/loongarch/insns.decode (identical bit patterns).

Machine model
-------------
- Registers r0..r31; r0 is hardwired to zero (writes dropped).
- sp = r3 starts at 0x800000 and grows down (a 1 MiB stack region).
- Memory starts as a copy of the image and grows on demand, zero-filled,
  so stack and data writes past the end of the image just work.
- Unaligned load/store addresses are allowed (the manual, section 2.1.8,
  permits implementations that support unaligned access; we are one).
- All 64-bit arithmetic wraps mod 2^64; ADDI.W/LU12I.W sign-extend their
  32-bit results per the manual.
- An instruction word that does not decode (e.g. 0x00000000 padding) raises
  RuntimeError with the pc and word -- better a loud crash than silent
  garbage execution.  Use NOP (andi r0, r0, 0 = 0x03400000) for padding.
- Execution ends when pc >= len(image) (falls off the end); a negative pc
  (wild branch below address 0) also ends execution.
"""

MASK64 = (1 << 64) - 1
SP_INIT = 0x800000  # initial stack pointer (r3)
MAX_MEM = 0x1000000  # 16 MiB sanity ceiling for on-demand memory growth

# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
# Opcode groups, exactly as Appendix B lays them out (opcode occupies the
# top bits, register fields the bottom bits):
#   6-bit opcodes (bits 31:26)   jirl b bl beq bne blt
#   10-bit opcodes (bits 31:22)  slti sltui addi.w addi.d andi ori
#                               ld.w ld.d st.w st.d
#   17-bit opcodes (bits 31:15)  add.d sub.d slt sltu and or xor
#   7-bit opcodes  (bits 31:25)  lu12i.w pcalau12i
OP6 = {0x13: "jirl", 0x14: "b", 0x15: "bl", 0x16: "beq", 0x17: "bne", 0x18: "blt"}
OP10 = {
    0x08: "slti",
    0x09: "sltui",
    0x0A: "addi.w",
    0x0B: "addi.d",
    0x0D: "andi",
    0x0E: "ori",
    0xA2: "ld.w",
    0xA3: "ld.d",
    0xA6: "st.w",
    0xA7: "st.d",
    # (the 10-bit opcodes of LD.W/D, ST.W/D are 10100010..10100111;
    #  binutils loongarch-opc.c and qemu insns.decode agree)
}
OP17 = {
    0x21: "add.d",
    0x23: "sub.d",
    0x24: "slt",
    0x25: "sltu",
    0x29: "and",
    0x2A: "or",
    0x2B: "xor",
}
OP7 = {0x0A: "lu12i.w", 0x0C: "pcaddu12i", 0x0D: "pcalau12i", 0x0E: "pcaddi"}

# instructions whose immediate is zero-extended (the manual: ui12)
UNSIGNED_IMM = frozenset(("andi", "ori"))


def sext(v: int, bits: int) -> int:
    """Sign-extend a `bits`-wide field (kept in the low bits) to a Python int."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


Decoded = tuple[str, int, int, int, int]  # (mnemonic, rd, rj, rk, imm)


def decode(word: int) -> Decoded | None:
    """Decode one 32-bit word -> (mnemonic, rd, rj, rk, imm).

    `imm` is the already sign-extended immediate/offset:
      si12 -> sext12; offs16 -> sext16 << 2 (byte offset); offs26 -> sext26 << 2;
      si20 -> sext20 (LU12I.W/PCALAU12I shift it themselves).
    For branches the second register operand is called rd in the manual
    (`beq rj, rd, offs16`); it comes back in the rd slot.  Returns None
    for an unrecognized word.
    """
    rd = word & 0x1F
    rj = (word >> 5) & 0x1F
    rk = (word >> 10) & 0x1F

    if word >> 26 in OP6:  # 6-bit opcodes, bits 31:26
        mnem = OP6[word >> 26]
        offs = sext((word >> 10) & 0xFFFF, 16) << 2
        if mnem in ("b", "bl"):
            offs26 = ((word >> 10) & 0xFFFF) | ((word & 0x3FF) << 16)
            return (mnem, 0, 0, 0, sext(offs26, 26) << 2)
        return (mnem, rd, rj, 0, offs)  # jirl/beq/bne/blt

    if word >> 22 in OP10:  # 10-bit opcodes, bits 31:22
        mnem = OP10[word >> 22]
        raw = (word >> 10) & 0xFFF
        imm = raw if mnem in UNSIGNED_IMM else sext(raw, 12)
        return (mnem, rd, rj, rk, imm)

    if word >> 15 in OP17:  # 17-bit opcodes, bits 31:15
        return (OP17[word >> 15], rd, rj, rk, 0)

    if word >> 25 in OP7:  # 7-bit opcodes, bits 31:25
        return (OP7[word >> 25], rd, 0, 0, sext((word >> 5) & 0xFFFFF, 20))

    return None


def sext32(v):
    """Sign-extend a 32-bit quantity to 64 bits (ADDI.W, LU12I.W, LD.W)."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def to_signed(v):
    """Interpret a 64-bit-masked value as a two's-complement signed int."""
    return v - (1 << 64) if v >= 1 << 63 else v


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------
class LA64:
    """Fetch / decode / execute / memory / write-back, one step per instruction."""

    def __init__(self, image: bytes, entry: int):
        self.image = bytes(image)  # original image: the halt boundary
        self.mem = bytearray(image)  # grows on demand, zero-filled
        self.regs: list[int] = [0] * 32
        self.regs[3] = SP_INIT  # sp
        self.pc = entry
        self.steps = 0

    # -- helpers ------------------------------------------------------------
    def _grow(self, end: int) -> None:
        if end > MAX_MEM:
            raise RuntimeError(f"memory access 0x{end:x} exceeds {MAX_MEM:#x} byte limit")
        if end > len(self.mem):
            self.mem.extend(b"\x00" * (end - len(self.mem)))

    def _fetch(self, pc: int) -> int:
        data = self.mem[pc : pc + 4]
        if len(data) < 4:  # ragged tail: pad with zeros
            data += b"\x00" * (4 - len(data))
        return int.from_bytes(data, "little")

    # -- one instruction ------------------------------------------------------
    def step(self) -> dict[str, int | str]:
        """Execute one instruction; return a trace record (dict)."""
        pc = self.pc
        word = self._fetch(pc)
        d = decode(word)
        if d is None:
            raise RuntimeError(
                f"pc=0x{pc:x}: unrecognized instruction word 0x{word:08x} "
                f"(use NOP = andi r0, r0, 0 = 0x03400000 for padding)"
            )
        mnem, rd, rj, rk, imm = d
        regs = self.regs
        mem = self.mem
        self.steps += 1
        src_rj = regs[rj]  # snapshot sources BEFORE execution,
        src_rk = regs[rk]  # so add.d r4, r4, r5 traces the old r4
        src_rb = regs[rd]  # second operand of branches (manual: rd)
        pc_next = pc + 4
        effect = ""

        def write(r, v):
            if r != 0:
                regs[r] = v & MASK64

        if mnem in ("addi.w", "addi.d"):
            if mnem == "addi.w":
                v = sext32((regs[rj] & 0xFFFFFFFF) + (imm & 0xFFFFFFFF))
            else:
                v = (regs[rj] + imm) & MASK64
            v &= MASK64
            write(rd, v)
            effect = f"r{rd} <- 0x{v:x}"

        elif mnem in ("andi", "ori"):
            v = (regs[rj] & imm) if mnem == "andi" else (regs[rj] | imm)
            write(rd, v)
            effect = f"r{rd} <- 0x{v:x}"

        elif mnem in ("add.d", "sub.d", "and", "or", "xor"):
            a, b = regs[rj], regs[rk]
            v = {"add.d": a + b, "sub.d": a - b, "and": a & b, "or": a | b, "xor": a ^ b}[
                mnem
            ] & MASK64
            write(rd, v)
            effect = f"r{rd} <- 0x{v:x}"

        elif mnem in ("slt", "sltu"):
            a, b = regs[rj], regs[rk]
            v = 1 if ((to_signed(a) < to_signed(b)) if mnem == "slt" else a < b) else 0
            write(rd, v)
            effect = f"r{rd} <- {v}"

        elif mnem in ("slti", "sltui"):
            a = regs[rj]
            # slti: signed compare; sltui: unsigned compare of rj with the
            # still-sign-extended immediate (manual 2.2.1.6)
            v = 1 if ((to_signed(a) < imm) if mnem == "slti" else a < (imm & MASK64)) else 0
            write(rd, v)
            effect = f"r{rd} <- {v}"

        elif mnem == "lu12i.w":
            v = sext32(imm << 12) & MASK64  # {si20, 12'b0}, sign-extended to 64
            write(rd, v)
            effect = f"r{rd} <- 0x{v:x}"

        elif mnem == "pcalau12i":
            v = ((pc + (imm << 12)) & MASK64) & ~0xFFF  # low 12 bits erased
            write(rd, v)
            effect = f"r{rd} <- 0x{v:x}"
        elif mnem == "pcaddi":
            # Vol1 2.2.1.8: GR[rd] = PC + SignExtend({si20, 2'b0}) — the
            # relaxed-form instruction lld emits for pcalau12i+addi.d pairs
            v = (pc + (imm << 2)) & MASK64
            write(rd, v)
            effect = f"r{rd} <- 0x{v:x}"

        elif mnem in ("beq", "bne", "blt"):
            target = pc + imm
            a, b = regs[rj], regs[rd]
            if mnem == "beq":
                taken = a == b
            elif mnem == "bne":
                taken = a != b
            else:
                taken = to_signed(a) < to_signed(b)
            if taken:
                pc_next = target
                effect = f"taken -> 0x{target & MASK64:x}"
            else:
                effect = f"not taken (0x{target & MASK64:x})"

        elif mnem in ("b", "bl"):
            pc_next = pc + imm
            if mnem == "bl":
                write(1, pc + 4)  # link into r1 (ra)
                effect = f"r1 <- 0x{pc + 4:x}, pc <- 0x{pc_next & MASK64:x}"
            else:
                effect = f"pc <- 0x{pc_next & MASK64:x}"

        elif mnem == "jirl":
            write(rd, pc + 4)
            pc_next = (regs[rj] + imm) & MASK64
            effect = f"r{rd} <- 0x{pc + 4:x}, pc <- 0x{pc_next:x}"

        elif mnem in ("ld.w", "ld.d"):
            addr = (regs[rj] + imm) & MASK64
            n = 4 if mnem == "ld.w" else 8
            self._grow(addr + n)
            v = int.from_bytes(mem[addr : addr + n], "little")
            if mnem == "ld.w":
                v = sext32(v)  # LD.W sign-extends the word
            v &= MASK64
            write(rd, v)
            effect = f"r{rd} <- M[0x{addr:x}] = 0x{v:x}"

        elif mnem in ("st.w", "st.d"):
            addr = (regs[rj] + imm) & MASK64
            n = 4 if mnem == "st.w" else 8
            self._grow(addr + n)
            v = regs[rd] & (0xFFFFFFFF if mnem == "st.w" else MASK64)
            mem[addr : addr + n] = v.to_bytes(n, "little")
            effect = f"M[0x{addr:x}] <- r{rd} = 0x{v:x}"

        self.pc = pc_next

        return {
            "pc": pc,
            "word": word,
            "mnem": mnem,
            "rd": rd,
            "rj": rj,
            "rk": rk,
            "imm": imm,
            "ops": self._ops_str(mnem, rd, rj, rk, imm, pc + imm),
            "srcs": self._srcs_str(mnem, rd, rj, rk, src_rj, src_rk, src_rb),
            "effect": effect,
        }

    # -- operand / source-value strings for the trace -------------------------
    def _ops_str(self, mnem: str, rd: int, rj: int, rk: int, imm: int, target: int) -> str:
        if mnem in ("beq", "bne", "blt"):
            return f"r{rj}, r{rd}, 0x{target & MASK64:x}"
        if mnem in ("b", "bl"):
            return f"0x{target & MASK64:x}"
        if mnem == "jirl":
            return f"r{rd}, r{rj}, 0x{imm:x}"
        if mnem in ("addi.w", "addi.d", "slti", "sltui", "andi", "ori"):
            return f"r{rd}, r{rj}, {imm}"
        if mnem in ("ld.w", "ld.d", "st.w", "st.d"):
            return f"r{rd}, r{rj}, {imm}"
        if mnem in ("lu12i.w", "pcalau12i"):
            return f"r{rd}, 0x{imm:x}"
        return f"r{rd}, r{rj}, r{rk}"

    def _srcs_str(
        self, mnem: str, rd: int, rj: int, rk: int, src_rj: int, src_rk: int, src_rb: int
    ) -> str:
        if mnem in ("b", "bl"):
            return ""
        if mnem in ("beq", "bne", "blt"):
            return f"r{rj}=0x{src_rj:x} r{rd}=0x{src_rb:x}"
        if mnem in ("add.d", "sub.d", "and", "or", "xor", "slt", "sltu"):
            return f"r{rj}=0x{src_rj:x} r{rk}=0x{src_rk:x}"
        return f"r{rj}=0x{src_rj:x}"

    # -- run until the PC walks off the end of the image -----------------------
    def run(self) -> int:
        while 0 <= self.pc < len(self.image):
            self.step()
        return self.regs[4]  # a0


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------
def run(image: bytes, entry: int) -> int:
    """Execute `image` from `entry` (a byte offset into the image) until the
    PC runs off the end; return r4 (a0)."""
    return LA64(image, entry).run()


def trace(image: bytes, entry: int, limit: int = 10000) -> list[str]:
    """Run like run(), printing one line per step: pc, mnemonic, operand
    register values, and the effect.  Returns the lines as a list of str."""
    m = LA64(image, entry)
    lines = []
    for _ in range(limit):
        if not (0 <= m.pc < len(m.image)):
            break
        rec = m.step()
        line = (
            f"0x{rec['pc']:08x}  {rec['mnem']:<8} {rec['ops']:<26} "
            f"{rec['srcs']:<34} -> {rec['effect']}"
        )
        print(line)
        lines.append(line)
    if lines and m.pc < len(m.image):
        lines.append(f"... truncated at {limit} steps (pc=0x{m.pc:x})")
        print(lines[-1])
    return lines


if __name__ == "__main__":
    # Tiny walk-through: 2 + 3 = 5, the hello-world of this machine.
    # Words stored little-endian, exactly as LD.D/ST.D put them on the wire.
    prog = b"".join(w.to_bytes(4, "little") for w in (0x02C00804, 0x02C00C05, 0x00109484))
    print("larch_emu: LoongArch LA64 interpreter")
    print("program: addi.d r4, r0, 2 ; addi.d r5, r0, 3 ; add.d r4, r4, r5")
    print("result:  run(prog, 0) =", run(prog, 0))
    print()
    trace(prog, 0)
