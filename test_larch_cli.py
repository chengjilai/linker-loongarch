# SPDX-License-Identifier: Apache-2.0
"""test_larch_cli.py — unittest suite for the as/ld/run command driver."""

import contextlib
import importlib.util
import io
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("larch_cli", "larch_cli.py")
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
spec = importlib.util.spec_from_file_location("larch_asm", "larch_asm.py")
assert spec is not None and spec.loader is not None
asm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asm)
spec = importlib.util.spec_from_file_location("elf_loongarch", "elf_loongarch.py")
assert spec is not None and spec.loader is not None
elf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elf)


def run_cmd(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(argv)
    return buf.getvalue()


class TestAs(unittest.TestCase):
    def test_text_and_elf_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            src = d / "one.s"
            src.write_text(".text\n.global _start\n_start:\nla.local r4, magic\n")
            obj = d / "one.obj"
            out = run_cmd(cli._cmd_as, [str(src), "-o", str(obj)])
            self.assertIn("wrote", out)
            self.assertIn("REL .text 0 PCALA_HI20 magic", obj.read_text())

            o = d / "one.o"
            run_cmd(cli._cmd_as, [str(src), "-o", str(o)])
            rb = elf.read_object(str(o))
            self.assertIn("_start", rb.symbols)


class TestLdRun(unittest.TestCase):
    def test_link_and_run_toolchain_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            inputs = []
            for stem, source in (("a", asm.PROG_A), ("b", asm.PROG_B), ("c", asm.PROG_C)):
                src = d / f"{stem}.s"
                src.write_text(source)
                obj = d / f"{stem}.obj"
                run_cmd(cli._cmd_as, [str(src), "-o", str(obj)])
                inputs.append(obj)
            image = d / "prog.bin"
            out = run_cmd(
                cli._cmd_ld,
                [str(p) for p in inputs]
                + ["-o", str(image), "-e", "_start", "--run", "--print-syms"],
            )
            self.assertIn("a0 = 6", out)
            self.assertIn("_start", out)
            self.assertTrue(image.exists())
            out = run_cmd(cli._cmd_run, [str(image), "-e", "0"])
            self.assertIn("a0 = 6", out)


if __name__ == "__main__":
    unittest.main()
