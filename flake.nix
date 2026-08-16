# linker-loongarch — a small LoongArch static linker, pure Python (stdlib only).
#
# The linker half of a from-scratch LoongArch toolchain: parse -> resolve
# strong>weak>common -> merge sections -> synthesize .plt/.got/.common ->
# apply relocations -> verify -> emulate, with the LoongArch relocations
# (R_LARCH_64, PCALA_HI20/LO12, GOT_PC_HI20/LO12, B26) applied with the
# ABI's verbatim formulas, the PLT stub taken from binutils'
# loongarch_make_plt_entry (pcalau12i $t2 / ld.d $t2,$t2 / jirl $t1,$t2),
# R_LARCH_RELAX pair relaxation following lld's relaxPCHi20Lo12, and the
# linked image running on larch_emu.py, a small LoongArch interpreter.
#
# Grounding: LoongArch ELF ABI (loongson.github.io, relocation formulas
# verbatim), LoongArch ISA Volume 1 (instruction semantics), binutils
# bfd/elfnn-loongarch.c (PLT stub), gABI ch.4 + x86-64 psABI (S/A/P/G/GP
# definitions) — all cited in linker_loongarch.py's docstring.
#
# Run:
#   nix run                 # link the 3 demo objects, verify, emulate -> a0 = 6
#   nix run .#asm           # the toolchain demo: assemble -> link -> emulate
#   nix run .#larch-as      # assemble one .s file to .obj or .o
#   nix run .#larch-ld      # link .s/.obj/.o inputs into a raw image
#   nix run .#larch-run     # emulate an image and print a0
#   nix run .#dis           # disassemble the linked image (symbols + relocs)
#   nix run .#elf           # write real ELF64-loongarch .o files, read back
#   nix run .#test          # unittest suites (5 modules)
#   nix run .#lint          # ruff (strict ALL, see pyproject.toml)
#   nix run .#typecheck     # ty (strict rules, annotations in the modules)
#   nix develop             # python3
{
  description = "linker-loongarch: a small LoongArch static linker (relocate, verify, emulate)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      src = ./.;
    in
    {
      apps.${system} = {
        default = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "linker-loongarch" ''
            set -euo pipefail
            cd ${src}
            exec ${pkgs.python3}/bin/python3 -B linker_loongarch.py "$@"
          ''}/bin/linker-loongarch";
        };
        asm = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "larch-asm" ''
            set -euo pipefail
            cd ${src}
            exec ${pkgs.python3}/bin/python3 -B larch_asm.py "$@"
          ''}/bin/larch-asm";
        };
        larch-as = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "larch-as" ''
            set -euo pipefail
            cd ${src}
            exec ${pkgs.python3}/bin/python3 -B larch_cli.py as "$@"
          ''}/bin/larch-as";
        };
        larch-ld = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "larch-ld" ''
            set -euo pipefail
            cd ${src}
            exec ${pkgs.python3}/bin/python3 -B larch_cli.py ld "$@"
          ''}/bin/larch-ld";
        };
        larch-run = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "larch-run" ''
            set -euo pipefail
            cd ${src}
            exec ${pkgs.python3}/bin/python3 -B larch_cli.py run "$@"
          ''}/bin/larch-run";
        };
        dis = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "larch-dis" ''
            set -euo pipefail
            cd ${src}
            exec ${pkgs.python3}/bin/python3 -B larch_dis.py "$@"
          ''}/bin/larch-dis";
        };
        elf = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "elf-loongarch" ''
            set -euo pipefail
            cd ${src}
            exec ${pkgs.python3}/bin/python3 -B elf_loongarch.py "$@"
          ''}/bin/elf-loongarch";
        };
        lint = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "lint" ''
            set -euo pipefail
            cd ${src}
            ${pkgs.ruff}/bin/ruff check --no-cache *.py
            ${pkgs.ruff}/bin/ruff format --check --no-cache *.py
          ''}/bin/lint";
        };
        typecheck = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "typecheck" ''
            set -euo pipefail
            cd ${src}
            exec ${pkgs.ty}/bin/ty check
          ''}/bin/typecheck";
        };
        test = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "test" ''
            set -euo pipefail
            cd ${src}
            ${pkgs.python3}/bin/python3 -B -m unittest -v \
              test_linker_loongarch test_larch_emu test_larch_asm \
              test_larch_dis test_elf_loongarch test_larch_cli
          ''}/bin/test";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.python3 pkgs.ruff pkgs.ty ];
      };
    };
}
