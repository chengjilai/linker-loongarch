# GNU as fixture

`fixture.o` was produced from `fixture.s` with:

```sh
loongarch64-unknown-linux-gnu-as fixture.s -o fixture.o
```

Toolchain: GNU Binutils 2.46. It is committed so the reader tests do
not require a LoongArch cross assembler.

The fixture intentionally covers the real-object cases the toy writer
does not emit: R_LARCH_RELAX markers from `la.local`, a local-label R_LARCH_B16
branch, SHT_NOBITS `.bss`, R_LARCH_32, and non-zero RELA addends.
