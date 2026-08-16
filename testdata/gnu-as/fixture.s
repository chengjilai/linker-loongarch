.text
.global _start
_start:
  la.local $r4, magic
  ld.d $r4, $r4, 0
  beq $r4, $r4, done
  addi.d $r4, $r0, 99
done:
  addi.d $r4, $r4, 0
  b image_end

.data
.global magic
magic:
  .quad 7
.global magic_ref
magic_ref:
  .quad magic + 8
.global magic_word
magic_word:
  .word magic

.bss
.global bss_value
bss_value:
  .space 8

.comm image_end, 0, 1
