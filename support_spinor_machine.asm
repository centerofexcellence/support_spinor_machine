section .data
y dq 0.5
model dq 1.0
n dq 100.0
m dq 50.0
section .text
global _start
_start:
    vpxord zmm0, zmm0, zmm0 ; Clear zmm0 to zero
.loop:
    vpxorq zmm1, zmm1, zmm1 ; Clear zmm1 to zero
    rdrand eax ; Read a random number from RDRAND
    vpbroadcastd zmm2, eax ; Broadcast the random number to all elements of zmm2
    vcvtdq2ps zmm2, zmm2 ; Convert the integer to a float
    vmulps zmm2, zmm2, model ; Multiply by the model weight
    vaddps zmm1, zmm1, zmm2 ; Add to the current sum
    rdrand eax ; Read another random number from RDRAND
    vpbroadcastd zmm3, eax ; Broadcast the random number to all elements of zmm3
    vcvtdq2ps zmm3, zmm3 ; Convert the integer to a float
    vmulps zmm3, zmm3, model ; Multiply by the model weight
    vaddps zmm1, zmm1, zmm3 ; Add to the current sum
    rdrand eax ; Read another random number from RDRAND
    vpbroadcastd zmm4, eax ; Broadcast the random number to all elements of zmm4
    vcvtdq2ps zmm4, zmm4 ; Convert the integer to a float
    vmulps zmm4, zmm4, model ; Multiply by the model weight
    vaddps zmm1, zmm1, zmm4 ; Add to the current sum
    rdrand eax ; Read another random number from RDRAND
    vpbroadcastd zmm5, eax ; Broadcast the random number to all elements of zmm5
    vcvtdq2ps zmm5, zmm5 ; Convert the integer to a float
    vmulps zmm5, zmm5, model ; Multiply by the model weight
    vaddps zmm1, zmm1, zmm5 ; Add to the current sum
    rdrand eax ; Read another random number from RDRAND
    vpbroadcastd zmm6, eax ; Broadcast the random number to all elements of zmm6
    vcvtdq2ps zmm6, zmm6 ; Convert the integer to a float
    vmulps zmm6, zmm6, model ; Multiply by the model weight
    vaddps zmm1, zmm1, zmm6 ; Add to the current sum
    rdrand eax ; Read another random number from RDRAND
    vpbroadcastd zmm7, eax ; Broadcast the random number to all elements of zmm7
    vcvtdq2ps zmm7, zmm7 ; Convert the integer to a float
    vmulps zmm7, zmm7, model ; Multiply by the model weight
    vaddps zmm1, zmm1, zmm7 ; Add to the current sum
    rdrand eax ; Read another random number from RDRAND
    vpbroadcastd zmm8, eax ; Broadcast the random number to all elements of zmm8
    vcvtdq2ps zmm8, [rsi + 64]
    vcvtdq2ps zmm9, [rsi + 96]
    vcvtdq2ps zmm10, [rsi + 128]
    vcvtdq2ps zmm11, [rsi + 160]

    ; Compute the dot product of input with the weights
    vfmadd231ps zmm0, zmm8, [r15 + 64]
    vfmadd231ps zmm1, zmm9, [r15 + 96]
    vfmadd231ps zmm2, zmm10, [r15 + 128]
    vfmadd231ps zmm3, zmm11, [r15 + 160]

    add r15, 192
    add rsi, 64
    dec ecx
    jnz loop

    ; Reduce the 4 results into a single value
    vaddps zmm0, zmm0, zmm1
    vaddps zmm2, zmm2, zmm3
    vaddps zmm0, zmm0, zmm2
    vextractf32x8 dword ptr [rsp + 12], zmm0, 1

    ; Compare the result to the threshold and return the label
    ; If the result is greater than the threshold, return 1
    ; Otherwise, return 0
    xor eax, eax
    vcomiss [rsp + 12], xmm7
    seta al

    ; Clean up the stack and return the result
    add rsp, 16
    ret
