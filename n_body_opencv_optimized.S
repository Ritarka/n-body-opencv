	.file	"main.cpp"
	.intel_syntax noprefix
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.p2align 4
	.type	_Z10calcForce2P4Bodyfi._omp_fn.0, @function
_Z10calcForce2P4Bodyfi._omp_fn.0:
.LFB15561:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r15
	push	r14
	push	r13
	push	r12
	push	rbx
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	mov	ebx, DWORD PTR 12[rdi]
	mov	r12, rdi
	and	rsp, -64
	call	omp_get_num_threads@PLT
	mov	r13d, eax
	call	omp_get_thread_num@PLT
	mov	ecx, eax
	mov	eax, ebx
	cdq
	idiv	r13d
	cmp	ecx, edx
	jl	.L2
.L13:
	imul	ecx, eax
	add	ecx, edx
	lea	edx, [rax+rcx]
	cmp	ecx, edx
	jge	.L20
	mov	r9d, ebx
	mov	rdx, QWORD PTR masses[rip]
	mov	rdi, QWORD PTR body_x[rip]
	mov	rsi, QWORD PTR body_y[rip]
	movsx	r8, ecx
	mov	r10, QWORD PTR [r12]
	vxorps	xmm14, xmm14, xmm14
	vcvtss2sd	xmm14, xmm14, DWORD PTR 8[r12]
	mov	r11d, eax
	mov	r12d, ebx
	vmovsd	xmm13, QWORD PTR .LC1[rip]
	vmovsd	xmm11, QWORD PTR .LC3[rip]
	shr	r9d, 3
	vmovsd	xmm23, QWORD PTR .LC5[rip]
	vbroadcastsd	ymm12, xmm11
	vbroadcastsd	ymm17, xmm23
	vbroadcastsd	ymm16, xmm13
	vbroadcastsd	zmm3, xmm11
	add	r11, r8
	lea	rcx, 0[0+r8*8]
	lea	r13d, -1[rbx]
	and	r12d, -8
	sal	r11, 3
	vbroadcastsd	zmm8, xmm23
	vbroadcastsd	zmm7, xmm13
	sal	r9, 6
	.p2align 4
	.p2align 3
.L5:
	vdivsd	xmm4, xmm13, QWORD PTR [rdx+rcx]
	test	ebx, ebx
	jle	.L23
	vmovsd	xmm10, QWORD PTR [rdi+rcx]
	vmovsd	xmm9, QWORD PTR [rsi+rcx]
	cmp	r13d, 6
	jbe	.L14
	vbroadcastsd	zmm6, xmm10
	vbroadcastsd	zmm5, xmm9
	vbroadcastsd	zmm2, xmm4
	xor	eax, eax
	vxorpd	xmm18, xmm18, xmm18
	vmovsd	xmm19, xmm18, xmm18
	.p2align 4
	.p2align 3
.L7:
	vmovupd	zmm1, ZMMWORD PTR [rdi+rax]
	vsubpd	zmm0, zmm1, zmm6
	vmovupd	zmm1, ZMMWORD PTR [rsi+rax]
	vdivpd	zmm0, zmm0, zmm3
	vsubpd	zmm20, zmm1, zmm5
	vdivpd	zmm20, zmm20, zmm3
	vmulpd	zmm1, zmm20, zmm20
	vfmadd231pd	zmm1, zmm0, zmm0
	vaddpd	zmm1, zmm1, zmm8
	vsqrtpd	zmm1, zmm1
	vdivpd	zmm1, zmm7, zmm1
	vmulpd	zmm21, zmm1, zmm1
	vmulpd	zmm1, zmm21, zmm1
	vmulpd	zmm0, zmm1, zmm0
	vmulpd	zmm0, zmm0, ZMMWORD PTR [rdx+rax]
	vmulpd	zmm0, zmm0, zmm2
	vaddsd	xmm19, xmm19, xmm0
	vunpckhpd	xmm22, xmm0, xmm0
	vaddsd	xmm22, xmm22, xmm19
	vextractf64x2	xmm19, ymm0, 1
	vaddsd	xmm22, xmm19, xmm22
	valignq	ymm19, ymm0, ymm0, 3
	vextractf64x4	ymm0, zmm0, 0x1
	vunpckhpd	xmm21, xmm0, xmm0
	vaddsd	xmm19, xmm19, xmm22
	vaddsd	xmm19, xmm0, xmm19
	vaddsd	xmm21, xmm21, xmm19
	vextractf64x2	xmm19, ymm0, 1
	valignq	ymm0, ymm0, ymm0, 3
	vaddsd	xmm19, xmm19, xmm21
	vaddsd	xmm19, xmm19, xmm0
	vmulpd	zmm0, zmm1, zmm20
	vmulpd	zmm0, zmm0, ZMMWORD PTR [rdx+rax]
	add	rax, 64
	vmulpd	zmm0, zmm0, zmm2
	vaddsd	xmm18, xmm18, xmm0
	valignq	ymm1, ymm0, ymm0, 3
	vunpckhpd	xmm20, xmm0, xmm0
	vaddsd	xmm20, xmm20, xmm18
	vextractf64x2	xmm18, ymm0, 1
	vextractf64x4	ymm0, zmm0, 0x1
	vaddsd	xmm18, xmm18, xmm20
	vaddsd	xmm1, xmm1, xmm18
	vunpckhpd	xmm18, xmm0, xmm0
	vaddsd	xmm1, xmm0, xmm1
	vaddsd	xmm18, xmm18, xmm1
	vextractf64x2	xmm1, ymm0, 1
	valignq	ymm0, ymm0, ymm0, 3
	vaddsd	xmm18, xmm1, xmm18
	vaddsd	xmm18, xmm18, xmm0
	cmp	rax, r9
	jne	.L7
	cmp	r12d, ebx
	je	.L9
	mov	eax, r12d
	mov	r8d, r12d
.L6:
	mov	r14d, ebx
	sub	r14d, eax
	lea	r15d, -1[r14]
	cmp	r15d, 2
	jbe	.L11
	sal	rax, 3
	vbroadcastsd	ymm0, xmm10
	vbroadcastsd	ymm1, xmm9
	vbroadcastsd	ymm5, xmm4
	vmovupd	ymm6, YMMWORD PTR [rdi+rax]
	vsubpd	ymm0, ymm6, ymm0
	vmovupd	ymm6, YMMWORD PTR [rsi+rax]
	vdivpd	ymm0, ymm0, ymm12
	vsubpd	ymm1, ymm6, ymm1
	vmovupd	ymm6, YMMWORD PTR [rdx+rax]
	mov	eax, r14d
	and	eax, -4
	add	r8d, eax
	and	r14d, 3
	vdivpd	ymm1, ymm1, ymm12
	vmulpd	ymm2, ymm1, ymm1
	vfmadd231pd	ymm2, ymm0, ymm0
	vaddpd	ymm2, ymm2, ymm17
	vsqrtpd	ymm2, ymm2
	vdivpd	ymm2, ymm16, ymm2
	vmulpd	ymm20, ymm2, ymm2
	vmulpd	ymm20, ymm20, ymm2
	vmulpd	ymm0, ymm0, ymm20
	vmulpd	ymm0, ymm0, ymm6
	vmulpd	ymm0, ymm0, ymm5
	vaddsd	xmm2, xmm19, xmm0
	vunpckhpd	xmm21, xmm0, xmm0
	vextractf64x2	xmm19, ymm0, 1
	vaddsd	xmm2, xmm2, xmm21
	valignq	ymm0, ymm0, ymm0, 3
	vaddsd	xmm2, xmm2, xmm19
	vaddsd	xmm19, xmm2, xmm0
	vmulpd	ymm0, ymm1, ymm20
	vmulpd	ymm0, ymm0, ymm6
	vmulpd	ymm0, ymm0, ymm5
	vaddsd	xmm1, xmm18, xmm0
	vunpckhpd	xmm2, xmm0, xmm0
	vaddsd	xmm1, xmm1, xmm2
	vextractf64x2	xmm2, ymm0, 1
	valignq	ymm0, ymm0, ymm0, 3
	vaddsd	xmm1, xmm1, xmm2
	vaddsd	xmm18, xmm1, xmm0
	je	.L9
.L11:
	movsx	rax, r8d
	lea	r14d, 1[r8]
	sal	rax, 3
	vmovsd	xmm1, QWORD PTR [rdi+rax]
	vmovsd	xmm0, QWORD PTR [rsi+rax]
	vsubsd	xmm1, xmm1, xmm10
	vsubsd	xmm0, xmm0, xmm9
	vdivsd	xmm1, xmm1, xmm11
	vdivsd	xmm0, xmm0, xmm11
	vmulsd	xmm2, xmm0, xmm0
	vfmadd231sd	xmm2, xmm1, xmm1
	vaddsd	xmm2, xmm2, xmm23
	vsqrtsd	xmm2, xmm2, xmm2
	vdivsd	xmm2, xmm13, xmm2
	vmulsd	xmm5, xmm2, xmm2
	vmulsd	xmm2, xmm5, xmm2
	vmovsd	xmm5, QWORD PTR [rdx+rax]
	vmulsd	xmm1, xmm1, xmm2
	vmulsd	xmm0, xmm0, xmm2
	vmulsd	xmm1, xmm1, xmm5
	vmulsd	xmm0, xmm0, xmm5
	vfmadd231sd	xmm19, xmm4, xmm1
	vfmadd231sd	xmm18, xmm4, xmm0
	cmp	ebx, r14d
	jle	.L9
	vmovsd	xmm0, QWORD PTR 8[rdi+rax]
	vmovsd	xmm5, QWORD PTR 8[rsi+rax]
	vsubsd	xmm0, xmm0, xmm10
	vsubsd	xmm5, xmm5, xmm9
	vdivsd	xmm0, xmm0, xmm11
	vdivsd	xmm5, xmm5, xmm11
	vmulsd	xmm1, xmm5, xmm5
	vfmadd231sd	xmm1, xmm0, xmm0
	add	r8d, 2
	vaddsd	xmm1, xmm1, xmm23
	vsqrtsd	xmm1, xmm1, xmm1
	vdivsd	xmm1, xmm13, xmm1
	vmulsd	xmm2, xmm1, xmm1
	vmulsd	xmm1, xmm2, xmm1
	vmovsd	xmm2, QWORD PTR 8[rdx+rax]
	vmulsd	xmm0, xmm1, xmm0
	vmulsd	xmm1, xmm1, xmm5
	vmulsd	xmm0, xmm0, xmm2
	vmulsd	xmm1, xmm1, xmm2
	vfmadd231sd	xmm19, xmm4, xmm0
	vfmadd231sd	xmm18, xmm4, xmm1
	cmp	ebx, r8d
	jle	.L9
	vmovsd	xmm1, QWORD PTR 16[rdi+rax]
	vmovsd	xmm0, QWORD PTR 16[rsi+rax]
	vsubsd	xmm1, xmm1, xmm10
	vsubsd	xmm0, xmm0, xmm9
	vdivsd	xmm1, xmm1, xmm11
	vdivsd	xmm0, xmm0, xmm11
	vmulsd	xmm2, xmm0, xmm0
	vfmadd231sd	xmm2, xmm1, xmm1
	vaddsd	xmm2, xmm2, xmm23
	vsqrtsd	xmm2, xmm2, xmm2
	vdivsd	xmm2, xmm13, xmm2
	vmulsd	xmm5, xmm2, xmm2
	vmulsd	xmm2, xmm5, xmm2
	vmovsd	xmm5, QWORD PTR 16[rdx+rax]
	vmulsd	xmm1, xmm1, xmm2
	vmulsd	xmm0, xmm0, xmm2
	vmulsd	xmm1, xmm1, xmm5
	vmulsd	xmm0, xmm0, xmm5
	vfmadd231sd	xmm19, xmm4, xmm1
	vfmadd231sd	xmm18, xmm4, xmm0
.L9:
	vmovsd	xmm1, xmm18, xmm18
	vmovsd	xmm0, xmm19, xmm19
	vfmadd213sd	xmm1, xmm14, QWORD PTR 8[r10+rcx*8]
	vfmadd213sd	xmm0, xmm14, QWORD PTR [r10+rcx*8]
	vunpcklpd	xmm18, xmm19, xmm18
	vunpcklpd	xmm0, xmm0, xmm1
	vinsertf64x2	ymm0, ymm0, xmm18, 0x1
	vmovupd	YMMWORD PTR [r10+rcx*8], ymm0
	add	rcx, 8
	cmp	rcx, r11
	jne	.L5
	vzeroupper
.L20:
	lea	rsp, -40[rbp]
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4
	.p2align 3
.L23:
	.cfi_restore_state
	vxorpd	xmm18, xmm18, xmm18
	vmovsd	xmm19, xmm18, xmm18
	jmp	.L9
.L14:
	vxorpd	xmm18, xmm18, xmm18
	xor	eax, eax
	xor	r8d, r8d
	vmovsd	xmm19, xmm18, xmm18
	jmp	.L6
.L2:
	inc	eax
	xor	edx, edx
	jmp	.L13
	.cfi_endproc
.LFE15561:
	.size	_Z10calcForce2P4Bodyfi._omp_fn.0, .-_Z10calcForce2P4Bodyfi._omp_fn.0
	.p2align 4
	.type	_Z9calcForceP4Bodyfi._omp_fn.0, @function
_Z9calcForceP4Bodyfi._omp_fn.0:
.LFB15562:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r13
	push	r12
	push	rbx
	.cfi_offset 13, -24
	.cfi_offset 12, -32
	.cfi_offset 3, -40
	mov	ebx, DWORD PTR 12[rdi]
	and	rsp, -32
	mov	r12, rdi
	call	omp_get_num_threads@PLT
	mov	r13d, eax
	call	omp_get_thread_num@PLT
	mov	ecx, eax
	mov	eax, ebx
	cdq
	idiv	r13d
	cmp	ecx, edx
	jl	.L25
.L30:
	imul	ecx, eax
	add	edx, ecx
	lea	ecx, [rax+rdx]
	cmp	edx, ecx
	jge	.L34
	lea	r10d, -1[rbx]
	vbroadcastsd	ymm6, QWORD PTR .LC11[rip]
	vbroadcastsd	ymm8, QWORD PTR .LC5[rip]
	movsx	rdx, edx
	shr	r10d, 2
	vxorps	xmm9, xmm9, xmm9
	vcvtss2sd	xmm9, xmm9, DWORD PTR 8[r12]
	vmovsd	xmm10, QWORD PTR .LC1[rip]
	mov	r11, QWORD PTR [r12]
	mov	r12d, eax
	inc	r10d
	lea	r9, 0[0+rdx*8]
	add	r12, rdx
	lea	r8, masses[rip]
	lea	rdi, body_x[rip]
	sal	r12, 3
	sal	r10, 5
	lea	rsi, body_y[rip]
	.p2align 4
	.p2align 3
.L28:
	mov	rax, QWORD PTR [r8]
	vxorpd	xmm5, xmm5, xmm5
	vmovapd	ymm4, ymm5
	vdivsd	xmm7, xmm10, QWORD PTR [rax+r9]
	mov	rax, QWORD PTR [rdi]
	vbroadcastsd	ymm7, xmm7
	vbroadcastsd	ymm12, QWORD PTR [rax+r9]
	mov	rax, QWORD PTR [rsi]
	vbroadcastsd	ymm11, QWORD PTR [rax+r9]
	xor	eax, eax
	test	ebx, ebx
	jle	.L29
	.p2align 4
	.p2align 3
.L27:
#APP
# 177 "main.cpp" 1
	# === BEGIN INNER LOOP ===
# 0 "" 2
#NO_APP
	mov	rcx, QWORD PTR [rdi]
	mov	rdx, QWORD PTR [r8]
	vmovupd	ymm3, YMMWORD PTR [rcx+rax]
	mov	rcx, QWORD PTR [rsi]
	vmovupd	ymm2, YMMWORD PTR [rcx+rax]
	vsubpd	ymm3, ymm3, ymm12
	vmulpd	ymm3, ymm3, ymm6
	vsubpd	ymm2, ymm2, ymm11
	vmulpd	ymm2, ymm2, ymm6
	vmulpd	ymm0, ymm2, ymm2
	vfmadd231pd	ymm0, ymm3, ymm3
	vaddpd	ymm0, ymm0, ymm8
	vrsqrt14pd	ymm0, ymm0
	vmulpd	ymm1, ymm0, ymm0
	vmulpd	ymm0, ymm1, ymm0
	vmulpd	ymm0, ymm0, YMMWORD PTR [rdx+rax]
	vmulpd	ymm0, ymm0, ymm7
	vfmadd231pd	ymm4, ymm3, ymm0
	vfmadd231pd	ymm5, ymm2, ymm0
#APP
# 212 "main.cpp" 1
	# === END INNER LOOP ===
# 0 "" 2
#NO_APP
	add	rax, 32
	cmp	rax, r10
	jne	.L27
.L29:
	vmovapd	xmm0, xmm4
	vextractf64x2	xmm4, ymm4, 0x1
	vaddpd	xmm0, xmm0, xmm4
	vunpckhpd	xmm1, xmm0, xmm0
	vaddsd	xmm0, xmm0, xmm1
	vmovapd	xmm1, xmm5
	vextractf64x2	xmm5, ymm5, 0x1
	vaddpd	xmm1, xmm1, xmm5
	vunpckhpd	xmm2, xmm1, xmm1
	vaddsd	xmm1, xmm1, xmm2
#APP
# 219 "main.cpp" 1
	# === END CRITICAL LOOP ===
# 0 "" 2
#NO_APP
	vmovsd	xmm2, xmm0, xmm0
	vmovsd	xmm3, xmm1, xmm1
	vfmadd213sd	xmm2, xmm9, QWORD PTR [r11+r9*8]
	vfmadd213sd	xmm3, xmm9, QWORD PTR 8[r11+r9*8]
	vunpcklpd	xmm0, xmm0, xmm1
	vunpcklpd	xmm1, xmm2, xmm3
	vinsertf128	ymm0, ymm1, xmm0, 0x1
	vmovupd	YMMWORD PTR [r11+r9*8], ymm0
	add	r9, 8
	cmp	r12, r9
	jne	.L28
	vzeroupper
.L34:
	lea	rsp, -24[rbp]
	pop	rbx
	pop	r12
	pop	r13
	pop	rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L25:
	.cfi_restore_state
	inc	eax
	xor	edx, edx
	jmp	.L30
	.cfi_endproc
.LFE15562:
	.size	_Z9calcForceP4Bodyfi._omp_fn.0, .-_Z9calcForceP4Bodyfi._omp_fn.0
	.p2align 4
	.type	main._omp_fn.0, @function
main._omp_fn.0:
.LFB15563:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA15563
	endbr64
	push	r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	push	r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	push	r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	mov	ecx, 1
	push	r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	push	rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	push	rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	mov	r13, rdi
	sub	rsp, 136
	.cfi_def_cfa_offset 192
	movsx	rsi, DWORD PTR 24[rdi]
	vmovsd	xmm1, QWORD PTR 8[rdi]
	mov	ebx, DWORD PTR 32[rdi]
	mov	ebp, DWORD PTR 28[rdi]
	mov	r12, QWORD PTR 16[rdi]
	lea	r9, 88[rsp]
	lea	r8, 80[rsp]
	xor	edi, edi
	mov	edx, 1
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 120[rsp], rax
	xor	eax, eax
	vmovsd	QWORD PTR 24[rsp], xmm1
	mov	QWORD PTR 16[rsp], r9
	mov	QWORD PTR 8[rsp], r8
	call	GOMP_loop_nonmonotonic_dynamic_start@PLT
	test	al, al
	je	.L37
	cmp	ebx, 4
	mov	r8, QWORD PTR 8[rsp]
	mov	r9, QWORD PTR 16[rsp]
	vxorps	xmm2, xmm2, xmm2
	vmovsd	xmm1, QWORD PTR 24[rsp]
	jne	.L39
	cmp	ebp, 1
	jle	.L57
.L39:
	lea	r13, body_x[rip]
	lea	r14, body_y[rip]
	mov	rbx, r8
	mov	rbp, r9
	.p2align 4
	.p2align 3
.L41:
	mov	rax, QWORD PTR 80[rsp]
	mov	r9d, DWORD PTR 88[rsp]
	mov	r8, QWORD PTR 0[r13]
	mov	rdi, QWORD PTR [r14]
	mov	edx, eax
	cdqe
	sal	rax, 3
	.p2align 4
	.p2align 3
.L40:
	vcvttsd2si	esi, QWORD PTR [r12+rax*8]
	vcvtsi2sd	xmm0, xmm2, esi
	lea	rcx, [r8+rax]
	inc	edx
	vfmadd213sd	xmm0, xmm1, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm0
	lea	rcx, [rdi+rax]
	vcvttsd2si	esi, QWORD PTR 8[r12+rax*8]
	add	rax, 8
	vcvtsi2sd	xmm0, xmm2, esi
	vfmadd213sd	xmm0, xmm1, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm0
	cmp	edx, r9d
	jl	.L40
	mov	rsi, rbp
	mov	rdi, rbx
	vmovsd	QWORD PTR 8[rsp], xmm1
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vmovsd	xmm1, QWORD PTR 8[rsp]
	vxorps	xmm2, xmm2, xmm2
	test	al, al
	jne	.L41
.L37:
	call	GOMP_loop_end_nowait@PLT
	mov	rax, QWORD PTR 120[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L58
	add	rsp, 136
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	pop	rbx
	.cfi_def_cfa_offset 48
	pop	rbp
	.cfi_def_cfa_offset 40
	pop	r12
	.cfi_def_cfa_offset 32
	pop	r13
	.cfi_def_cfa_offset 24
	pop	r14
	.cfi_def_cfa_offset 16
	pop	r15
	.cfi_def_cfa_offset 8
	ret
.L57:
	.cfi_restore_state
	lea	rax, body_x[rip]
	lea	r15, masses[rip]
	mov	QWORD PTR 56[rsp], r12
	vmovsd	QWORD PTR 48[rsp], xmm1
	mov	QWORD PTR 32[rsp], rax
	lea	rax, body_y[rip]
	mov	QWORD PTR 64[rsp], r8
	mov	QWORD PTR 72[rsp], r9
	mov	QWORD PTR 40[rsp], rax
	lea	rax, 96[rsp]
	mov	QWORD PTR 24[rsp], rax
	.p2align 4
	.p2align 3
.L38:
	mov	eax, DWORD PTR 88[rsp]
	movsx	r12, DWORD PTR 80[rsp]
	mov	rbx, QWORD PTR 56[rsp]
	mov	DWORD PTR 16[rsp], eax
	mov	rax, r12
	mov	r14, r12
	sal	rax, 6
	lea	rbp, 32[rbx+rax]
	mov	rbx, rbp
	.p2align 4
	.p2align 3
.L42:
	mov	rax, QWORD PTR 32[rsp]
	vcvttsd2si	esi, QWORD PTR -32[rbx]
	vcvtsi2sd	xmm0, xmm2, esi
	lea	rdx, 0[0+r14*8]
	vmovsd	xmm3, QWORD PTR 48[rsp]
	inc	r14
	mov	r12, QWORD PTR [rax]
	mov	rax, QWORD PTR 40[rsp]
	mov	rsi, QWORD PTR [rax]
	add	r12, rdx
	vfmadd213sd	xmm0, xmm3, QWORD PTR [r12]
	vmovsd	QWORD PTR [r12], xmm0
	vcvttsd2si	edi, QWORD PTR -24[rbx]
	vcvtsi2sd	xmm0, xmm2, edi
	add	rsi, rdx
	vfmadd213sd	xmm0, xmm3, QWORD PTR [rsi]
	vmovsd	QWORD PTR [rsi], xmm0
	mov	rsi, QWORD PTR [r15]
	vcvttsd2si	ebp, QWORD PTR [rsi+rdx]
	call	round@PLT
	vmovsd	QWORD PTR 8[rsp], xmm0
	vmovsd	xmm0, QWORD PTR [r12]
	call	round@PLT
	mov	rcx, QWORD PTR 0[r13]
	vmovhpd	xmm0, xmm0, QWORD PTR 8[rsp]
	mov	QWORD PTR 112[rsp], 0
	mov	DWORD PTR 96[rsp], 50397184
	sub	rsp, 8
	.cfi_def_cfa_offset 200
	mov	edx, ebp
	vcvttpd2dq	xmm4, xmm0
	mov	r9d, 8
	vmovq	rsi, xmm4
	mov	r8d, -1
	mov	QWORD PTR 112[rsp], rcx
	push	0
	.cfi_def_cfa_offset 208
	mov	rdi, QWORD PTR 40[rsp]
	mov	rcx, rbx
	add	rbx, 64
	call	_ZN2cv6circleERKNS_17_InputOutputArrayENS_6Point_IiEEiRKNS_7Scalar_IdEEiii@PLT
	pop	rax
	.cfi_def_cfa_offset 200
	pop	rdx
	.cfi_def_cfa_offset 192
	cmp	DWORD PTR 16[rsp], r14d
	vxorps	xmm2, xmm2, xmm2
	jg	.L42
	mov	rsi, QWORD PTR 72[rsp]
	mov	rdi, QWORD PTR 64[rsp]
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vxorps	xmm2, xmm2, xmm2
	test	al, al
	jne	.L38
	jmp	.L37
.L58:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE15563:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA15563:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE15563-.LLSDACSB15563
.LLSDACSB15563:
.LLSDACSE15563:
	.text
	.size	main._omp_fn.0, .-main._omp_fn.0
	.section	.text._ZNSt6vectorIdSaIdEED2Ev,"axG",@progbits,_ZNSt6vectorIdSaIdEED5Ev,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt6vectorIdSaIdEED2Ev
	.type	_ZNSt6vectorIdSaIdEED2Ev, @function
_ZNSt6vectorIdSaIdEED2Ev:
.LFB15548:
	.cfi_startproc
	endbr64
	mov	rax, QWORD PTR [rdi]
	test	rax, rax
	je	.L61
	mov	rsi, QWORD PTR 16[rdi]
	mov	rdi, rax
	sub	rsi, rax
	jmp	_ZdlPvm@PLT
	.p2align 4
	.p2align 3
.L61:
	ret
	.cfi_endproc
.LFE15548:
	.size	_ZNSt6vectorIdSaIdEED2Ev, .-_ZNSt6vectorIdSaIdEED2Ev
	.weak	_ZNSt6vectorIdSaIdEED1Ev
	.set	_ZNSt6vectorIdSaIdEED1Ev,_ZNSt6vectorIdSaIdEED2Ev
	.text
	.p2align 4
	.globl	_Z10calcForce2P4Bodyfi
	.type	_Z10calcForce2P4Bodyfi, @function
_Z10calcForce2P4Bodyfi:
.LFB14018:
	.cfi_startproc
	endbr64
	sub	rsp, 40
	.cfi_def_cfa_offset 48
	xor	ecx, ecx
	mov	edx, 2
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 24[rsp], rax
	xor	eax, eax
	mov	DWORD PTR 12[rsp], esi
	mov	QWORD PTR [rsp], rdi
	vmovss	DWORD PTR 8[rsp], xmm0
	mov	rsi, rsp
	lea	rdi, _Z10calcForce2P4Bodyfi._omp_fn.0[rip]
	call	GOMP_parallel@PLT
	mov	rax, QWORD PTR 24[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L66
	add	rsp, 40
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L66:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE14018:
	.size	_Z10calcForce2P4Bodyfi, .-_Z10calcForce2P4Bodyfi
	.p2align 4
	.globl	_Z20horizontal_sum_m256dDv4_d
	.type	_Z20horizontal_sum_m256dDv4_d, @function
_Z20horizontal_sum_m256dDv4_d:
.LFB14019:
	.cfi_startproc
	endbr64
	vmovapd	xmm1, xmm0
	vextractf64x2	xmm0, ymm0, 0x1
	vaddpd	xmm0, xmm1, xmm0
	vunpckhpd	xmm1, xmm0, xmm0
	vaddsd	xmm0, xmm0, xmm1
	ret
	.cfi_endproc
.LFE14019:
	.size	_Z20horizontal_sum_m256dDv4_d, .-_Z20horizontal_sum_m256dDv4_d
	.p2align 4
	.globl	_Z9calcForceP4Bodyfi
	.type	_Z9calcForceP4Bodyfi, @function
_Z9calcForceP4Bodyfi:
.LFB14020:
	.cfi_startproc
	endbr64
	sub	rsp, 40
	.cfi_def_cfa_offset 48
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 24[rsp], rax
	xor	eax, eax
#APP
# 155 "main.cpp" 1
	# === BEGIN CRITICAL LOOP ===
# 0 "" 2
#NO_APP
	mov	DWORD PTR 12[rsp], esi
	mov	QWORD PTR [rsp], rdi
	vmovss	DWORD PTR 8[rsp], xmm0
	xor	ecx, ecx
	mov	rsi, rsp
	mov	edx, 2
	lea	rdi, _Z9calcForceP4Bodyfi._omp_fn.0[rip]
	call	GOMP_parallel@PLT
	mov	rax, QWORD PTR 24[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L72
	add	rsp, 40
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L72:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE14020:
	.size	_Z9calcForceP4Bodyfi, .-_Z9calcForceP4Bodyfi
	.section	.rodata._ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_.str1.1,"aMS",@progbits,1
.LC12:
	.string	"vector::_M_realloc_insert"
	.section	.text._ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_,"axG",@progbits,_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_
	.type	_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_, @function
_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_:
.LFB15153:
	.cfi_startproc
	endbr64
	push	r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	push	r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	push	r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	movabs	rcx, 1152921504606846975
	push	r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	push	rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	push	rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	sub	rsp, 40
	.cfi_def_cfa_offset 96
	mov	r12, QWORD PTR 8[rdi]
	mov	r13, QWORD PTR [rdi]
	mov	rax, r12
	sub	rax, r13
	sar	rax, 3
	cmp	rax, rcx
	je	.L96
	mov	r15, rsi
	mov	rbp, rdi
	mov	r14, rsi
	sub	r15, r13
	cmp	r13, r12
	je	.L97
	lea	rcx, [rax+rax]
	cmp	rcx, rax
	jb	.L88
	test	rcx, rcx
	jne	.L98
	xor	ebx, ebx
	xor	edi, edi
.L79:
	lea	rcx, 8[rdi+r15]
	sub	r12, r14
	vmovsd	xmm0, QWORD PTR [rdx]
	vmovq	xmm2, rdi
	lea	rdx, [rcx+r12]
	vmovsd	QWORD PTR [rdi+r15], xmm0
	vpinsrq	xmm1, xmm2, rdx, 1
	vmovdqa	XMMWORD PTR [rsp], xmm1
	test	r15, r15
	jg	.L99
	test	r12, r12
	jle	.L83
	mov	rdx, r12
	mov	rsi, r14
	mov	rdi, rcx
	call	memcpy@PLT
.L83:
	test	r13, r13
	jne	.L82
.L85:
	vmovdqa	xmm3, XMMWORD PTR [rsp]
	mov	QWORD PTR 16[rbp], rbx
	vmovdqu	XMMWORD PTR 0[rbp], xmm3
	add	rsp, 40
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	pop	rbx
	.cfi_def_cfa_offset 48
	pop	rbp
	.cfi_def_cfa_offset 40
	pop	r12
	.cfi_def_cfa_offset 32
	pop	r13
	.cfi_def_cfa_offset 24
	pop	r14
	.cfi_def_cfa_offset 16
	pop	r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4
	.p2align 3
.L88:
	.cfi_restore_state
	movabs	rbx, 9223372036854775800
.L78:
	mov	rdi, rbx
	mov	QWORD PTR [rsp], rdx
	call	_Znwm@PLT
	mov	rdx, QWORD PTR [rsp]
	mov	rdi, rax
	add	rbx, rax
	jmp	.L79
	.p2align 4
	.p2align 3
.L99:
	mov	rdx, r15
	mov	rsi, r13
	mov	QWORD PTR 24[rsp], rcx
	call	memmove@PLT
	test	r12, r12
	jg	.L100
.L82:
	mov	rsi, QWORD PTR 16[rbp]
	mov	rdi, r13
	sub	rsi, r13
	call	_ZdlPvm@PLT
	jmp	.L85
	.p2align 4
	.p2align 3
.L97:
	add	rax, 1
	jc	.L88
	movabs	rcx, 1152921504606846975
	cmp	rax, rcx
	mov	rbx, rcx
	cmovbe	rbx, rax
	sal	rbx, 3
	jmp	.L78
	.p2align 4
	.p2align 3
.L100:
	mov	rdi, QWORD PTR 24[rsp]
	mov	rsi, r14
	mov	rdx, r12
	call	memcpy@PLT
	mov	rsi, QWORD PTR 16[rbp]
	mov	rdi, r13
	sub	rsi, r13
	call	_ZdlPvm@PLT
	jmp	.L85
.L98:
	movabs	rax, 1152921504606846975
	cmp	rcx, rax
	cmova	rcx, rax
	lea	rbx, 0[0+rcx*8]
	jmp	.L78
.L96:
	lea	rdi, .LC12[rip]
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
.LFE15153:
	.size	_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_, .-_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_
	.text
	.p2align 4
	.globl	_Z16randomlyAllocateP4Bodyi
	.type	_Z16randomlyAllocateP4Bodyi, @function
_Z16randomlyAllocateP4Bodyi:
.LFB14017:
	.cfi_startproc
	endbr64
	push	r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	push	r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	push	r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	mov	r13, rdi
	push	r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	push	rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	push	rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	xor	edi, edi
	sub	rsp, 40
	.cfi_def_cfa_offset 96
	movsx	rbx, esi
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 24[rsp], rax
	xor	eax, eax
	call	srand@PLT
	test	ebx, ebx
	jle	.L101
	lea	r15, body_x[rip]
	movabs	rbp, 8589934597
	sal	rbx, 6
	lea	r12, 0[r13+rbx]
	mov	ebx, 1
	jmp	.L103
	.p2align 4
	.p2align 3
.L117:
	vmovsd	QWORD PTR [rsi], xmm0
	add	rsi, 8
	mov	QWORD PTR body_x[rip+8], rsi
.L105:
	imul	rbx, rbx, 16807
	vxorpd	xmm2, xmm2, xmm2
	vxorpd	xmm4, xmm4, xmm4
	mov	rax, rbx
	mul	rbp
	mov	rax, rbx
	sub	rax, rdx
	shr	rax
	add	rdx, rax
	shr	rdx, 30
	mov	rax, rdx
	sal	rax, 31
	sub	rax, rdx
	sub	rbx, rax
	imul	rsi, rbx, 16807
	lea	rax, -1[rbx]
	vcvtusi2sd	xmm1, xmm2, rax
	vaddsd	xmm1, xmm1, xmm4
	mov	rax, rsi
	mul	rbp
	mov	rax, rsi
	sub	rax, rdx
	shr	rax
	lea	rbx, [rdx+rax]
	shr	rbx, 30
	mov	rax, rbx
	sal	rax, 31
	sub	rax, rbx
	mov	rbx, rsi
	sub	rbx, rax
	lea	rax, -1[rbx]
	vcvtusi2sd	xmm0, xmm2, rax
	vfmadd132sd	xmm0, xmm1, QWORD PTR .LC15[rip]
	vdivsd	xmm0, xmm0, QWORD PTR .LC16[rip]
	vcomisd	xmm0, QWORD PTR .LC1[rip]
	jnb	.L113
	vfmadd132sd	xmm0, xmm4, QWORD PTR .LC17[rip]
.L106:
	mov	rsi, QWORD PTR body_y[rip+8]
	vmovsd	QWORD PTR 16[rsp], xmm0
	cmp	rsi, QWORD PTR body_y[rip+16]
	je	.L107
	vmovsd	QWORD PTR [rsi], xmm0
	add	rsi, 8
	mov	QWORD PTR body_y[rip+8], rsi
.L108:
	call	rand@PLT
	mov	rsi, QWORD PTR masses[rip+8]
	vxorpd	xmm5, xmm5, xmm5
	cdq
	shr	edx, 30
	add	eax, edx
	and	eax, 3
	sub	eax, edx
	inc	eax
	cmp	rsi, QWORD PTR masses[rip+16]
	vcvtsi2sd	xmm0, xmm5, eax
	vmovsd	QWORD PTR 16[rsp], xmm0
	je	.L109
	vmovsd	QWORD PTR [rsi], xmm0
	add	rsi, 8
	mov	QWORD PTR masses[rip+8], rsi
.L110:
	call	rand@PLT
	vxorpd	xmm6, xmm6, xmm6
	add	r13, 64
	mov	edx, eax
	cdqe
	imul	rax, rax, -2139062143
	mov	esi, edx
	sar	esi, 31
	shr	rax, 32
	add	eax, edx
	sar	eax, 7
	sub	eax, esi
	mov	esi, eax
	sal	esi, 8
	sub	esi, eax
	sub	edx, esi
	vcvtsi2sd	xmm0, xmm6, edx
	vmovsd	QWORD PTR 8[rsp], xmm0
	call	rand@PLT
	mov	r14d, eax
	call	rand@PLT
	vmovsd	xmm7, QWORD PTR 8[rsp]
	mov	QWORD PTR -8[r13], 0x000000000
	mov	esi, eax
	cdqe
	mov	edx, r14d
	vmovsd	QWORD PTR -16[r13], xmm7
	imul	rax, rax, -2139062143
	mov	edi, esi
	sar	edi, 31
	shr	rax, 32
	add	eax, esi
	sar	eax, 7
	sub	eax, edi
	mov	edi, eax
	sal	edi, 8
	sub	edi, eax
	movsx	rax, r14d
	imul	rax, rax, -2139062143
	sub	esi, edi
	vmovd	xmm0, esi
	mov	esi, r14d
	sar	esi, 31
	shr	rax, 32
	add	eax, r14d
	sar	eax, 7
	sub	eax, esi
	mov	esi, eax
	sal	esi, 8
	sub	esi, eax
	sub	edx, esi
	vpinsrd	xmm0, xmm0, edx, 1
	vcvtdq2pd	xmm0, xmm0
	vmovupd	XMMWORD PTR -32[r13], xmm0
	cmp	r12, r13
	je	.L101
.L103:
	imul	rbx, rbx, 16807
	vxorpd	xmm3, xmm3, xmm3
	vxorpd	xmm4, xmm4, xmm4
	mov	rax, rbx
	mul	rbp
	mov	rax, rbx
	sub	rax, rdx
	shr	rax
	add	rdx, rax
	shr	rdx, 30
	mov	rax, rdx
	sal	rax, 31
	sub	rax, rdx
	sub	rbx, rax
	imul	rsi, rbx, 16807
	lea	rax, -1[rbx]
	vcvtusi2sd	xmm1, xmm3, rax
	vaddsd	xmm1, xmm1, xmm4
	mov	rax, rsi
	mul	rbp
	mov	rax, rsi
	sub	rax, rdx
	shr	rax
	lea	rbx, [rdx+rax]
	shr	rbx, 30
	mov	rax, rbx
	sal	rax, 31
	sub	rax, rbx
	mov	rbx, rsi
	sub	rbx, rax
	lea	rax, -1[rbx]
	vcvtusi2sd	xmm0, xmm3, rax
	vfmadd132sd	xmm0, xmm1, QWORD PTR .LC15[rip]
	vdivsd	xmm0, xmm0, QWORD PTR .LC16[rip]
	vcomisd	xmm0, QWORD PTR .LC1[rip]
	jnb	.L114
	vfmadd132sd	xmm0, xmm4, QWORD PTR .LC18[rip]
.L111:
	mov	rsi, QWORD PTR body_x[rip+8]
	vmovsd	QWORD PTR 16[rsp], xmm0
	cmp	rsi, QWORD PTR body_x[rip+16]
	jne	.L117
	lea	rdx, 16[rsp]
	mov	rdi, r15
	call	_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_
	jmp	.L105
	.p2align 4
	.p2align 3
.L109:
	lea	rdx, 16[rsp]
	lea	rdi, masses[rip]
	call	_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_
	jmp	.L110
	.p2align 4
	.p2align 3
.L107:
	lea	rdx, 16[rsp]
	lea	rdi, body_y[rip]
	call	_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_
	jmp	.L108
	.p2align 4
	.p2align 3
.L101:
	mov	rax, QWORD PTR 24[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L118
	add	rsp, 40
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	pop	rbx
	.cfi_def_cfa_offset 48
	pop	rbp
	.cfi_def_cfa_offset 40
	pop	r12
	.cfi_def_cfa_offset 32
	pop	r13
	.cfi_def_cfa_offset 24
	pop	r14
	.cfi_def_cfa_offset 16
	pop	r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4
	.p2align 3
.L113:
	.cfi_restore_state
	vmovsd	xmm0, QWORD PTR .LC14[rip]
	jmp	.L106
	.p2align 4
	.p2align 3
.L114:
	vmovsd	xmm0, QWORD PTR .LC13[rip]
	jmp	.L111
.L118:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE14017:
	.size	_Z16randomlyAllocateP4Bodyi, .-_Z16randomlyAllocateP4Bodyi
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC20:
	.string	"Done with : %d\n"
.LC23:
	.string	"Epoch: "
.LC28:
	.string	"Epoch_"
.LC29:
	.string	".png"
.LC30:
	.string	"basic_string::append"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB35:
	.section	.text.startup,"ax",@progbits
.LHOTB35:
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB14021:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA14021
	endbr64
	lea	r10, 8[rsp]
	.cfi_def_cfa 10, 0
	and	rsp, -32
	mov	ecx, 16
	mov	edx, 1200
	push	QWORD PTR -8[r10]
	push	rbp
	mov	esi, 800
	mov	rbp, rsp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	push	r15
	push	r14
	push	r13
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	lea	r15, -576[rbp]
	push	r12
	push	r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	push	rbx
	sub	rsp, 768
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	mov	rdi, r15
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR -56[rbp], rax
	xor	eax, eax
.LEHB0:
	call	_ZN2cv3MatC1Eiii@PLT
.LEHE0:
	vmovapd	ymm0, YMMWORD PTR .LC19[rip]
	lea	rax, -624[rbp]
	mov	rdi, r15
	mov	rsi, rax
	mov	rbx, rax
	mov	QWORD PTR -776[rbp], rax
	vmovapd	YMMWORD PTR -624[rbp], ymm0
	vzeroupper
.LEHB1:
	call	_ZN2cv3MataSERKNS_7Scalar_IdEE@PLT
	lea	rax, -608[rbp]
	xor	edx, edx
	mov	rdi, rbx
	mov	QWORD PTR -720[rbp], 0
	mov	QWORD PTR -768[rbp], rax
	mov	QWORD PTR -624[rbp], rax
	lea	rax, -744[rbp]
	mov	DWORD PTR -736[rbp], 16842752
	mov	rsi, rax
	mov	QWORD PTR -728[rbp], r15
	mov	QWORD PTR -744[rbp], 30
	mov	QWORD PTR -800[rbp], rax
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@PLT
.LEHE1:
	vmovdqa	xmm0, XMMWORD PTR .LC32[rip]
	mov	rdx, QWORD PTR -744[rbp]
	mov	QWORD PTR -624[rbp], rax
	lea	r12, -736[rbp]
	mov	rdi, QWORD PTR -776[rbp]
	mov	rsi, r12
	mov	QWORD PTR -608[rbp], rdx
	vmovdqu	XMMWORD PTR [rax], xmm0
	vmovdqa	xmm0, XMMWORD PTR .LC33[rip]
	vmovdqu	XMMWORD PTR 14[rax], xmm0
	mov	rax, QWORD PTR -744[rbp]
	mov	rdx, QWORD PTR -624[rbp]
	mov	QWORD PTR -616[rbp], rax
	mov	BYTE PTR [rdx+rax], 0
.LEHB2:
	call	_ZN2cv6imshowERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKNS_11_InputArrayE@PLT
.LEHE2:
	mov	rdi, QWORD PTR -776[rbp]
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@PLT
	mov	edi, 396288
	call	malloc@PLT
	mov	esi, 6192
	mov	rdi, rax
	mov	r13, rax
.LEHB3:
	call	_Z16randomlyAllocateP4Bodyi
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	xor	r14d, r14d
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	lea	rax, -272[rbp]
	mov	QWORD PTR -792[rbp], rax
	mov	QWORD PTR -760[rbp], 0
	.p2align 4
	.p2align 3
.L166:
	mov	rax, QWORD PTR -760[rbp]
	mov	ecx, eax
	imul	eax, eax, -858993459
	rorx	eax, eax, 1
	cmp	eax, 429496729
	ja	.L120
.L177:
	mov	edx, ecx
	lea	rsi, .LC20[rip]
	mov	edi, 2
	xor	eax, eax
	call	__printf_chk@PLT
.L120:
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	cmp	r14d, 1
	jle	.L179
.L178:
	xor	ebx, ebx
	.p2align 4
	.p2align 3
.L122:
	movabs	rax, 26595454867210
	xor	ecx, ecx
	mov	edx, 2
	mov	rsi, r12
	lea	rdi, _Z10calcForce2P4Bodyfi._omp_fn.0[rip]
	mov	QWORD PTR -728[rbp], rax
	mov	QWORD PTR -736[rbp], r13
	call	GOMP_parallel@PLT
	mov	rax, QWORD PTR .LC21[rip]
	mov	DWORD PTR -704[rbp], ebx
	xor	ecx, ecx
	xor	edx, edx
	mov	rsi, r12
	lea	rdi, main._omp_fn.0[rip]
	mov	DWORD PTR -708[rbp], r14d
	inc	ebx
	mov	QWORD PTR -720[rbp], r13
	mov	DWORD PTR -712[rbp], 6192
	mov	QWORD PTR -736[rbp], r15
	mov	QWORD PTR -728[rbp], rax
	call	GOMP_parallel@PLT
	cmp	ebx, 5
	jne	.L122
	cmp	r14d, 1
	jle	.L200
.L123:
	mov	edi, 30
	call	_ZN2cv7waitKeyEi@PLT
	cmp	al, 113
	je	.L167
	test	r14d, r14d
	jne	.L133
	cmp	al, 112
	je	.L131
.L135:
	xor	r14d, r14d
.L132:
	inc	QWORD PTR -760[rbp]
	mov	rax, QWORD PTR -760[rbp]
	add	QWORD PTR -792[rbp], 2
	cmp	rax, 100
	jne	.L166
	.p2align 4
	.p2align 3
.L167:
	mov	rdi, r15
	call	_ZN2cv3MatD1Ev@PLT
	mov	rax, QWORD PTR -56[rbp]
	sub	rax, QWORD PTR fs:40
	jne	.L198
	lea	rsp, -48[rbp]
	xor	eax, eax
	pop	rbx
	pop	r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	lea	rsp, -8[r10]
	.cfi_def_cfa 7, 8
	ret
.L153:
	.cfi_restore_state
	inc	QWORD PTR -760[rbp]
	mov	rax, QWORD PTR -760[rbp]
	mov	ebx, 10
	add	QWORD PTR -792[rbp], 2
	cdq
	mov	ecx, eax
	idiv	ebx
	mov	r14d, edx
	test	edx, edx
	je	.L177
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	xor	r14d, r14d
	.p2align 4
	.p2align 3
.L179:
	mov	rsi, QWORD PTR -776[rbp]
	vxorpd	xmm0, xmm0, xmm0
	mov	rdi, r15
	vmovapd	YMMWORD PTR -624[rbp], ymm0
	vzeroupper
	call	_ZN2cv3MataSERKNS_7Scalar_IdEE@PLT
	jmp	.L178
	.p2align 4
	.p2align 3
.L131:
	test	r14d, r14d
	je	.L152
.L133:
	dec	r14d
	jne	.L132
.L152:
	cmp	DWORD PTR -760[rbp], 9
	jle	.L201
	.p2align 4
	.p2align 3
.L151:
	xor	edi, edi
	call	_ZN2cv7waitKeyEi@PLT
.LEHE3:
	cmp	al, 112
	je	.L135
	cmp	al, 113
	je	.L167
	cmp	al, 115
	je	.L136
	cmp	al, 110
	jne	.L151
	mov	r14d, 5
	jmp	.L132
	.p2align 4
	.p2align 3
.L136:
	lea	rbx, -656[rbp]
	lea	rax, -640[rbp]
	mov	edx, 45
	mov	esi, 2
	mov	rdi, rbx
	mov	QWORD PTR -784[rbp], rax
	mov	QWORD PTR -656[rbp], rax
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@PLT
	vmovdqa64	zmm1, ZMMWORD PTR .LC25[rip]
	vmovdqa64	zmm2, ZMMWORD PTR .LC26[rip]
	mov	rax, QWORD PTR -792[rbp]
	mov	r8d, 6
	lea	rcx, .LC28[rip]
	xor	esi, esi
	vmovdqa64	zmm3, ZMMWORD PTR .LC27[rip]
	vmovdqa	xmm4, XMMWORD PTR .LC34[rip]
	mov	rdi, rbx
	vmovdqu64	ZMMWORD PTR -272[rbp], zmm1
	vmovdqu64	ZMMWORD PTR -208[rbp], zmm2
	vmovdqu64	ZMMWORD PTR -144[rbp], zmm3
	vmovdqu	XMMWORD PTR -87[rbp], xmm4
	movzx	edx, WORD PTR [rax]
	mov	rax, QWORD PTR -656[rbp]
	mov	WORD PTR [rax], dx
	xor	edx, edx
	vzeroupper
.LEHB4:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm@PLT
.LEHE4:
	mov	rcx, QWORD PTR -768[rbp]
	lea	rdx, 16[rax]
	mov	rsi, QWORD PTR 8[rax]
	mov	QWORD PTR -624[rbp], rcx
	mov	rcx, QWORD PTR [rax]
	cmp	rcx, rdx
	je	.L202
	mov	QWORD PTR -624[rbp], rcx
	mov	rcx, QWORD PTR 16[rax]
	mov	QWORD PTR -608[rbp], rcx
.L150:
	mov	rcx, QWORD PTR 8[rax]
	mov	QWORD PTR -616[rbp], rcx
	mov	QWORD PTR 8[rax], 0
	mov	BYTE PTR 16[rax], 0
	mov	QWORD PTR [rax], rdx
	movabs	rax, 4611686018427387903
	sub	rax, QWORD PTR -616[rbp]
	cmp	rax, 3
	jbe	.L140
	mov	rdi, QWORD PTR -776[rbp]
	mov	edx, 4
	lea	rsi, .LC29[rip]
.LEHB5:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm@PLT
.LEHE5:
	lea	rbx, -672[rbp]
	lea	r8, 16[rax]
	mov	rcx, QWORD PTR 8[rax]
	mov	QWORD PTR -688[rbp], rbx
	mov	rdx, QWORD PTR [rax]
	cmp	r8, rdx
	je	.L203
	mov	QWORD PTR -688[rbp], rdx
	mov	rdx, QWORD PTR 16[rax]
	mov	QWORD PTR -672[rbp], rdx
.L149:
	mov	rdx, QWORD PTR 8[rax]
	mov	QWORD PTR -680[rbp], rdx
	mov	QWORD PTR [rax], r8
	mov	BYTE PTR 16[rax], 0
	mov	QWORD PTR 8[rax], 0
	mov	rdi, QWORD PTR -624[rbp]
	mov	rax, QWORD PTR -768[rbp]
	cmp	rdi, rax
	je	.L148
	mov	rax, QWORD PTR -608[rbp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
.L148:
	mov	rdi, QWORD PTR -656[rbp]
	mov	rax, QWORD PTR -784[rbp]
	cmp	rdi, rax
	je	.L147
	mov	rax, QWORD PTR -640[rbp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
.L147:
	mov	rdi, QWORD PTR -688[rbp]
	cmp	rdi, rbx
	je	.L151
	mov	rax, QWORD PTR -672[rbp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
	jmp	.L151
	.p2align 4
	.p2align 3
.L200:
	vmovapd	ymm5, YMMWORD PTR .LC22[rip]
	cmp	DWORD PTR -760[rbp], 9
	lea	rax, -640[rbp]
	lea	rbx, -656[rbp]
	mov	edx, 45
	mov	QWORD PTR -784[rbp], rax
	mov	QWORD PTR -656[rbp], rax
	vmovapd	YMMWORD PTR -688[rbp], ymm5
	jle	.L204
	mov	esi, 2
	mov	rdi, rbx
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@PLT
	vmovdqa64	zmm6, ZMMWORD PTR .LC25[rip]
	vmovdqa64	zmm7, ZMMWORD PTR .LC26[rip]
	mov	rsi, QWORD PTR -760[rbp]
	mov	rax, QWORD PTR -656[rbp]
	vmovdqu64	ZMMWORD PTR -480[rbp], zmm6
	vmovdqa64	zmm6, ZMMWORD PTR .LC27[rip]
	vmovdqu64	ZMMWORD PTR -416[rbp], zmm7
	vmovdqa	xmm7, XMMWORD PTR .LC34[rip]
	vmovdqu64	ZMMWORD PTR -352[rbp], zmm6
	vmovdqu	XMMWORD PTR -295[rbp], xmm7
	movzx	edx, BYTE PTR -479[rbp+rsi*2]
	mov	BYTE PTR 1[rax], dl
	movzx	edx, BYTE PTR -480[rbp+rsi*2]
	vzeroupper
.L176:
	mov	BYTE PTR [rax], dl
	mov	r8d, 7
	lea	rcx, .LC23[rip]
	xor	edx, edx
	xor	esi, esi
	mov	rdi, rbx
.LEHB6:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm@PLT
.LEHE6:
	mov	rdx, QWORD PTR -768[rbp]
	mov	rsi, QWORD PTR 8[rax]
	mov	QWORD PTR -624[rbp], rdx
	mov	rcx, QWORD PTR [rax]
	lea	rdx, 16[rax]
	cmp	rcx, rdx
	je	.L205
	mov	QWORD PTR -624[rbp], rcx
	mov	rcx, QWORD PTR 16[rax]
	mov	QWORD PTR -608[rbp], rcx
.L126:
	mov	rcx, QWORD PTR 8[rax]
	mov	rsi, QWORD PTR -776[rbp]
	mov	r9d, 1
	lea	r8, -688[rbp]
	mov	rdi, r12
	vmovsd	xmm0, QWORD PTR .LC24[rip]
	mov	QWORD PTR -616[rbp], rcx
	mov	QWORD PTR [rax], rdx
	mov	QWORD PTR 8[rax], 0
	mov	BYTE PTR 16[rax], 0
	mov	ecx, 3
	mov	QWORD PTR -720[rbp], 0
	mov	DWORD PTR -736[rbp], 50397184
	mov	QWORD PTR -728[rbp], r15
	movabs	rdx, 128849018890
	push	0
	push	8
.LEHB7:
	.cfi_escape 0x2e,0x10
	call	_ZN2cv7putTextERKNS_17_InputOutputArrayERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEENS_6Point_IiEEidNS_7Scalar_IdEEiib@PLT
.LEHE7:
	pop	rsi
	mov	rax, QWORD PTR -768[rbp]
	pop	rdi
	mov	rdi, QWORD PTR -624[rbp]
	cmp	rdi, rax
	je	.L127
	mov	rax, QWORD PTR -608[rbp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
.L127:
	mov	rdi, QWORD PTR -656[rbp]
	mov	rax, QWORD PTR -784[rbp]
	cmp	rdi, rax
	je	.L128
	mov	rax, QWORD PTR -640[rbp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
.L128:
	mov	rbx, QWORD PTR -768[rbp]
	mov	rsi, QWORD PTR -800[rbp]
	xor	edx, edx
	mov	QWORD PTR -720[rbp], 0
	mov	rdi, QWORD PTR -776[rbp]
	mov	DWORD PTR -736[rbp], 16842752
	mov	QWORD PTR -728[rbp], r15
	mov	QWORD PTR -744[rbp], 30
	mov	QWORD PTR -624[rbp], rbx
.LEHB8:
	.cfi_escape 0x2e,0
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@PLT
.LEHE8:
	vmovdqa	xmm0, XMMWORD PTR .LC32[rip]
	mov	rdx, QWORD PTR -744[rbp]
	mov	QWORD PTR -624[rbp], rax
	mov	rsi, r12
	mov	rdi, QWORD PTR -776[rbp]
	mov	QWORD PTR -608[rbp], rdx
	vmovdqu	XMMWORD PTR [rax], xmm0
	vmovdqa	xmm0, XMMWORD PTR .LC33[rip]
	vmovdqu	XMMWORD PTR 14[rax], xmm0
	mov	rax, QWORD PTR -744[rbp]
	mov	rdx, QWORD PTR -624[rbp]
	mov	QWORD PTR -616[rbp], rax
	mov	BYTE PTR [rdx+rax], 0
.LEHB9:
	call	_ZN2cv6imshowERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKNS_11_InputArrayE@PLT
.LEHE9:
	mov	rdi, QWORD PTR -624[rbp]
	cmp	rdi, rbx
	je	.L123
	mov	rax, QWORD PTR -608[rbp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
	jmp	.L123
.L204:
	mov	esi, 1
	mov	rdi, rbx
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@PLT
	movzx	ecx, BYTE PTR -760[rbp]
	mov	rax, QWORD PTR -656[rbp]
	lea	edx, 48[rcx]
	jmp	.L176
.L203:
	inc	ecx
	mov	rdi, rbx
	mov	rsi, r8
	rep movsb
	jmp	.L149
.L202:
	mov	rdi, QWORD PTR -768[rbp]
	lea	ecx, 1[rsi]
	mov	rsi, rdx
	rep movsb
	jmp	.L150
.L205:
	mov	rdi, QWORD PTR -768[rbp]
	lea	ecx, 1[rsi]
	mov	rsi, rdx
	rep movsb
	jmp	.L126
.L201:
	lea	r14, -656[rbp]
.L134:
	xor	edi, edi
.LEHB10:
	call	_ZN2cv7waitKeyEi@PLT
.LEHE10:
	cmp	al, 112
	je	.L153
	cmp	al, 113
	je	.L167
	cmp	al, 115
	je	.L206
	cmp	al, 110
	jne	.L134
	inc	QWORD PTR -760[rbp]
	mov	rax, QWORD PTR -760[rbp]
	mov	esi, 10
	add	QWORD PTR -792[rbp], 2
	cdq
	mov	ecx, eax
	idiv	esi
	test	edx, edx
	je	.L182
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	mov	r14d, 5
	jmp	.L178
.L206:
	lea	rax, -640[rbp]
	mov	edx, 45
	mov	esi, 1
	mov	rdi, r14
	mov	QWORD PTR -784[rbp], rax
	mov	QWORD PTR -656[rbp], rax
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@PLT
	movzx	eax, BYTE PTR -760[rbp]
	mov	r8d, 6
	lea	rcx, .LC28[rip]
	xor	esi, esi
	mov	rdi, r14
	mov	rbx, r14
	lea	edx, 48[rax]
	mov	rax, QWORD PTR -656[rbp]
	mov	BYTE PTR [rax], dl
	xor	edx, edx
.LEHB11:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm@PLT
.LEHE11:
	mov	rdx, QWORD PTR -768[rbp]
	mov	rsi, QWORD PTR 8[rax]
	mov	QWORD PTR -624[rbp], rdx
	mov	rcx, QWORD PTR [rax]
	lea	rdx, 16[rax]
	cmp	rcx, rdx
	je	.L207
	mov	QWORD PTR -624[rbp], rcx
	mov	rcx, QWORD PTR 16[rax]
	mov	QWORD PTR -608[rbp], rcx
.L156:
	mov	rcx, QWORD PTR 8[rax]
	mov	QWORD PTR -616[rbp], rcx
	xor	ecx, ecx
	mov	BYTE PTR 16[rax], 0
	mov	QWORD PTR [rax], rdx
	mov	QWORD PTR 8[rax], rcx
	movabs	rax, 4611686018427387903
	sub	rax, QWORD PTR -616[rbp]
	cmp	rax, 3
	jbe	.L140
	mov	rdi, QWORD PTR -776[rbp]
	mov	edx, 4
	lea	rsi, .LC29[rip]
.LEHB12:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm@PLT
	lea	rbx, -672[rbp]
	lea	rdx, 16[rax]
	mov	rsi, QWORD PTR 8[rax]
	mov	QWORD PTR -688[rbp], rbx
	mov	rcx, QWORD PTR [rax]
	cmp	rcx, rdx
	je	.L208
	mov	QWORD PTR -688[rbp], rcx
	mov	rcx, QWORD PTR 16[rax]
	mov	QWORD PTR -672[rbp], rcx
.L160:
	mov	rcx, QWORD PTR 8[rax]
	mov	QWORD PTR -680[rbp], rcx
	mov	QWORD PTR [rax], rdx
	xor	edx, edx
	mov	BYTE PTR 16[rax], 0
	mov	QWORD PTR 8[rax], rdx
	mov	rdi, QWORD PTR -624[rbp]
	mov	rax, QWORD PTR -768[rbp]
	cmp	rdi, rax
	je	.L161
	mov	rax, QWORD PTR -608[rbp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
.L161:
	mov	rdi, QWORD PTR -656[rbp]
	mov	rax, QWORD PTR -784[rbp]
	cmp	rdi, rax
	je	.L162
	mov	rax, QWORD PTR -640[rbp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
.L162:
	mov	rdi, QWORD PTR -688[rbp]
	cmp	rdi, rbx
	je	.L134
	mov	rax, QWORD PTR -672[rbp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
	jmp	.L134
.L208:
	lea	ecx, 1[rsi]
	mov	rdi, rbx
	mov	rsi, rdx
	rep movsb
	jmp	.L160
.L207:
	mov	rdi, QWORD PTR -768[rbp]
	lea	ecx, 1[rsi]
	mov	rsi, rdx
	rep movsb
	jmp	.L156
.L182:
	mov	r14d, 5
	jmp	.L177
.L198:
	call	__stack_chk_fail@PLT
.L140:
	mov	rax, QWORD PTR -56[rbp]
	sub	rax, QWORD PTR fs:40
	jne	.L198
	lea	rdi, .LC30[rip]
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE12:
.L189:
	endbr64
	mov	r12, rax
	jmp	.L173
.L184:
	endbr64
	mov	rbx, rax
	jmp	.L168
.L183:
	endbr64
	mov	rbx, rax
	vzeroupper
	jmp	.L169
.L187:
	endbr64
	mov	rbx, rax
	jmp	.L172
.L188:
	endbr64
	mov	r12, rax
	vzeroupper
	jmp	.L174
.L185:
	endbr64
	mov	r12, rax
	vzeroupper
	jmp	.L171
.L186:
	endbr64
	mov	r12, rax
	jmp	.L170
	.section	.gcc_except_table
.LLSDA14021:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE14021-.LLSDACSB14021
.LLSDACSB14021:
	.uleb128 .LEHB0-.LFB14021
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB14021
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L183-.LFB14021
	.uleb128 0
	.uleb128 .LEHB2-.LFB14021
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L184-.LFB14021
	.uleb128 0
	.uleb128 .LEHB3-.LFB14021
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L183-.LFB14021
	.uleb128 0
	.uleb128 .LEHB4-.LFB14021
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L188-.LFB14021
	.uleb128 0
	.uleb128 .LEHB5-.LFB14021
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L189-.LFB14021
	.uleb128 0
	.uleb128 .LEHB6-.LFB14021
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L185-.LFB14021
	.uleb128 0
	.uleb128 .LEHB7-.LFB14021
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L186-.LFB14021
	.uleb128 0
	.uleb128 .LEHB8-.LFB14021
	.uleb128 .LEHE8-.LEHB8
	.uleb128 .L183-.LFB14021
	.uleb128 0
	.uleb128 .LEHB9-.LFB14021
	.uleb128 .LEHE9-.LEHB9
	.uleb128 .L187-.LFB14021
	.uleb128 0
	.uleb128 .LEHB10-.LFB14021
	.uleb128 .LEHE10-.LEHB10
	.uleb128 .L183-.LFB14021
	.uleb128 0
	.uleb128 .LEHB11-.LFB14021
	.uleb128 .LEHE11-.LEHB11
	.uleb128 .L188-.LFB14021
	.uleb128 0
	.uleb128 .LEHB12-.LFB14021
	.uleb128 .LEHE12-.LEHB12
	.uleb128 .L189-.LFB14021
	.uleb128 0
.LLSDACSE14021:
	.section	.text.startup
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC14021
	.type	main.cold, @function
main.cold:
.LFSB14021:
.L173:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	mov	rdi, QWORD PTR -776[rbp]
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@PLT
.L174:
	mov	rdi, rbx
	mov	rbx, r12
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@PLT
	jmp	.L169
.L168:
	mov	rdi, QWORD PTR -776[rbp]
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@PLT
.L169:
	mov	rdi, r15
	call	_ZN2cv3MatD1Ev@PLT
	mov	rax, QWORD PTR -56[rbp]
	sub	rax, QWORD PTR fs:40
	jne	.L209
	mov	rdi, rbx
.LEHB13:
	call	_Unwind_Resume@PLT
.LEHE13:
.L172:
	mov	rdi, QWORD PTR -776[rbp]
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@PLT
	jmp	.L169
.L170:
	mov	rdi, QWORD PTR -776[rbp]
	vzeroupper
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@PLT
.L171:
	mov	rdi, rbx
	mov	rbx, r12
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@PLT
	jmp	.L169
.L209:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE14021:
	.section	.gcc_except_table
.LLSDAC14021:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC14021-.LLSDACSBC14021
.LLSDACSBC14021:
	.uleb128 .LEHB13-.LCOLDB35
	.uleb128 .LEHE13-.LEHB13
	.uleb128 0
	.uleb128 0
.LLSDACSEC14021:
	.section	.text.unlikely
	.section	.text.startup
	.size	main, .-main
	.section	.text.unlikely
	.size	main.cold, .-main.cold
.LCOLDE35:
	.section	.text.startup
.LHOTE35:
	.p2align 4
	.type	_GLOBAL__sub_I_body_x, @function
_GLOBAL__sub_I_body_x:
.LFB15560:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	lea	rbp, __dso_handle[rip]
	push	rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	lea	rbx, _ZNSt6vectorIdSaIdEED1Ev[rip]
	sub	rsp, 8
	.cfi_def_cfa_offset 32
	mov	rdx, rbp
	mov	rdi, rbx
	lea	rsi, body_x[rip]
	mov	QWORD PTR body_x[rip], 0
	mov	QWORD PTR body_x[rip+8], 0
	mov	QWORD PTR body_x[rip+16], 0
	call	__cxa_atexit@PLT
	mov	rdx, rbp
	mov	rdi, rbx
	lea	rsi, body_y[rip]
	mov	QWORD PTR body_y[rip], 0
	mov	QWORD PTR body_y[rip+8], 0
	mov	QWORD PTR body_y[rip+16], 0
	call	__cxa_atexit@PLT
	mov	QWORD PTR masses[rip], 0
	mov	QWORD PTR masses[rip+8], 0
	mov	QWORD PTR masses[rip+16], 0
	add	rsp, 8
	.cfi_def_cfa_offset 24
	mov	rdx, rbp
	mov	rdi, rbx
	pop	rbx
	.cfi_def_cfa_offset 16
	pop	rbp
	.cfi_def_cfa_offset 8
	lea	rsi, masses[rip]
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE15560:
	.size	_GLOBAL__sub_I_body_x, .-_GLOBAL__sub_I_body_x
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_body_x
	.globl	masses
	.bss
	.align 16
	.type	masses, @object
	.size	masses, 24
masses:
	.zero	24
	.globl	body_y
	.align 16
	.type	body_y, @object
	.size	body_y, 24
body_y:
	.zero	24
	.globl	body_x
	.align 16
	.type	body_x, @object
	.size	body_x, 24
body_x:
	.zero	24
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC1:
	.long	0
	.long	1072693248
	.align 8
.LC3:
	.long	0
	.long	1079574528
	.align 8
.LC5:
	.long	0
	.long	1073741824
	.align 8
.LC11:
	.long	1202590843
	.long	1065646817
	.align 8
.LC13:
	.long	-1
	.long	1083359231
	.align 8
.LC14:
	.long	-1
	.long	1082720255
	.align 8
.LC15:
	.long	-8388608
	.long	1105199103
	.align 8
.LC16:
	.long	-16777216
	.long	1137704959
	.align 8
.LC17:
	.long	0
	.long	1082720256
	.align 8
.LC18:
	.long	0
	.long	1083359232
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC19:
	.long	0
	.long	1080033280
	.long	0
	.long	1080033280
	.long	0
	.long	1080033280
	.long	0
	.long	0
	.section	.rodata.cst8
	.align 8
.LC21:
	.long	1202590843
	.long	1066695393
	.section	.rodata.cst32
	.align 32
.LC22:
	.long	0
	.long	1081073664
	.long	0
	.long	1081073664
	.long	0
	.long	1081073664
	.long	0
	.long	0
	.section	.rodata.cst8
	.align 8
.LC24:
	.long	0
	.long	1071644672
	.section	.rodata
	.align 64
.LC25:
	.quad	3688503277381496880
	.quad	3976738051646829616
	.quad	3544667369688283184
	.quad	3832902143785906737
	.quad	4121136918051239473
	.quad	3689066235924983858
	.quad	3977301010190316594
	.quad	3545230328231770162
	.align 64
.LC26:
	.quad	3833465102329393715
	.quad	4121699876594726451
	.quad	3689629194468470836
	.quad	3977863968733803572
	.quad	3545793286775257140
	.quad	3834028060872880693
	.quad	4122262835138213429
	.quad	3690192153011957814
	.align 64
.LC27:
	.quad	3978426927277290550
	.quad	3546356245318744118
	.quad	3834591019416367671
	.quad	4122825793681700407
	.quad	3690755111555444792
	.quad	3978989885820777528
	.quad	3546919203862231096
	.quad	3835153977959854649
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC32:
	.quad	7163384716917222734
	.quad	8530224819029108076
	.align 16
.LC33:
	.quad	7589445503065618017
	.quad	7957695015158576493
	.align 16
.LC34:
	.quad	4122263930388298034
	.quad	16106987313379638
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
