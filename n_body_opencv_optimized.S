	.file	"main.cpp"
	.intel_syntax noprefix
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.p2align 4
	.type	_Z10calcForce2P4Bodyfi._omp_fn.0, @function
_Z10calcForce2P4Bodyfi._omp_fn.0:
.LFB15480:
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
.LFE15480:
	.size	_Z10calcForce2P4Bodyfi._omp_fn.0, .-_Z10calcForce2P4Bodyfi._omp_fn.0
	.p2align 4
	.type	_Z9calcForceP4Bodyfi._omp_fn.0, @function
_Z9calcForceP4Bodyfi._omp_fn.0:
.LFB15481:
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
	mov	ebx, DWORD PTR 76[rdi]
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
	lea	r9d, -1[rbx]
	vmovapd	ymm6, YMMWORD PTR 32[r12]
	vmovapd	ymm12, YMMWORD PTR [r12]
	mov	r10, QWORD PTR 64[r12]
	movsx	rdx, edx
	mov	r11d, eax
	vxorps	xmm13, xmm13, xmm13
	vcvtss2sd	xmm13, xmm13, DWORD PTR 72[r12]
	shr	r9d, 3
	add	r11, rdx
	inc	r9d
	lea	r8, 0[0+rdx*8]
	sal	r11, 3
	sal	r9, 6
	.p2align 4
	.p2align 3
.L28:
	mov	rax, QWORD PTR body_x[rip]
	vbroadcastsd	ymm9, QWORD PTR [rax+r8]
	mov	rax, QWORD PTR body_y[rip]
	vbroadcastsd	ymm10, QWORD PTR [rax+r8]
	mov	rax, QWORD PTR inv_masses[rip]
	vbroadcastsd	ymm11, QWORD PTR [rax+r8]
	test	ebx, ebx
	jle	.L36
	vxorpd	xmm5, xmm5, xmm5
	xor	eax, eax
	vmovapd	ymm4, ymm5
	vmovapd	ymm8, ymm5
	vmovapd	ymm7, ymm5
	.p2align 4
	.p2align 3
.L27:
#APP
# 188 "main.cpp" 1
	# === BEGIN INNER LOOP ===
# 0 "" 2
#NO_APP
	mov	rdi, QWORD PTR body_x[rip]
	mov	rsi, QWORD PTR body_y[rip]
	mov	rcx, QWORD PTR masses[rip]
	vmovapd	ymm1, YMMWORD PTR [rdi+rax]
	vsubpd	ymm3, ymm1, ymm9
	vmovapd	ymm1, YMMWORD PTR [rsi+rax]
	vmulpd	ymm3, ymm3, ymm6
	vsubpd	ymm2, ymm1, ymm10
	vmulpd	ymm2, ymm2, ymm6
	vmulpd	ymm0, ymm2, ymm2
	vfmadd231pd	ymm0, ymm3, ymm3
	vaddpd	ymm0, ymm12, ymm0
	vrsqrt14pd	ymm0, ymm0
	vmulpd	ymm1, ymm0, ymm0
	vmulpd	ymm0, ymm1, ymm0
	vmulpd	ymm0, ymm0, YMMWORD PTR [rcx+rax]
	vmulpd	ymm0, ymm0, ymm11
	vfmadd231pd	ymm8, ymm2, ymm0
	vfmadd231pd	ymm7, ymm3, ymm0
	vmovapd	ymm2, YMMWORD PTR 32[rsi+rax]
	vmovapd	ymm3, YMMWORD PTR 32[rdi+rax]
	vsubpd	ymm2, ymm2, ymm10
	vsubpd	ymm3, ymm3, ymm9
	vmulpd	ymm2, ymm2, ymm6
	vmulpd	ymm3, ymm3, ymm6
	vmulpd	ymm0, ymm2, ymm2
	vfmadd231pd	ymm0, ymm3, ymm3
	vaddpd	ymm0, ymm12, ymm0
	vrsqrt14pd	ymm0, ymm0
	vmulpd	ymm1, ymm0, ymm0
	vmulpd	ymm0, ymm1, ymm0
	vmulpd	ymm0, ymm0, YMMWORD PTR 32[rcx+rax]
	vmulpd	ymm0, ymm0, ymm11
	vfmadd231pd	ymm4, ymm3, ymm0
	vfmadd231pd	ymm5, ymm2, ymm0
#APP
# 263 "main.cpp" 1
	# === END INNER LOOP ===
# 0 "" 2
#NO_APP
	add	rax, 64
	cmp	rax, r9
	jne	.L27
.L29:
	vmovapd	xmm0, xmm7
	vextractf64x2	xmm7, ymm7, 0x1
	vaddpd	xmm0, xmm0, xmm7
	vunpckhpd	xmm1, xmm0, xmm0
	vaddsd	xmm1, xmm0, xmm1
	vmovapd	xmm0, xmm8
	vextractf64x2	xmm8, ymm8, 0x1
	vaddpd	xmm0, xmm0, xmm8
	vunpckhpd	xmm2, xmm0, xmm0
	vaddsd	xmm0, xmm0, xmm2
	vmovapd	xmm2, xmm4
	vextractf64x2	xmm4, ymm4, 0x1
	vaddpd	xmm4, xmm4, xmm2
	vunpckhpd	xmm2, xmm4, xmm4
	vaddsd	xmm4, xmm4, xmm2
	vmovapd	xmm2, xmm5
	vextractf64x2	xmm5, ymm5, 0x1
	vaddsd	xmm4, xmm4, xmm1
	vaddpd	xmm5, xmm5, xmm2
	vunpckhpd	xmm2, xmm5, xmm5
	vaddsd	xmm5, xmm5, xmm2
	vaddsd	xmm5, xmm5, xmm0
#APP
# 276 "main.cpp" 1
	# === END CRITICAL LOOP ===
# 0 "" 2
#NO_APP
	vmovsd	xmm0, xmm4, xmm4
	vmovsd	xmm1, xmm5, xmm5
	vfmadd213sd	xmm0, xmm13, QWORD PTR [r10+r8*8]
	vfmadd213sd	xmm1, xmm13, QWORD PTR 8[r10+r8*8]
	vunpcklpd	xmm4, xmm4, xmm5
	vunpcklpd	xmm0, xmm0, xmm1
	vinsertf128	ymm0, ymm0, xmm4, 0x1
	vmovupd	YMMWORD PTR [r10+r8*8], ymm0
	add	r8, 8
	cmp	r11, r8
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
	.p2align 4
	.p2align 3
.L36:
	.cfi_restore_state
	vxorpd	xmm5, xmm5, xmm5
	vmovapd	ymm4, ymm5
	vmovapd	ymm8, ymm5
	vmovapd	ymm7, ymm5
	jmp	.L29
.L25:
	inc	eax
	xor	edx, edx
	jmp	.L30
	.cfi_endproc
.LFE15481:
	.size	_Z9calcForceP4Bodyfi._omp_fn.0, .-_Z9calcForceP4Bodyfi._omp_fn.0
	.p2align 4
	.type	main._omp_fn.0, @function
main._omp_fn.0:
.LFB15482:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	ecx, 1
	mov	edx, 1
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r15
	push	r14
	push	r13
	push	r12
	push	rbx
	and	rsp, -64
	sub	rsp, 64
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movsx	rsi, DWORD PTR 24[rdi]
	vmovsd	xmm8, QWORD PTR 8[rdi]
	mov	r15, QWORD PTR 16[rdi]
	lea	r9, 48[rsp]
	lea	r8, 40[rsp]
	xor	edi, edi
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 56[rsp], rax
	xor	eax, eax
	vmovsd	QWORD PTR 24[rsp], xmm8
	mov	QWORD PTR [rsp], r9
	mov	QWORD PTR 8[rsp], r8
	call	GOMP_loop_nonmonotonic_dynamic_start@PLT
	test	al, al
	je	.L38
	vmovdqa64	zmm4, ZMMWORD PTR .LC10[rip]
	vmovdqa64	zmm0, ZMMWORD PTR .LC11[rip]
	vxorps	xmm9, xmm9, xmm9
	vmovsd	xmm8, QWORD PTR 24[rsp]
	vmovdqa	ymm10, YMMWORD PTR .LC12[rip]
	vmovdqa	ymm5, YMMWORD PTR .LC13[rip]
	.p2align 4
	.p2align 3
.L46:
	mov	r9, QWORD PTR 48[rsp]
	mov	r10, QWORD PTR 40[rsp]
	mov	r14, QWORD PTR body_x[rip]
	mov	r11, QWORD PTR body_y[rip]
	mov	r12d, r9d
	mov	r8d, r10d
	movsx	rdx, r10d
	mov	DWORD PTR 20[rsp], r9d
	sub	r12d, r10d
	lea	edi, -1[r12]
	cmp	edi, 7
	jbe	.L39
	cmp	r10d, r9d
	jge	.L39
	lea	r13, 128[0+rdx*8]
	mov	rbx, r12
	mov	rax, rdx
	sal	rax, 6
	mov	QWORD PTR 24[rsp], r13
	sal	rbx, 6
	lea	r13, 0[0+rdx*8]
	lea	rbx, -48[rax+rbx]
	add	rax, r15
	lea	r12, 0[r13+r12*8]
	lea	rsi, [r11+r13]
	lea	rcx, [r14+r13]
	add	rbx, r15
	lea	r13, [r14+r12]
	cmp	rax, r13
	setnb	BYTE PTR 24[rsp]
	cmp	rcx, rbx
	setnb	r13b
	add	r12, r11
	or	r13b, BYTE PTR 24[rsp]
	cmp	rax, r12
	setnb	r12b
	cmp	rsi, rbx
	setnb	bl
	or	r12d, ebx
	test	r13b, r12b
	je	.L39
	lea	r13, 128[0+rdx*8]
	lea	rbx, [r11+r13]
	lea	r12, [r14+r13]
	cmp	rcx, rbx
	setnb	bl
	cmp	rsi, r12
	setnb	r12b
	or	r12b, bl
	je	.L39
	cmp	edi, 15
	jbe	.L48
	mov	r8d, edi
	vbroadcastsd	zmm12, xmm8
	shr	r8d, 4
	sal	r8, 7
	add	r8, rcx
	.p2align 4
	.p2align 3
.L41:
	vmovupd	zmm2, ZMMWORD PTR [rax]
	vmovupd	zmm1, ZMMWORD PTR 128[rax]
	vmovupd	zmm3, ZMMWORD PTR 384[rax]
	sub	rcx, -128
	vmovupd	zmm11, ZMMWORD PTR 384[rax]
	vmovupd	zmm13, ZMMWORD PTR 896[rax]
	sub	rsi, -128
	add	rax, 1024
	vpermt2pd	zmm1, zmm4, ZMMWORD PTR -832[rax]
	vpermt2pd	zmm2, zmm4, ZMMWORD PTR -960[rax]
	vpermt2pd	zmm3, zmm4, ZMMWORD PTR -576[rax]
	vpermt2pd	zmm2, zmm0, zmm1
	vmovupd	zmm1, ZMMWORD PTR -768[rax]
	vpermt2pd	zmm11, zmm0, ZMMWORD PTR -576[rax]
	vpermt2pd	zmm13, zmm0, ZMMWORD PTR -64[rax]
	vpermt2pd	zmm1, zmm4, ZMMWORD PTR -704[rax]
	vpermt2pd	zmm1, zmm0, zmm3
	vmovupd	zmm3, ZMMWORD PTR -128[rax]
	vpermt2pd	zmm2, zmm0, zmm1
	vmovupd	zmm1, ZMMWORD PTR -512[rax]
	vmovapd	zmm14, zmm2
	vmovupd	zmm2, ZMMWORD PTR -384[rax]
	vpermt2pd	zmm3, zmm4, ZMMWORD PTR -64[rax]
	vpermt2pd	zmm1, zmm4, ZMMWORD PTR -448[rax]
	vpermt2pd	zmm2, zmm4, ZMMWORD PTR -320[rax]
	vpermt2pd	zmm1, zmm0, zmm2
	vmovupd	zmm2, ZMMWORD PTR -256[rax]
	vpermt2pd	zmm2, zmm4, ZMMWORD PTR -192[rax]
	vpermt2pd	zmm2, zmm0, zmm3
	vmovapd	zmm3, zmm1
	vmovupd	zmm1, ZMMWORD PTR -896[rax]
	vpermt2pd	zmm3, zmm0, zmm2
	vmovupd	zmm2, ZMMWORD PTR -1024[rax]
	vpermt2pd	zmm1, zmm0, ZMMWORD PTR -832[rax]
	vpermt2pd	zmm2, zmm0, ZMMWORD PTR -960[rax]
	vpermt2pd	zmm2, zmm0, zmm1
	vmovupd	zmm1, ZMMWORD PTR -768[rax]
	vpermt2pd	zmm1, zmm0, ZMMWORD PTR -704[rax]
	vpermt2pd	zmm1, zmm0, zmm11
	vmovupd	zmm11, ZMMWORD PTR -384[rax]
	vpermt2pd	zmm2, zmm0, zmm1
	vmovupd	zmm1, ZMMWORD PTR -512[rax]
	vcvttpd2dq	ymm2, zmm2
	vcvtdq2pd	zmm2, ymm2
	vpermt2pd	zmm11, zmm0, ZMMWORD PTR -320[rax]
	vpermt2pd	zmm1, zmm0, ZMMWORD PTR -448[rax]
	vpermt2pd	zmm1, zmm0, zmm11
	vmovupd	zmm11, ZMMWORD PTR -256[rax]
	vpermt2pd	zmm11, zmm0, ZMMWORD PTR -192[rax]
	vfmadd213pd	zmm2, zmm12, ZMMWORD PTR -128[rcx]
	vmovupd	ZMMWORD PTR -128[rcx], zmm2
	vcvttpd2dq	ymm2, zmm14
	vpermt2pd	zmm11, zmm0, zmm13
	vcvtdq2pd	zmm2, ymm2
	vpermt2pd	zmm1, zmm0, zmm11
	vcvttpd2dq	ymm1, zmm1
	vcvtdq2pd	zmm1, ymm1
	vfmadd213pd	zmm1, zmm12, ZMMWORD PTR -64[rcx]
	vmovupd	ZMMWORD PTR -64[rcx], zmm1
	vcvttpd2dq	ymm1, zmm3
	vfmadd213pd	zmm2, zmm12, ZMMWORD PTR -128[rsi]
	vmovupd	ZMMWORD PTR -128[rsi], zmm2
	vcvtdq2pd	zmm1, ymm1
	vfmadd213pd	zmm1, zmm12, ZMMWORD PTR -64[rsi]
	vmovupd	ZMMWORD PTR -64[rsi], zmm1
	cmp	r8, rcx
	jne	.L41
	mov	eax, edi
	and	eax, -16
	sub	edi, eax
	lea	r8d, [rax+r10]
	cmp	edi, 7
	jbe	.L42
.L40:
	add	rax, rdx
	and	edi, -8
	vbroadcastsd	ymm3, xmm8
	lea	rdx, 0[0+rax*8]
	add	r8d, edi
	sal	rax, 6
	add	rax, r15
	lea	rcx, [r14+rdx]
	add	rdx, r11
	vmovupd	ymm2, YMMWORD PTR [rax]
	vmovupd	ymm18, YMMWORD PTR 64[rax]
	vmovupd	ymm27, YMMWORD PTR 32[rax]
	vmovupd	ymm26, YMMWORD PTR 96[rax]
	vmovupd	ymm14, YMMWORD PTR 128[rax]
	vmovupd	ymm17, YMMWORD PTR 192[rax]
	vmovupd	ymm24, YMMWORD PTR 160[rax]
	vmovupd	ymm23, YMMWORD PTR 224[rax]
	vmovupd	ymm16, YMMWORD PTR 320[rax]
	vmovupd	ymm22, YMMWORD PTR 288[rax]
	vmovupd	ymm21, YMMWORD PTR 352[rax]
	vmovupd	ymm13, YMMWORD PTR 384[rax]
	vmovupd	ymm15, YMMWORD PTR 448[rax]
	vmovupd	ymm20, YMMWORD PTR 416[rax]
	vmovupd	ymm19, YMMWORD PTR 480[rax]
	vmovapd	ymm1, ymm2
	vmovapd	ymm11, ymm18
	vpermt2pd	ymm2, ymm5, ymm27
	vpermt2pd	ymm11, ymm10, ymm26
	vpermt2pd	ymm1, ymm10, ymm27
	vmovapd	ymm12, ymm14
	vpermt2pd	ymm18, ymm5, ymm26
	vpermt2pd	ymm1, ymm5, ymm11
	vmovapd	ymm11, ymm17
	vpermt2pd	ymm12, ymm10, ymm24
	vmovapd	ymm25, ymm16
	vpermt2pd	ymm11, ymm10, ymm23
	vpermt2pd	ymm25, ymm10, ymm21
	vpermt2pd	ymm14, ymm5, ymm24
	vpermt2pd	ymm17, ymm5, ymm23
	vpermt2pd	ymm12, ymm5, ymm11
	vmovapd	ymm28, ymm15
	vpermt2pd	ymm16, ymm5, ymm21
	vpermt2pd	ymm15, ymm5, ymm19
	vpermt2pd	ymm1, ymm5, ymm12
	vpermt2pd	ymm28, ymm10, ymm19
	vpermt2pd	ymm2, ymm5, ymm18
	vpermt2pd	ymm14, ymm5, ymm17
	vmovapd	ymm11, ymm1
	vmovupd	ymm1, YMMWORD PTR 256[rax]
	vpermt2pd	ymm2, ymm5, ymm14
	vcvttpd2dq	xmm2, ymm2
	vcvtdq2pd	ymm2, xmm2
	vfmadd213pd	ymm2, ymm3, YMMWORD PTR [rcx]
	vmovupd	YMMWORD PTR [rcx], ymm2
	vmovapd	ymm12, ymm1
	vpermt2pd	ymm1, ymm5, ymm22
	vpermt2pd	ymm12, ymm10, ymm22
	vpermt2pd	ymm1, ymm5, ymm16
	vpermt2pd	ymm12, ymm5, ymm25
	vmovapd	ymm25, ymm13
	vpermt2pd	ymm13, ymm5, ymm20
	vpermt2pd	ymm25, ymm10, ymm20
	vpermt2pd	ymm13, ymm5, ymm15
	vpermt2pd	ymm25, ymm5, ymm28
	vpermt2pd	ymm1, ymm5, ymm13
	vpermt2pd	ymm12, ymm5, ymm25
	vcvttpd2dq	xmm1, ymm1
	vcvtdq2pd	ymm1, xmm1
	vfmadd213pd	ymm1, ymm3, YMMWORD PTR 32[rcx]
	vcvttpd2dq	xmm12, ymm12
	vmovupd	YMMWORD PTR 32[rcx], ymm1
	vcvtdq2pd	ymm12, xmm12
	vcvttpd2dq	xmm1, ymm11
	vcvtdq2pd	ymm1, xmm1
	vfmadd213pd	ymm1, ymm3, YMMWORD PTR [rdx]
	vfmadd213pd	ymm3, ymm12, YMMWORD PTR 32[rdx]
	vmovupd	YMMWORD PTR [rdx], ymm1
	vmovupd	YMMWORD PTR 32[rdx], ymm3
.L42:
	movsx	rdx, r8d
	lea	rax, 0[0+rdx*8]
	sal	rdx, 6
	lea	rcx, [r14+rax]
	lea	rsi, [r15+rdx]
	vcvttsd2si	edi, QWORD PTR [rsi]
	vcvtsi2sd	xmm1, xmm9, edi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	rcx, [r11+rax]
	vcvttsd2si	esi, QWORD PTR 8[rsi]
	vcvtsi2sd	xmm1, xmm9, esi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	ecx, 1[r8]
	cmp	r9d, ecx
	jle	.L45
	lea	rcx, 8[r14+rax]
	lea	rsi, 64[r15+rdx]
	vcvttsd2si	edi, QWORD PTR [rsi]
	vcvtsi2sd	xmm1, xmm9, edi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	rcx, 8[r11+rax]
	vcvttsd2si	esi, QWORD PTR 8[rsi]
	vcvtsi2sd	xmm1, xmm9, esi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	ecx, 2[r8]
	cmp	r9d, ecx
	jle	.L45
	lea	rcx, 16[r14+rax]
	lea	rsi, 128[r15+rdx]
	vcvttsd2si	edi, QWORD PTR [rsi]
	vcvtsi2sd	xmm1, xmm9, edi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	rcx, 16[r11+rax]
	vcvttsd2si	esi, QWORD PTR 8[rsi]
	vcvtsi2sd	xmm1, xmm9, esi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	ecx, 3[r8]
	cmp	r9d, ecx
	jle	.L45
	lea	rcx, 24[r14+rax]
	lea	rsi, 192[r15+rdx]
	vcvttsd2si	edi, QWORD PTR [rsi]
	vcvtsi2sd	xmm1, xmm9, edi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	rcx, 24[r11+rax]
	vcvttsd2si	esi, QWORD PTR 8[rsi]
	vcvtsi2sd	xmm1, xmm9, esi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	ecx, 4[r8]
	cmp	r9d, ecx
	jle	.L45
	lea	rcx, 32[r14+rax]
	lea	rsi, 256[r15+rdx]
	vcvttsd2si	edi, QWORD PTR [rsi]
	vcvtsi2sd	xmm1, xmm9, edi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	rcx, 32[r11+rax]
	vcvttsd2si	esi, QWORD PTR 8[rsi]
	vcvtsi2sd	xmm1, xmm9, esi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	ecx, 5[r8]
	cmp	r9d, ecx
	jle	.L45
	lea	rcx, 40[r14+rax]
	lea	rsi, 320[r15+rdx]
	vcvttsd2si	edi, QWORD PTR [rsi]
	vcvtsi2sd	xmm1, xmm9, edi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	rcx, 40[r11+rax]
	vcvttsd2si	esi, QWORD PTR 8[rsi]
	vcvtsi2sd	xmm1, xmm9, esi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	ecx, 6[r8]
	cmp	r9d, ecx
	jle	.L45
	lea	rcx, 48[r14+rax]
	lea	rsi, 384[r15+rdx]
	add	r8d, 7
	vcvttsd2si	edi, QWORD PTR [rsi]
	vcvtsi2sd	xmm1, xmm9, edi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	lea	rcx, 48[r11+rax]
	vcvttsd2si	esi, QWORD PTR 8[rsi]
	vcvtsi2sd	xmm1, xmm9, esi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	cmp	r9d, r8d
	jle	.L45
	lea	rcx, 56[r14+rax]
	lea	rdx, 448[r15+rdx]
	lea	rax, 56[r11+rax]
	vcvttsd2si	esi, QWORD PTR [rdx]
	vcvtsi2sd	xmm1, xmm9, esi
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rcx]
	vmovsd	QWORD PTR [rcx], xmm1
	vcvttsd2si	edx, QWORD PTR 8[rdx]
	vcvtsi2sd	xmm1, xmm9, edx
	vfmadd213sd	xmm1, xmm8, QWORD PTR [rax]
	vmovsd	QWORD PTR [rax], xmm1
	.p2align 4
	.p2align 3
.L45:
	mov	rsi, QWORD PTR [rsp]
	mov	rdi, QWORD PTR 8[rsp]
	vmovsd	QWORD PTR 24[rsp], xmm8
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vmovdqa64	zmm4, ZMMWORD PTR .LC10[rip]
	vmovdqa64	zmm0, ZMMWORD PTR .LC11[rip]
	vmovsd	xmm8, QWORD PTR 24[rsp]
	test	al, al
	vxorps	xmm9, xmm9, xmm9
	vmovdqa	ymm10, YMMWORD PTR .LC12[rip]
	vmovdqa	ymm5, YMMWORD PTR .LC13[rip]
	jne	.L46
	vzeroupper
.L38:
	call	GOMP_loop_end_nowait@PLT
	mov	rax, QWORD PTR 56[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L69
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
.L39:
	.cfi_restore_state
	mov	esi, DWORD PTR 20[rsp]
	mov	rax, rdx
	sal	rax, 6
	add	rax, r15
	.p2align 4
	.p2align 3
.L44:
	vcvttsd2si	ecx, QWORD PTR [rax]
	vcvtsi2sd	xmm1, xmm9, ecx
	vfmadd213sd	xmm1, xmm8, QWORD PTR [r14+rdx*8]
	vmovsd	QWORD PTR [r14+rdx*8], xmm1
	vcvttsd2si	ecx, QWORD PTR 8[rax]
	vcvtsi2sd	xmm1, xmm9, ecx
	vfmadd213sd	xmm1, xmm8, QWORD PTR [r11+rdx*8]
	vmovsd	QWORD PTR [r11+rdx*8], xmm1
	inc	rdx
	add	rax, 64
	cmp	esi, edx
	jg	.L44
	jmp	.L45
.L48:
	xor	eax, eax
	jmp	.L40
.L69:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE15482:
	.size	main._omp_fn.0, .-main._omp_fn.0
	.p2align 4
	.globl	_Z16randomlyAllocateP4Bodyi
	.type	_Z16randomlyAllocateP4Bodyi, @function
_Z16randomlyAllocateP4Bodyi:
.LFB14007:
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
	push	r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	push	rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	push	rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	mov	r12, rdi
	sub	rsp, 24
	.cfi_def_cfa_offset 80
	movsx	rbp, esi
	xor	edi, edi
	call	srand@PLT
	test	ebp, ebp
	jle	.L79
	xor	r13d, r13d
	mov	ebx, 1
	movabs	r14, 8589934597
	vxorps	xmm1, xmm1, xmm1
	sal	rbp, 3
	vxorpd	xmm3, xmm3, xmm3
	.p2align 4
	.p2align 3
.L74:
	imul	rbx, rbx, 16807
	mov	rsi, QWORD PTR body_x[rip]
	mov	rax, rbx
	mul	r14
	mov	rax, rbx
	add	rsi, r13
	sub	rax, rdx
	shr	rax
	add	rdx, rax
	shr	rdx, 30
	mov	rax, rdx
	sal	rax, 31
	sub	rax, rdx
	sub	rbx, rax
	imul	rcx, rbx, 16807
	lea	rax, -1[rbx]
	vcvtusi2sd	xmm2, xmm1, rax
	vaddsd	xmm2, xmm2, xmm3
	mov	rax, rcx
	mul	r14
	mov	rax, rcx
	sub	rax, rdx
	shr	rax
	add	rdx, rax
	shr	rdx, 30
	mov	rax, rdx
	sal	rax, 31
	sub	rax, rdx
	sub	rcx, rax
	lea	rax, -1[rcx]
	mov	rdx, rcx
	vcvtusi2sd	xmm0, xmm1, rax
	vfmadd132sd	xmm0, xmm2, QWORD PTR .LC16[rip]
	vdivsd	xmm0, xmm0, QWORD PTR .LC17[rip]
	vcomisd	xmm0, QWORD PTR .LC1[rip]
	jnb	.L76
	vfmadd132sd	xmm0, xmm3, QWORD PTR .LC18[rip]
.L72:
	imul	rcx, rdx, 16807
	vmovsd	QWORD PTR [rsi], xmm0
	mov	rsi, QWORD PTR body_y[rip]
	mov	rax, rcx
	mul	r14
	mov	rax, rcx
	add	rsi, r13
	sub	rax, rdx
	shr	rax
	add	rdx, rax
	shr	rdx, 30
	mov	rax, rdx
	sal	rax, 31
	sub	rax, rdx
	sub	rcx, rax
	lea	rax, -1[rcx]
	imul	rcx, rcx, 16807
	vcvtusi2sd	xmm2, xmm1, rax
	vaddsd	xmm2, xmm2, xmm3
	mov	rax, rcx
	mul	r14
	mov	rax, rcx
	sub	rax, rdx
	shr	rax
	lea	rbx, [rdx+rax]
	shr	rbx, 30
	mov	rax, rbx
	sal	rax, 31
	sub	rax, rbx
	mov	rbx, rcx
	sub	rbx, rax
	lea	rax, -1[rbx]
	vcvtusi2sd	xmm0, xmm1, rax
	vfmadd132sd	xmm0, xmm2, QWORD PTR .LC16[rip]
	vdivsd	xmm0, xmm0, QWORD PTR .LC17[rip]
	vcomisd	xmm0, QWORD PTR .LC1[rip]
	jnb	.L77
	vfmadd132sd	xmm0, xmm3, QWORD PTR .LC19[rip]
.L73:
	vmovsd	QWORD PTR [rsi], xmm0
	call	rand@PLT
	vxorps	xmm1, xmm1, xmm1
	vmovsd	xmm4, QWORD PTR .LC1[rip]
	cdq
	shr	edx, 30
	add	eax, edx
	and	eax, 3
	sub	eax, edx
	inc	eax
	vcvtsi2sd	xmm0, xmm1, eax
	mov	rax, QWORD PTR masses[rip]
	vmovsd	QWORD PTR [rax+r13], xmm0
	mov	rax, QWORD PTR inv_masses[rip]
	vdivsd	xmm0, xmm4, xmm0
	vmovsd	QWORD PTR [rax+r13], xmm0
	call	rand@PLT
	vxorps	xmm1, xmm1, xmm1
	movsx	rdx, eax
	mov	ecx, eax
	imul	rdx, rdx, -2139062143
	sar	ecx, 31
	shr	rdx, 32
	add	edx, eax
	sar	edx, 7
	sub	edx, ecx
	mov	ecx, edx
	sal	ecx, 8
	sub	ecx, edx
	sub	eax, ecx
	vcvtsi2sd	xmm0, xmm1, eax
	vmovsd	QWORD PTR 8[rsp], xmm0
	call	rand@PLT
	mov	r15d, eax
	call	rand@PLT
	vmovsd	xmm5, QWORD PTR 8[rsp]
	mov	QWORD PTR 56[r12+r13*8], 0x000000000
	mov	edx, eax
	cdqe
	vmovsd	QWORD PTR 48[r12+r13*8], xmm5
	vxorps	xmm1, xmm1, xmm1
	imul	rax, rax, -2139062143
	mov	ecx, edx
	vxorpd	xmm3, xmm3, xmm3
	sar	ecx, 31
	shr	rax, 32
	add	eax, edx
	sar	eax, 7
	sub	eax, ecx
	mov	ecx, eax
	sal	ecx, 8
	sub	ecx, eax
	movsx	rax, r15d
	imul	rax, rax, -2139062143
	sub	edx, ecx
	vmovd	xmm0, edx
	mov	edx, r15d
	shr	rax, 32
	add	eax, r15d
	sar	eax, 7
	sar	edx, 31
	sub	eax, edx
	mov	edx, eax
	sal	edx, 8
	sub	edx, eax
	sub	r15d, edx
	vpinsrd	xmm0, xmm0, r15d, 1
	vcvtdq2pd	xmm0, xmm0
	vmovupd	XMMWORD PTR 32[r12+r13*8], xmm0
	add	r13, 8
	cmp	r13, rbp
	jne	.L74
.L79:
	add	rsp, 24
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
.L77:
	.cfi_restore_state
	vmovsd	xmm0, QWORD PTR .LC15[rip]
	jmp	.L73
	.p2align 4
	.p2align 3
.L76:
	vmovsd	xmm0, QWORD PTR .LC14[rip]
	jmp	.L72
	.cfi_endproc
.LFE14007:
	.size	_Z16randomlyAllocateP4Bodyi, .-_Z16randomlyAllocateP4Bodyi
	.p2align 4
	.globl	_Z10calcForce2P4Bodyfi
	.type	_Z10calcForce2P4Bodyfi, @function
_Z10calcForce2P4Bodyfi:
.LFB14008:
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
	jne	.L85
	add	rsp, 40
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L85:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE14008:
	.size	_Z10calcForce2P4Bodyfi, .-_Z10calcForce2P4Bodyfi
	.p2align 4
	.globl	_Z20horizontal_sum_m256dDv4_d
	.type	_Z20horizontal_sum_m256dDv4_d, @function
_Z20horizontal_sum_m256dDv4_d:
.LFB14009:
	.cfi_startproc
	endbr64
	vmovapd	xmm1, xmm0
	vextractf64x2	xmm0, ymm0, 0x1
	vaddpd	xmm0, xmm1, xmm0
	vunpckhpd	xmm1, xmm0, xmm0
	vaddsd	xmm0, xmm0, xmm1
	ret
	.cfi_endproc
.LFE14009:
	.size	_Z20horizontal_sum_m256dDv4_d, .-_Z20horizontal_sum_m256dDv4_d
	.p2align 4
	.globl	_Z9calcForceP4Bodyfi
	.type	_Z9calcForceP4Bodyfi, @function
_Z9calcForceP4Bodyfi:
.LFB14010:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	and	rsp, -32
	add	rsp, -128
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 120[rsp], rax
	xor	eax, eax
#APP
# 164 "main.cpp" 1
	# === BEGIN CRITICAL LOOP ===
# 0 "" 2
#NO_APP
	vmovss	DWORD PTR 72[rsp], xmm0
	mov	DWORD PTR 76[rsp], esi
	mov	QWORD PTR 64[rsp], rdi
	vbroadcastsd	ymm1, QWORD PTR .LC21[rip]
	xor	ecx, ecx
	mov	rsi, rsp
	mov	edx, 2
	lea	rdi, _Z9calcForceP4Bodyfi._omp_fn.0[rip]
	vmovapd	YMMWORD PTR 32[rsp], ymm1
	vbroadcastsd	ymm1, QWORD PTR .LC5[rip]
	vmovapd	YMMWORD PTR [rsp], ymm1
	vzeroupper
	call	GOMP_parallel@PLT
	mov	rax, QWORD PTR 120[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L91
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L91:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE14010:
	.size	_Z9calcForceP4Bodyfi, .-_Z9calcForceP4Bodyfi
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC22:
	.string	"Done with : %d\n"
.LC27:
	.string	"Epoch_"
.LC28:
	.string	".png"
.LC29:
	.string	"basic_string::append"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB32:
	.section	.text.startup,"ax",@progbits
.LHOTB32:
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB14011:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA14011
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	ecx, 16
	mov	edx, 1200
	mov	esi, 800
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r15
	push	r14
	push	r13
	push	r12
	push	rbx
	and	rsp, -32
	sub	rsp, 576
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	lea	r15, 256[rsp]
	mov	rdi, r15
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 568[rsp], rax
	xor	eax, eax
.LEHB0:
	call	_ZN2cv3MatC1Eiii@PLT
.LEHE0:
	mov	esi, 49536
	mov	edi, 64
	call	aligned_alloc@PLT
	mov	esi, 49536
	mov	edi, 64
	mov	QWORD PTR body_x[rip], rax
	call	aligned_alloc@PLT
	mov	esi, 49536
	mov	edi, 64
	mov	QWORD PTR body_y[rip], rax
	call	aligned_alloc@PLT
	mov	esi, 49536
	mov	edi, 64
	mov	QWORD PTR masses[rip], rax
	call	aligned_alloc@PLT
	mov	edi, 396288
	mov	QWORD PTR inv_masses[rip], rax
	call	malloc@PLT
	mov	esi, 6192
	mov	rdi, rax
	mov	r13, rax
	call	_Z16randomlyAllocateP4Bodyi
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	lea	rax, 352[rsp]
	mov	DWORD PTR 60[rsp], 0
	mov	DWORD PTR 48[rsp], 0
	mov	QWORD PTR 40[rsp], rax
	.p2align 4
	.p2align 3
.L134:
	imul	eax, DWORD PTR 60[rsp], -858993459
	rorx	eax, eax, 1
	cmp	eax, 429496729
	ja	.L96
.L93:
	mov	edx, DWORD PTR 60[rsp]
	lea	rsi, .LC22[rip]
	mov	edi, 2
	xor	eax, eax
.LEHB1:
	call	__printf_chk@PLT
.L96:
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	cmp	DWORD PTR 48[rsp], 1
	jle	.L94
.L95:
	lea	r12, 64[rsp]
	movabs	r14, 26595454867210
.L140:
	mov	ebx, 5
	.p2align 4
	.p2align 3
.L97:
#APP
# 164 "main.cpp" 1
	# === BEGIN CRITICAL LOOP ===
# 0 "" 2
#NO_APP
	vmovapd	ymm1, YMMWORD PTR .LC20[rip]
	vmovapd	ymm2, YMMWORD PTR .LC8[rip]
	xor	ecx, ecx
	mov	edx, 2
	mov	rsi, r12
	lea	rdi, _Z9calcForceP4Bodyfi._omp_fn.0[rip]
	mov	QWORD PTR 136[rsp], r14
	mov	QWORD PTR 128[rsp], r13
	vmovapd	YMMWORD PTR 96[rsp], ymm1
	vmovapd	YMMWORD PTR 64[rsp], ymm2
	vzeroupper
	call	GOMP_parallel@PLT
	mov	rax, QWORD PTR .LC23[rip]
	xor	ecx, ecx
	xor	edx, edx
	mov	rsi, r12
	lea	rdi, main._omp_fn.0[rip]
	mov	QWORD PTR 80[rsp], r13
	mov	DWORD PTR 88[rsp], 6192
	mov	QWORD PTR 64[rsp], r15
	mov	QWORD PTR 72[rsp], rax
	call	GOMP_parallel@PLT
	dec	ebx
	jne	.L97
	mov	edi, 30
	call	_ZN2cv7waitKeyEi@PLT
	cmp	al, 113
	je	.L98
	mov	edi, DWORD PTR 48[rsp]
	test	edi, edi
	jne	.L101
	cmp	al, 112
	je	.L99
.L103:
	mov	DWORD PTR 48[rsp], 0
.L100:
	inc	DWORD PTR 60[rsp]
	mov	eax, DWORD PTR 60[rsp]
	add	QWORD PTR 40[rsp], 2
	cmp	eax, 100
	jne	.L134
	mov	rdi, QWORD PTR body_x[rip]
	call	free@PLT
	mov	rdi, QWORD PTR body_y[rip]
	call	free@PLT
	mov	rdi, QWORD PTR masses[rip]
	call	free@PLT
	mov	rdi, QWORD PTR inv_masses[rip]
	call	free@PLT
	mov	rdi, r15
	call	_ZN2cv3MatD1Ev@PLT
	jmp	.L135
.L121:
	inc	DWORD PTR 60[rsp]
	mov	eax, DWORD PTR 60[rsp]
	mov	ebx, 10
	add	QWORD PTR 40[rsp], 2
	cdq
	idiv	ebx
	mov	DWORD PTR 48[rsp], edx
	test	edx, edx
	je	.L93
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	xor	eax, eax
	mov	DWORD PTR 48[rsp], eax
	.p2align 4
	.p2align 3
.L94:
	lea	rbx, 224[rsp]
	vxorpd	xmm0, xmm0, xmm0
	mov	rdi, r15
	vmovapd	YMMWORD PTR 224[rsp], ymm0
	mov	rsi, rbx
	vzeroupper
	call	_ZN2cv3MataSERKNS_7Scalar_IdEE@PLT
	jmp	.L95
	.p2align 4
	.p2align 3
.L99:
	mov	esi, DWORD PTR 48[rsp]
	test	esi, esi
	je	.L120
.L101:
	dec	DWORD PTR 48[rsp]
	jne	.L100
.L120:
	cmp	DWORD PTR 60[rsp], 9
	jle	.L159
	.p2align 4
	.p2align 3
.L119:
	xor	edi, edi
	call	_ZN2cv7waitKeyEi@PLT
.LEHE1:
	cmp	al, 112
	je	.L103
	cmp	al, 113
	je	.L98
	cmp	al, 115
	je	.L104
	cmp	al, 110
	jne	.L119
	mov	DWORD PTR 48[rsp], 5
	jmp	.L100
	.p2align 4
	.p2align 3
.L104:
	lea	rbx, 192[rsp]
	lea	rax, 208[rsp]
	mov	edx, 45
	mov	esi, 2
	mov	rdi, rbx
	mov	QWORD PTR 32[rsp], rax
	mov	QWORD PTR 192[rsp], rax
	mov	QWORD PTR 24[rsp], rbx
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@PLT
	vmovdqa64	zmm3, ZMMWORD PTR .LC24[rip]
	vmovdqa64	zmm4, ZMMWORD PTR .LC25[rip]
	mov	rax, QWORD PTR 40[rsp]
	mov	r8d, 6
	lea	rcx, .LC27[rip]
	xor	esi, esi
	vmovdqa64	zmm5, ZMMWORD PTR .LC26[rip]
	vmovdqa	xmm6, XMMWORD PTR .LC31[rip]
	mov	rdi, rbx
	vmovdqu64	ZMMWORD PTR 352[rsp], zmm3
	vmovdqu64	ZMMWORD PTR 416[rsp], zmm4
	vmovdqu64	ZMMWORD PTR 480[rsp], zmm5
	vmovdqu	XMMWORD PTR 537[rsp], xmm6
	movzx	edx, WORD PTR [rax]
	mov	rax, QWORD PTR 192[rsp]
	mov	WORD PTR [rax], dx
	xor	edx, edx
	vzeroupper
.LEHB2:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm@PLT
.LEHE2:
	lea	rcx, 240[rsp]
	lea	r8, 16[rax]
	mov	QWORD PTR 224[rsp], rcx
	mov	rdx, QWORD PTR [rax]
	mov	QWORD PTR 48[rsp], rcx
	mov	rcx, QWORD PTR 8[rax]
	cmp	r8, rdx
	je	.L160
	mov	QWORD PTR 224[rsp], rdx
	mov	rdx, QWORD PTR 16[rax]
	mov	QWORD PTR 240[rsp], rdx
.L118:
	mov	rdx, QWORD PTR 8[rax]
	mov	QWORD PTR 232[rsp], rdx
	mov	QWORD PTR 8[rax], 0
	mov	BYTE PTR 16[rax], 0
	mov	QWORD PTR [rax], r8
	movabs	rax, 4611686018427387903
	sub	rax, QWORD PTR 232[rsp]
	cmp	rax, 3
	jbe	.L108
	lea	rbx, 224[rsp]
	mov	edx, 4
	lea	rsi, .LC28[rip]
	mov	rdi, rbx
.LEHB3:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm@PLT
.LEHE3:
	lea	rbx, 176[rsp]
	lea	rdx, 16[rax]
	mov	rsi, QWORD PTR 8[rax]
	mov	QWORD PTR 160[rsp], rbx
	mov	rcx, QWORD PTR [rax]
	cmp	rcx, rdx
	je	.L161
	mov	QWORD PTR 160[rsp], rcx
	mov	rcx, QWORD PTR 16[rax]
	mov	QWORD PTR 176[rsp], rcx
.L117:
	mov	QWORD PTR [rax], rdx
	mov	BYTE PTR 16[rax], 0
	mov	QWORD PTR 8[rax], 0
	mov	rdi, QWORD PTR 224[rsp]
	mov	rax, QWORD PTR 48[rsp]
	mov	QWORD PTR 168[rsp], rsi
	cmp	rdi, rax
	je	.L116
	mov	rax, QWORD PTR 240[rsp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
.L116:
	mov	rdi, QWORD PTR 192[rsp]
	mov	rax, QWORD PTR 32[rsp]
	cmp	rdi, rax
	je	.L115
	mov	rax, QWORD PTR 208[rsp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
.L115:
	mov	rdi, QWORD PTR 160[rsp]
	cmp	rdi, rbx
	je	.L119
	mov	rax, QWORD PTR 176[rsp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
	jmp	.L119
	.p2align 4
	.p2align 3
.L98:
	mov	rdi, r15
	call	_ZN2cv3MatD1Ev@PLT
.L135:
	mov	rax, QWORD PTR 568[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L157
	lea	rsp, -40[rbp]
	xor	eax, eax
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L161:
	.cfi_restore_state
	lea	ecx, 1[rsi]
	mov	rdi, rbx
	mov	rsi, rdx
	rep movsb
	mov	rsi, QWORD PTR 8[rax]
	jmp	.L117
.L160:
	mov	rdi, QWORD PTR 48[rsp]
	inc	ecx
	mov	rsi, r8
	rep movsb
	jmp	.L118
.L159:
	lea	rax, 192[rsp]
	mov	QWORD PTR 16[rsp], rax
.L102:
	xor	edi, edi
.LEHB4:
	call	_ZN2cv7waitKeyEi@PLT
.LEHE4:
	cmp	al, 112
	je	.L121
	cmp	al, 113
	je	.L98
	cmp	al, 115
	je	.L162
	cmp	al, 110
	jne	.L102
	inc	DWORD PTR 60[rsp]
	mov	eax, DWORD PTR 60[rsp]
	mov	ecx, 10
	add	QWORD PTR 40[rsp], 2
	mov	DWORD PTR 48[rsp], 5
	cdq
	idiv	ecx
	test	edx, edx
	je	.L93
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	jmp	.L140
.L162:
	mov	rax, QWORD PTR 16[rsp]
	lea	rbx, 208[rsp]
	mov	edx, 45
	mov	esi, 1
	mov	QWORD PTR 32[rsp], rbx
	mov	QWORD PTR 192[rsp], rbx
	mov	rdi, rax
	mov	rbx, rax
	mov	QWORD PTR 24[rsp], rax
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@PLT
	movzx	eax, BYTE PTR 60[rsp]
	mov	r8d, 6
	lea	rcx, .LC27[rip]
	xor	esi, esi
	mov	rdi, rbx
	lea	edx, 48[rax]
	mov	rax, QWORD PTR 192[rsp]
	mov	BYTE PTR [rax], dl
	xor	edx, edx
.LEHB5:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm@PLT
.LEHE5:
	lea	rbx, 240[rsp]
	lea	rdx, 16[rax]
	mov	rsi, QWORD PTR 8[rax]
	mov	QWORD PTR 224[rsp], rbx
	mov	rcx, QWORD PTR [rax]
	mov	QWORD PTR 48[rsp], rbx
	cmp	rcx, rdx
	je	.L163
	mov	QWORD PTR 224[rsp], rcx
	mov	rcx, QWORD PTR 16[rax]
	mov	QWORD PTR 240[rsp], rcx
.L124:
	mov	rcx, QWORD PTR 8[rax]
	mov	QWORD PTR 232[rsp], rcx
	xor	ecx, ecx
	mov	BYTE PTR 16[rax], 0
	mov	QWORD PTR [rax], rdx
	mov	QWORD PTR 8[rax], rcx
	movabs	rax, 4611686018427387903
	sub	rax, QWORD PTR 232[rsp]
	cmp	rax, 3
	jbe	.L108
	lea	rbx, 224[rsp]
	mov	edx, 4
	lea	rsi, .LC28[rip]
	mov	rdi, rbx
.LEHB6:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm@PLT
	lea	rbx, 176[rsp]
	lea	rdx, 16[rax]
	mov	rcx, QWORD PTR 8[rax]
	mov	QWORD PTR 160[rsp], rbx
	mov	rsi, QWORD PTR [rax]
	cmp	rsi, rdx
	je	.L164
	mov	QWORD PTR 160[rsp], rsi
	mov	rsi, QWORD PTR 16[rax]
	mov	QWORD PTR 176[rsp], rsi
.L128:
	mov	QWORD PTR [rax], rdx
	xor	edx, edx
	mov	BYTE PTR 16[rax], 0
	mov	rdi, QWORD PTR 224[rsp]
	mov	QWORD PTR 8[rax], rdx
	mov	rax, QWORD PTR 48[rsp]
	mov	QWORD PTR 168[rsp], rcx
	cmp	rdi, rax
	je	.L129
	mov	rax, QWORD PTR 240[rsp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
.L129:
	mov	rdi, QWORD PTR 192[rsp]
	mov	rax, QWORD PTR 32[rsp]
	cmp	rdi, rax
	je	.L130
	mov	rax, QWORD PTR 208[rsp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
.L130:
	mov	rdi, QWORD PTR 160[rsp]
	cmp	rdi, rbx
	je	.L102
	mov	rax, QWORD PTR 176[rsp]
	lea	rsi, 1[rax]
	call	_ZdlPvm@PLT
	jmp	.L102
.L164:
	inc	ecx
	mov	rdi, rbx
	mov	rsi, rdx
	rep movsb
	mov	rcx, QWORD PTR 8[rax]
	jmp	.L128
.L163:
	lea	ecx, 1[rsi]
	mov	rdi, rbx
	mov	rsi, rdx
	rep movsb
	jmp	.L124
.L108:
	mov	rax, QWORD PTR 568[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L157
	lea	rdi, .LC29[rip]
	lea	rbx, 224[rsp]
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE6:
.L157:
	call	__stack_chk_fail@PLT
.L144:
	endbr64
	mov	rbx, rax
	vzeroupper
	jmp	.L137
.L143:
	endbr64
	mov	rbx, rax
	vzeroupper
	jmp	.L138
.L145:
	endbr64
	mov	r12, rax
	jmp	.L136
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA14011:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE14011-.LLSDACSB14011
.LLSDACSB14011:
	.uleb128 .LEHB0-.LFB14011
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB14011
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L143-.LFB14011
	.uleb128 0
	.uleb128 .LEHB2-.LFB14011
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L144-.LFB14011
	.uleb128 0
	.uleb128 .LEHB3-.LFB14011
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L145-.LFB14011
	.uleb128 0
	.uleb128 .LEHB4-.LFB14011
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L143-.LFB14011
	.uleb128 0
	.uleb128 .LEHB5-.LFB14011
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L144-.LFB14011
	.uleb128 0
	.uleb128 .LEHB6-.LFB14011
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L145-.LFB14011
	.uleb128 0
.LLSDACSE14011:
	.section	.text.startup
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC14011
	.type	main.cold, @function
main.cold:
.LFSB14011:
.L136:
	.cfi_def_cfa 6, 16
	.cfi_offset 3, -56
	.cfi_offset 6, -16
	.cfi_offset 12, -48
	.cfi_offset 13, -40
	.cfi_offset 14, -32
	.cfi_offset 15, -24
	mov	rdi, rbx
	vzeroupper
	mov	rbx, r12
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@PLT
.L137:
	mov	rdi, QWORD PTR 24[rsp]
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@PLT
.L138:
	mov	rdi, r15
	call	_ZN2cv3MatD1Ev@PLT
	mov	rax, QWORD PTR 568[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L165
	mov	rdi, rbx
.LEHB7:
	call	_Unwind_Resume@PLT
.LEHE7:
.L165:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE14011:
	.section	.gcc_except_table
.LLSDAC14011:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC14011-.LLSDACSBC14011
.LLSDACSBC14011:
	.uleb128 .LEHB7-.LCOLDB32
	.uleb128 .LEHE7-.LEHB7
	.uleb128 0
	.uleb128 0
.LLSDACSEC14011:
	.section	.text.unlikely
	.section	.text.startup
	.size	main, .-main
	.section	.text.unlikely
	.size	main.cold, .-main.cold
.LCOLDE32:
	.section	.text.startup
.LHOTE32:
	.globl	inv_masses
	.bss
	.align 8
	.type	inv_masses, @object
	.size	inv_masses, 8
inv_masses:
	.zero	8
	.globl	masses
	.align 8
	.type	masses, @object
	.size	masses, 8
masses:
	.zero	8
	.globl	body_y
	.align 8
	.type	body_y, @object
	.size	body_y, 8
body_y:
	.zero	8
	.globl	body_x
	.align 8
	.type	body_x, @object
	.size	body_x, 8
body_x:
	.zero	8
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC1:
	.long	0
	.long	1072693248
	.align 8
.LC3:
	.long	0
	.long	1079574528
	.set	.LC5,.LC8
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC8:
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.section	.rodata
	.align 64
.LC10:
	.quad	1
	.quad	3
	.quad	5
	.quad	7
	.quad	9
	.quad	11
	.quad	13
	.quad	15
	.align 64
.LC11:
	.quad	0
	.quad	2
	.quad	4
	.quad	6
	.quad	8
	.quad	10
	.quad	12
	.quad	14
	.set	.LC12,.LC10
	.set	.LC13,.LC11
	.section	.rodata.cst8
	.align 8
.LC14:
	.long	-1
	.long	1083359231
	.align 8
.LC15:
	.long	-1
	.long	1082720255
	.align 8
.LC16:
	.long	-8388608
	.long	1105199103
	.align 8
.LC17:
	.long	-16777216
	.long	1137704959
	.align 8
.LC18:
	.long	0
	.long	1083359232
	.align 8
.LC19:
	.long	0
	.long	1082720256
	.section	.rodata.cst32
	.align 32
.LC20:
	.long	1202590843
	.long	1065646817
	.long	1202590843
	.long	1065646817
	.long	1202590843
	.long	1065646817
	.long	1202590843
	.long	1065646817
	.set	.LC21,.LC20
	.section	.rodata.cst8
	.align 8
.LC23:
	.long	1202590843
	.long	1066695393
	.section	.rodata
	.align 64
.LC24:
	.quad	3688503277381496880
	.quad	3976738051646829616
	.quad	3544667369688283184
	.quad	3832902143785906737
	.quad	4121136918051239473
	.quad	3689066235924983858
	.quad	3977301010190316594
	.quad	3545230328231770162
	.align 64
.LC25:
	.quad	3833465102329393715
	.quad	4121699876594726451
	.quad	3689629194468470836
	.quad	3977863968733803572
	.quad	3545793286775257140
	.quad	3834028060872880693
	.quad	4122262835138213429
	.quad	3690192153011957814
	.align 64
.LC26:
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
.LC31:
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
