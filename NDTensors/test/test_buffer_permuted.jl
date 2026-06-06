# Tests for to_buffer_permuted (NDTensors) and lanczos_permute (ITensors)
# Covers: heap→buffer copy with permutation, one-pass data movement

using Test: @test, @testset
using NDTensors: NDTensors, Bumper, BlockSparse, BlockSparseTensor, to_buffer,
    blockoffsets, nnz, data, storage, inds, tensor, blockview, with_alloc_buffer,
    eachnzblock, dense
using LinearAlgebra: norm
using Random: randn!
using ITensors
import ITensors: lanczos_permute

# Single-thread BLAS for nested threading
if Threads.nthreads() > 1
    LinearAlgebra.BLAS.set_num_threads(1)
    NDTensors.Strided.disable_threads()
end

@testset "to_buffer_permuted BlockSparse" begin
    # 2D block-sparse tensor, 2 blocks
    locs = [(1, 2), (2, 1)]
    indsA = ([2, 3], [4, 5])
    A = BlockSparseTensor{Float64}(locs, indsA...)
    d = data(storage(A)); randn!(d)

    # Reference: standard permutedims on heap
    A_ref = NDTensors.permutedims(A, (2, 1))

    # Buffer-backed permuted copy (one pass)
    buf = Bumper.SlabBuffer()
    A_buf = with_alloc_buffer(buf) do
        to_buffer(A, (2, 1), buf)
    end

    @test data(storage(A_buf)) isa NDTensors.UnsafeArray
    @test blockoffsets(A_buf) == blockoffsets(A_ref)
    @test dense(copy(A_ref)) ≈ dense(copy(A_buf))
end

@testset "to_buffer_permuted BlockSparse identity perm" begin
    # Identity permutation should match non-permuted to_buffer
    locs = [(1, 2), (2, 1)]
    indsA = ([2, 3], [4, 5])
    A = BlockSparseTensor{Float64}(locs, indsA...)
    d = data(storage(A)); randn!(d)

    buf = Bumper.SlabBuffer()
    A_plain = with_alloc_buffer(buf) do
        to_buffer(A, buf)
    end
    buf2 = Bumper.SlabBuffer()
    A_perm = with_alloc_buffer(buf2) do
        to_buffer(A, (1, 2), buf2)
    end

    @test dense(copy(A_plain)) ≈ dense(copy(A_perm))
end

@testset "to_buffer_permuted BlockSparse ComplexF64" begin
    # Complex type
    locs = [(1, 2), (2, 1)]
    indsA = ([2, 3], [4, 5])
    A = BlockSparseTensor{ComplexF64}(locs, indsA...)
    d = data(storage(A)); for i in eachindex(d); d[i] = complex(randn(), randn()); end

    buf = Bumper.SlabBuffer()
    with_alloc_buffer(buf) do
        A_buf = to_buffer(A, (2, 1), buf)
        @test data(storage(A_buf)) isa NDTensors.UnsafeArray
        @test eltype(A_buf) == ComplexF64
        A_ref = NDTensors.permutedims(A, (2, 1))
        @test dense(copy(A_ref)) ≈ dense(copy(A_buf))
    end
end

@testset "to_buffer_permuted 3D nonswap" begin
    # 3D interleaved perm (non-trivial, non-swap)
    locs = [(1, 1, 1)]
    inds3 = ([2], [3], [4])
    A = BlockSparseTensor{Float64}(locs, inds3...)
    d = data(storage(A)); d[1:24] .= randn(24)

    A_ref = NDTensors.permutedims(A, (2, 1, 3))

    buf = Bumper.SlabBuffer()
    with_alloc_buffer(buf) do
        A_buf = to_buffer(A, (2, 1, 3), buf)
        @test data(storage(A_buf)) isa NDTensors.UnsafeArray
        @test dense(copy(A_ref)) ≈ dense(copy(A_buf))
    end
end

@testset "lanczos_permute correctness" begin
    # Build LH, R, ψ with matching QN structure
    spec = [QN(0) => 5, QN(1) => 5, QN(-1) => 5]
    i = Index(spec, "i")
    j = Index(spec, "j")
    k = Index(spec, "k")
    l = Index(spec, "l")
    s = Index([QN(1) => 2, QN(-1) => 2], "s")  # site index

    LH = random_itensor(QN(0), i, j, dag(s))  # left env
    ψ = random_itensor(QN(0), s, k)            # MPS tensor
    R = random_itensor(QN(0), dag(k), l)       # right env

    # Heap reference
    ref = LH * ψ * R

    # Buffer version with lanczos_permute
    buf = Bumper.SlabBuffer{2^20}()
    LH_b, R_b, ψ_b = lanczos_permute(LH, R, ψ, buf)

    # Contract using buffer-backed tensors (needs buffer for output allocation)
    buf_check = Bumper.SlabBuffer{2^20}()
    result = with_alloc_buffer(buf_check) do
        LH_b * ψ_b * R_b
    end

    @test isapprox(ref, copy(result); atol=1e-10)
end

@testset "lanczos_permute reuse across iterations" begin
    # Verify LH_b and R_b work with different ψ
    spec = [QN(0) => 5, QN(1) => 5, QN(-1) => 5]
    i = Index(spec, "i")
    j = Index(spec, "j")
    k = Index(spec, "k")
    l = Index(spec, "l")
    s = Index([QN(1) => 2, QN(-1) => 2], "s")

    LH = random_itensor(QN(0), i, j, dag(s))
    ψ1 = random_itensor(QN(0), s, k)
    ψ2 = random_itensor(QN(0), s, k)
    R = random_itensor(QN(0), dag(k), l)

    ref1 = LH * ψ1 * R
    ref2 = LH * ψ2 * R

    buf = Bumper.SlabBuffer{2^20}()
    LH_b, R_b, ψ1_b = lanczos_permute(LH, R, ψ1, buf)

    # First ψ
    buf_work = Bumper.SlabBuffer{2^20}()
    r1 = with_alloc_buffer(buf_work) do
        LH_b * ψ1_b * R_b
    end

    # Second ψ (separate lanczos_permute call)
    buf2 = Bumper.SlabBuffer{2^20}()
    LH_b2, R_b2, ψ2_b = lanczos_permute(LH, R, ψ2, buf2)
    r2 = with_alloc_buffer(buf_work) do
        LH_b2 * ψ2_b * R_b2
    end

    @test isapprox(ref1, copy(r1); atol=1e-10)
    @test isapprox(ref2, copy(r2); atol=1e-10)
end
