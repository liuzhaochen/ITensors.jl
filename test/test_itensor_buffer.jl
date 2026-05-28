@eval module $(gensym())
using ITensors
using NDTensors: NDTensors, Bumper, DenseTensor, data as ndata
using NDTensors: get_alloc_buffer, set_alloc_buffer!, with_alloc_buffer
using LinearAlgebra: norm, dot
using Test: @test, @test_throws, @testset

@testset "ITensor buffer constructors" begin
    @testset "ITensor(buf, i, j) — basic" begin
        buf = Bumper.SlabBuffer()
        i = Index(2, "i")
        j = Index(3, "j")

        A = ITensor(buf, i, j)
        @test ndims(A) == 2
        @test size(A) == (2, 3)
        @test eltype(NDTensors.storage(A)) == Float64
        # Storage should be UnsafeArray-backed
        @test ndata(NDTensors.storage(A)) isa Bumper.UnsafeArrays.UnsafeArray
    end

    @testset "ITensor(buf, ElT, i, j) — with eltype" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        j = Index(5, "j")

        A = ITensor(buf, ComplexF64, i, j)
        @test eltype(NDTensors.storage(A)) == ComplexF64
        @test ndims(A) == 2
        @test size(A) == (4, 5)

        B = ITensor(buf, Float32, i, j)
        @test eltype(NDTensors.storage(B)) == Float32
    end

    @testset "ITensor(buf, undef, i, j) — uninitialized" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")

        A = ITensor(buf, undef, i)
        @test ndims(A) == 1
        @test size(A) == (3,)

        B = ITensor(buf, ComplexF64, undef, i)
        @test eltype(NDTensors.storage(B)) == ComplexF64
    end

    @testset "ITensor(buf, inds...) — varargs" begin
        buf = Bumper.SlabBuffer()
        i = Index(2)
        j = Index(3)

        A = ITensor(buf, i, j)
        @test ndims(A) == 2

        B = ITensor(buf, Float64, i, j, j')
        @test ndims(B) == 3
    end

    @testset "ITensor(buf, (i, j)) — tuple of indices" begin
        buf = Bumper.SlabBuffer()
        is = (Index(2), Index(3))

        A = ITensor(buf, is)
        @test ndims(A) == 2
    end

    @testset "random_itensor(buf, i, j)" begin
        buf = Bumper.SlabBuffer()
        i = Index(5)
        j = Index(5)

        A = random_itensor(buf, i, j)
        @test ndims(A) == 2
        @test size(A) == (5, 5)
        @test norm(A) > 0
    end

    @testset "random_itensor(buf, ComplexF64, i, j)" begin
        buf = Bumper.SlabBuffer()
        i = Index(4)
        j = Index(4)

        A = random_itensor(buf, ComplexF64, i, j)
        @test eltype(NDTensors.storage(A)) == ComplexF64
        @test norm(A) > 0
    end

    @testset "itensor(buf, i, j) — alias" begin
        buf = Bumper.SlabBuffer()
        i = Index(2)
        j = Index(3)

        A = itensor(buf, i, j)
        @test ndims(A) == 2
        @test ndata(NDTensors.storage(A)) isa Bumper.UnsafeArrays.UnsafeArray
    end

    @testset "buffer borrows — valid until reset" begin
        buf = Bumper.SlabBuffer()
        i = Index(10)
        j = Index(10)

        A = ITensor(buf, i, j)
        A[i => 1, j => 1] = 3.14
        @test A[i => 1, j => 1] ≈ 3.14
        Bumper.reset_buffer!(buf)
    end

    @testset "copy buffer ITensor to heap" begin
        buf = Bumper.SlabBuffer()
        i = Index(3)
        j = Index(3)

        A = random_itensor(buf, i, j)
        A_copy = copy(A)
        @test norm(A_copy - A) < 1e-10
        @test ndata(NDTensors.storage(A_copy)) isa Vector
    end

    @testset "contract buffer-allocated ITensors" begin
        buf = Bumper.SlabBuffer()
        i = Index(2, "i")
        j = Index(3, "j")
        k = Index(4, "k")

        A = random_itensor(buf, i, j)
        B = random_itensor(buf, k, j)

        C = with_alloc_buffer(buf) do
            A * B
        end
        @test ndims(C) == 2
        @test size(C) == (2, 4)

        C_heap = copy(C)
        Bumper.reset_buffer!(buf)
        @test ndims(C_heap) == 2
    end

    @testset "getindex/setindex on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(2, "i")
        j = Index(3, "j")

        A = ITensor(buf, Float64, i, j)
        A[i => 1, j => 1] = 1.5
        A[i => 2, j => 3] = 2.5
        @test A[i => 1, j => 1] ≈ 1.5
        @test A[i => 2, j => 3] ≈ 2.5

        # Fill and read all elements
        for ii in 1:2, jj in 1:3
            A[i => ii, j => jj] = Float64(ii + 10 * jj)
        end
        for ii in 1:2, jj in 1:3
            @test A[i => ii, j => jj] ≈ Float64(ii + 10 * jj)
        end
    end

    @testset "add/subtract buffer ITensors" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        j = Index(4, "j")

        A = random_itensor(buf, i, j)
        B = random_itensor(buf, i, j)

        with_alloc_buffer(buf) do
            C = A + B
            D = A - B
            @test ndims(C) == 2
            @test ndims(D) == 2
            @test norm(C - A - B) < 1e-10
            @test norm(D - A + B) < 1e-10
        end
        Bumper.reset_buffer!(buf)
    end

    @testset "norm and scalar on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(5, "i")
        j = Index(5, "j")

        A = random_itensor(buf, i, j)
        @test norm(A) > 0
        @test isapprox(norm(A), norm(array(A)))
    end

    @testset "dag (adjoint) of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(3, "j")

        A = random_itensor(buf, i, j)
        Ad = dag(A)
        @test ndims(Ad) == 2
        @test norm(Ad) > 0
    end

    @testset "scalar multiplication of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")

        A = random_itensor(buf, i)
        B = 3.0 * A
        @test norm(B - 3.0 * A) < 1e-10

        C = A * 2.0
        @test norm(C - 2.0 * A) < 1e-10
    end

    @testset "SVD of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        j = Index(5, "j")
        k = Index(6, "k")

        A = random_itensor(buf, i, j, k)

        # SVD uses a combiner internally; wrap in with_alloc_buffer
        with_alloc_buffer(buf) do
            U, S, V, spec = svd(A, (i, j))
            @test ndims(U) == 3
            @test ndims(V) == 2
            @test size(S, 1) == min(4 * 5, 6)
        end
        Bumper.reset_buffer!(buf)
    end

    @testset "QR of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        j = Index(5, "j")
        k = Index(6, "k")

        A = random_itensor(buf, i, j, k)

        with_alloc_buffer(buf) do
            Q, R, q = qr(A, (i, j))
            @test ndims(Q) == 3
            @test ndims(R) == 2
        end
        Bumper.reset_buffer!(buf)
    end

    @testset "permute buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(4, "j")

        A = random_itensor(buf, i, j)

        Ap = permute(A, j, i)
        @test inds(Ap) == (j, i)
        @test A[i => 1, j => 1] ≈ Ap[j => 1, i => 1]
        @test A[i => 2, j => 3] ≈ Ap[j => 3, i => 2]
    end

    @testset "inner product of buffer ITensors" begin
        buf = Bumper.SlabBuffer()
        i = Index(5, "i")
        j = Index(5, "j")

        A = random_itensor(buf, i, j)
        B = random_itensor(buf, i, j)

        with_alloc_buffer(buf) do
            ip = dot(A, B)
            @test ip ≈ sum(array(A) .* conj(array(B)))
        end
    end

    @testset "convert buffer ITensor to Array" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(4, "j")

        A = random_itensor(buf, i, j)
        arr = Array(A, i, j)
        @test arr isa Matrix{Float64}
        @test size(arr) == (3, 4)
        @test arr ≈ array(A)
    end

    @testset "with_alloc_buffer context" begin
        buf = Bumper.SlabBuffer()
        i = Index(3)
        j = Index(3)

        @test get_alloc_buffer() === nothing

        with_alloc_buffer(buf) do
            @test get_alloc_buffer() === buf
            A = ITensor(buf, i, j)
            @test ndims(A) == 2
        end

        @test get_alloc_buffer() === nothing
    end

    @testset "ResizeBuffer works" begin
        buf = Bumper.ResizeBuffer()
        i = Index(10)
        j = Index(10)

        A = ITensor(buf, i, j)
        @test ndims(A) == 2

        A[i => 1, j => 1] = 1.0
        @test A[i => 1, j => 1] ≈ 1.0

        B = random_itensor(buf, i, j)
        @test ndims(B) == 2

        Bumper.reset_buffer!(buf)
    end

    @testset "buffer DenseTensor constructor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3)
        j = Index(4)

        DT = DenseTensor(buf, (dim(i), dim(j)))
        @test ndims(DT) == 2

        DT2 = NDTensors.DenseTensor(Float64, buf, (dim(i), dim(j)))
        @test ndims(DT2) == 2
    end

    # ── Cross-checked against test/base/test_itensor.jl ──

    @testset "fill! buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        A = ITensor(buf, i)
        fill!(A, 3.0)
        @test all(ndata(NDTensors.storage(A)) .== 3.0)
    end

    @testset "zero buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = random_itensor(buf, i)
        Z = zero(A)
        @test norm(Z) == 0
    end

    @testset "isreal/iszero on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = random_itensor(buf, i)
        @test isreal(A)
        @test !iszero(A)
        @test iszero(zero(A))
        Ac = random_itensor(buf, ComplexF64, i)
        @test !isreal(Ac)
    end

    @testset "real/imag of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = random_itensor(buf, ComplexF64, i)
        Ar = real(A)
        Ai = imag(A)
        @test norm(Ar + im * Ai - A) < 1e-10
    end

    @testset "sum/prod of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        A = random_itensor(buf, i)
        @test sum(A) ≈ sum(array(A))
        @test prod(A) ≈ prod(array(A))
    end

    @testset "Matrix/Vector of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(4, "j")
        A = random_itensor(buf, i, j)
        M = Matrix(A, i, j)
        @test M isa Matrix{Float64}
        @test size(M) == (3, 4)

        v = random_itensor(buf, i)
        V = Vector(v)
        @test V isa Vector{Float64}
        @test length(V) == 3
    end

    @testset "dot buffer × heap ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(5, "i")
        A_buf = random_itensor(buf, i)
        B_heap = random_itensor(i)
        ip = dot(A_buf, B_heap)
        @test ip ≈ sum(array(A_buf) .* array(B_heap))
    end

    @testset "complex/conj of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = random_itensor(buf, i)
        C = complex(A)
        @test imag(C) ≈ zero(A)
        Ac = conj(C)
        @test Ac ≈ C
    end

    @testset "axpy! on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        A = random_itensor(buf, i)
        B = random_itensor(buf, i)
        C = copy(A)
        axpy!(2.0, B, C)
        @test C ≈ A + 2.0 * B
    end

    @testset "scalar mul! on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = random_itensor(buf, i)
        rmul!(A, 2.0)
        @test norm(A) > 0
    end
end

nothing
end
