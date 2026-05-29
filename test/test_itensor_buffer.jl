@eval module $(gensym())
using ITensors
using ITensors: to_buffer, hassameinds
using NDTensors: NDTensors, Bumper, DenseTensor, data as ndata
using NDTensors: get_alloc_buffer, set_alloc_buffer!, with_alloc_buffer
using NDTensors: enable_threaded_blocksparse, disable_threaded_blocksparse
using LinearAlgebra: norm, dot
using Test: @test, @test_throws, @testset

@testset "ITensor buffer" begin
    @testset "to_buffer basic Dense ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(2, "i")
        j = Index(3, "j")

        A = to_buffer(random_itensor(i, j), buf)
        @test ndims(A) == 2
        @test size(A) == (2, 3)
        @test eltype(NDTensors.storage(A)) == Float64
        @test ndata(NDTensors.storage(A)) isa NDTensors.UnsafeArray
    end

    @testset "to_buffer with custom eltype" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        j = Index(5, "j")

        A = to_buffer(random_itensor(ComplexF64, i, j), buf)
        @test eltype(NDTensors.storage(A)) == ComplexF64
        @test ndims(A) == 2
        @test size(A) == (4, 5)
        @test ndata(NDTensors.storage(A)) isa NDTensors.UnsafeArray

        B = to_buffer(random_itensor(Float32, i, j), buf)
        @test eltype(NDTensors.storage(B)) == Float32
        @test ndata(NDTensors.storage(B)) isa NDTensors.UnsafeArray
    end

    @testset "to_buffer preserves data" begin
        buf = Bumper.SlabBuffer()
        i = Index(5)
        j = Index(5)

        A_heap = random_itensor(i, j)
        A = to_buffer(A_heap, buf)
        @test A ≈ A_heap
        @test ndata(NDTensors.storage(A)) isa NDTensors.UnsafeArray
    end

    @testset "buffer borrows — valid until reset" begin
        buf = Bumper.SlabBuffer()
        i = Index(10)
        j = Index(10)

        A = to_buffer(ITensor(zeros(Float64, dim(i), dim(j)), i, j), buf)
        A[i => 1, j => 1] = 3.14
        @test A[i => 1, j => 1] ≈ 3.14
        Bumper.reset_buffer!(buf)
    end

    @testset "copy buffer ITensor to heap" begin
        buf = Bumper.SlabBuffer()
        i = Index(3)
        j = Index(3)

        A = to_buffer(random_itensor(i, j), buf)
        A_copy = copy(A)
        with_alloc_buffer(buf) do
            @test norm(A_copy - A) < 1e-10
        end
        @test ndata(NDTensors.storage(A_copy)) isa Vector
    end

    @testset "contract buffer-allocated ITensors" begin
        buf = Bumper.SlabBuffer()
        i = Index(2, "i")
        j = Index(3, "j")
        k = Index(4, "k")

        A = to_buffer(random_itensor(i, j), buf)
        B = to_buffer(random_itensor(k, j), buf)

        C = with_alloc_buffer(buf) do
            A * B
        end
        @test ndata(NDTensors.storage(C)) isa NDTensors.UnsafeArray
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

        A = to_buffer(ITensor(zeros(Float64, dim(i), dim(j)), i, j), buf)
        A[i => 1, j => 1] = 1.5
        A[i => 2, j => 3] = 2.5
        @test A[i => 1, j => 1] ≈ 1.5
        @test A[i => 2, j => 3] ≈ 2.5

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

        A = to_buffer(random_itensor(i, j), buf)
        B = to_buffer(random_itensor(i, j), buf)

        with_alloc_buffer(buf) do
            C = A + B
            D = A - B
            @test ndata(NDTensors.storage(C)) isa NDTensors.UnsafeArray
            @test ndata(NDTensors.storage(D)) isa NDTensors.UnsafeArray
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

        A = to_buffer(random_itensor(i, j), buf)
        @test norm(A) > 0
        @test isapprox(norm(A), norm(array(A)))
    end

    @testset "dag (adjoint) of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(3, "j")

        A = to_buffer(random_itensor(i, j), buf)
        with_alloc_buffer(buf) do
            Ad = dag(A)
            @test ndims(Ad) == 2
            @test norm(Ad) > 0
        end
    end

    @testset "scalar multiplication of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")

        A = to_buffer(random_itensor(i), buf)
        with_alloc_buffer(buf) do
            B = 3.0 * A
            @test ndata(NDTensors.storage(B)) isa NDTensors.UnsafeArray
            @test norm(B - 3.0 * A) < 1e-10

            C = A * 2.0
            @test ndata(NDTensors.storage(C)) isa NDTensors.UnsafeArray
            @test norm(C - 2.0 * A) < 1e-10
        end
    end

    @testset "SVD of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        j = Index(5, "j")
        k = Index(6, "k")

        A = to_buffer(random_itensor(i, j, k), buf)
        with_alloc_buffer(buf) do
            U, S, V, spec = svd(A, (i, j))
            @test ndata(NDTensors.storage(U)) isa NDTensors.UnsafeArray
            @test ndata(NDTensors.storage(V)) isa NDTensors.UnsafeArray
            @test ndata(NDTensors.storage(S)) isa NDTensors.UnsafeArray
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

        A = to_buffer(random_itensor(i, j, k), buf)
        with_alloc_buffer(buf) do
            Q, R, q = qr(A, (i, j))
            @test ndata(NDTensors.storage(Q)) isa NDTensors.UnsafeArray
            @test ndata(NDTensors.storage(R)) isa NDTensors.UnsafeArray
            @test ndims(Q) == 3
            @test ndims(R) == 2
        end
        Bumper.reset_buffer!(buf)
    end

    @testset "permute buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(4, "j")

        A = to_buffer(random_itensor(i, j), buf)
        with_alloc_buffer(buf) do
            Ap = permute(A, j, i)
            @test inds(Ap) == (j, i)
            @test A[i => 1, j => 1] ≈ Ap[j => 1, i => 1]
            @test A[i => 2, j => 3] ≈ Ap[j => 3, i => 2]
        end
    end

    @testset "inner product of buffer ITensors" begin
        buf = Bumper.SlabBuffer()
        i = Index(5, "i")
        j = Index(5, "j")

        A = to_buffer(random_itensor(i, j), buf)
        B = to_buffer(random_itensor(i, j), buf)

        with_alloc_buffer(buf) do
            ip = dot(A, B)
            @test ip ≈ sum(array(A) .* conj(array(B)))
        end
    end

    @testset "convert buffer ITensor to Array" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(4, "j")

        A = to_buffer(random_itensor(i, j), buf)
        with_alloc_buffer(buf) do
            arr = Array(A, i, j)
            @test arr isa Matrix{Float64}
            @test size(arr) == (3, 4)
            @test arr ≈ array(A)
        end
    end

    @testset "to_buffer outside with_alloc_buffer still works" begin
        buf = Bumper.SlabBuffer()
        i = Index(3)
        j = Index(3)

        @test get_alloc_buffer() === nothing
        A = to_buffer(random_itensor(i, j), buf)
        @test ndims(A) == 2
        @test ndata(NDTensors.storage(A)) isa NDTensors.UnsafeArray
        @test get_alloc_buffer() === nothing
    end

    @testset "ResizeBuffer works with to_buffer" begin
        buf = Bumper.ResizeBuffer()
        i = Index(10)
        j = Index(10)

        A = to_buffer(ITensor(zeros(Float64, dim(i), dim(j)), i, j), buf)
        @test ndims(A) == 2
        A[i => 1, j => 1] = 1.0
        @test A[i => 1, j => 1] ≈ 1.0

        B = to_buffer(random_itensor(i, j), buf)
        @test ndims(B) == 2
        @test ndata(NDTensors.storage(B)) isa NDTensors.UnsafeArray
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
        A = to_buffer(ITensor(zeros(Float64, dim(i)), i), buf)
        fill!(A, 3.0)
        @test all(ndata(NDTensors.storage(A)) .== 3.0)
    end

    @testset "zero buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = to_buffer(random_itensor(i), buf)
        with_alloc_buffer(buf) do
            Z = zero(A)
            @test ndata(NDTensors.storage(Z)) isa NDTensors.UnsafeArray
            @test norm(Z) == 0
        end
    end

    @testset "isreal/iszero on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = to_buffer(random_itensor(i), buf)
        @test isreal(A)
        @test !iszero(A)
        Z = with_alloc_buffer(buf) do
            zero(A)
        end
        @test ndata(NDTensors.storage(Z)) isa NDTensors.UnsafeArray
        @test iszero(Z)
        Ac = to_buffer(random_itensor(ComplexF64, i), buf)
        @test !isreal(Ac)
    end

    @testset "real/imag of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = to_buffer(random_itensor(ComplexF64, i), buf)
        with_alloc_buffer(buf) do
            Ar = real(A)
            Ai = imag(A)
            @test ndata(NDTensors.storage(Ar)) isa NDTensors.UnsafeArray
            @test ndata(NDTensors.storage(Ai)) isa NDTensors.UnsafeArray
            @test eltype(Ar) == Float64
            @test eltype(Ai) == Float64
            # Compare element-by-element to avoid norm dispatch issue
            for n in 1:dim(i)
                @test Ar[i => n] ≈ real(A[i => n])
                @test Ai[i => n] ≈ imag(A[i => n])
            end
        end
    end

    @testset "sum/prod of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        A = to_buffer(random_itensor(i), buf)
        @test sum(A) ≈ sum(array(A))
        @test prod(A) ≈ prod(array(A))
    end

    @testset "Matrix/Vector of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(4, "j")
        A = to_buffer(random_itensor(i, j), buf)
        with_alloc_buffer(buf) do
            M = Matrix(A, i, j)
            @test M isa Matrix{Float64}
            @test size(M) == (3, 4)
        end

        v = to_buffer(random_itensor(i), buf)
        with_alloc_buffer(buf) do
            V = Vector(v)
            @test V isa Vector{Float64}
            @test length(V) == 3
        end
    end

    @testset "dot buffer × heap ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(5, "i")
        A_buf = to_buffer(random_itensor(i), buf)
        B_heap = random_itensor(i)
        with_alloc_buffer(buf) do
            ip = dot(A_buf, B_heap)
            @test ip ≈ sum(array(A_buf) .* array(B_heap))
        end
    end

    @testset "complex/conj of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = to_buffer(random_itensor(i), buf)
        with_alloc_buffer(buf) do
            C = complex(A)
            @test ndata(NDTensors.storage(C)) isa NDTensors.UnsafeArray
            @test imag(C) ≈ zero(A)
            Ac = conj(C)
            @test Ac ≈ C
        end
    end

    @testset "axpy! on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        A = to_buffer(random_itensor(i), buf)
        B = to_buffer(random_itensor(i), buf)
        C = copy(A)
        with_alloc_buffer(buf) do
            axpy!(2.0, B, C)
            @test C ≈ A + 2.0 * B
        end
    end

    @testset "scalar mul! on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = to_buffer(random_itensor(i), buf)
        rmul!(A, 2.0)
        @test norm(A) > 0
    end

    # ── Division ──

    @testset "Division of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4)
        A = to_buffer(random_itensor(i), buf)
        with_alloc_buffer(buf) do
            B = A / 2
            @test B isa ITensor
            @test ndata(NDTensors.storage(B)) isa NDTensors.UnsafeArray
            @test norm(2 * B - A) < 1e-10

            C = A / ITensor(2)
            @test C isa ITensor
            @test ndata(NDTensors.storage(C)) isa NDTensors.UnsafeArray
            @test norm(2 * C - A) < 1e-10
        end
    end

    # ── Trace ──

    @testset "trace of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(4, "j")
        k = Index(5, "k")

        T = to_buffer(random_itensor(j, k', i', k, j', i), buf)
        with_alloc_buffer(buf) do
            trT = tr(T)
            @test eltype(trT) == Float64
            @test trT isa Number
        end
        Bumper.reset_buffer!(buf)
    end

    # ── QN / block sparse buffer tests ──

    @testset "to_buffer QN ITensor" begin
        i = Index([QN(0) => 2, QN(1) => 1], "i")
        A = random_itensor(QN(0), i', dag(i))
        @test flux(A) == QN(0)
        buf = Bumper.SlabBuffer()
        Ab = to_buffer(A, buf)
        @test ndata(NDTensors.storage(Ab)) isa NDTensors.UnsafeArray
        @test flux(Ab) == QN(0)
        @test nnzblocks(Ab) == nnzblocks(A)
        @test Ab[i' => 1, dag(i) => 1] ≈ A[i' => 1, dag(i) => 1]
        @test Ab[i' => 2, dag(i) => 2] ≈ A[i' => 2, dag(i) => 2]
    end

    @testset "buffer×heap QN contraction returns heap (mixed types)" begin
        i = Index([QN(0) => 2, QN(1) => 1], "i")
        j = Index([QN(0) => 2, QN(1) => 1], "j")
        k = Index([QN(0) => 2, QN(1) => 1], "k")
        A = random_itensor(QN(0), i, j)
        B = random_itensor(QN(0), j', k)
        buf = Bumper.SlabBuffer()
        Ab = to_buffer(A, buf)
        with_alloc_buffer(buf) do
            C = Ab * B
            # Mixed buffer×heap → heap (promote_type: UnsafeArray × Vector = Vector)
            @test ndata(NDTensors.storage(C)) isa Vector
            @test flux(C) == QN(0)
            @test C ≈ A * B
        end
    end

    @testset "SVD of buffer-backed QN ITensor" begin
        i = Index([QN(0) => 2, QN(1) => 1], "i")
        j = Index([QN(0) => 2, QN(1) => 1], "j")
        A = random_itensor(QN(0), i, dag(j))
        buf = Bumper.SlabBuffer()
        Ab = to_buffer(A, buf)
        with_alloc_buffer(buf) do
            U, S, V = svd(Ab, i)
            @test ndata(NDTensors.storage(U)) isa NDTensors.UnsafeArray
            @test ndata(NDTensors.storage(V)) isa NDTensors.UnsafeArray
            @test ndata(NDTensors.storage(S)) isa NDTensors.UnsafeArray
            @test ndims(U) == 2 && ndims(V) == 2 && ndims(S) == 2
            @test flux(U) == QN(0)
        end
    end

    # ── Combiner ──

    @testset "combine Dense ITensor with combiner" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(4, "j")
        k = Index(5, "k")

        A = to_buffer(random_itensor(i, j, k), buf)
        C = combiner(i, j)
        with_alloc_buffer(buf) do
            AC = A * C
            @test ndata(NDTensors.storage(AC)) isa NDTensors.UnsafeArray
            @test hassameinds(AC, (combinedind(C), k))
            # Uncombine
            Ap = AC * dag(C)
            @test ndata(NDTensors.storage(Ap)) isa NDTensors.UnsafeArray
            @test hassameinds(Ap, inds(A))
            @test A ≈ Ap
        end
        Bumper.reset_buffer!(buf)
    end

    @testset "combine QN ITensor with combiner" begin
        buf = Bumper.SlabBuffer()
        i1 = Index([QN(0) => 2, QN(1) => 3], "i1")
        A = random_itensor(i1', dag(i1))
        Ab = to_buffer(A, buf)
        C = combiner(dag(i1); dir = ITensors.Out)
        with_alloc_buffer(buf) do
            AC = Ab * C
            @test ndata(NDTensors.storage(AC)) isa NDTensors.UnsafeArray
            @test nnzblocks(AC) == nnzblocks(A)
            Ap = AC * dag(C)
            @test ndata(NDTensors.storage(Ap)) isa NDTensors.UnsafeArray
            @test hassameinds(Ap, inds(A))
            @test A ≈ Ap
        end
        Bumper.reset_buffer!(buf)
    end

    # ── onehot / setelt ──

    @testset "onehot of buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        with_alloc_buffer(buf) do
            T = onehot(i => 2)
            @test T isa ITensor
            @test T[i => 1] ≈ 0.0
            @test T[i => 2] ≈ 1.0
            @test T[i => 3] ≈ 0.0
        end
    end

    @testset "setelt on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        j = Index(3, "j")
        with_alloc_buffer(buf) do
            T = setelt(i => 2, j => 1)
            @test T isa ITensor
            @test T[i => 2, j => 1] ≈ 1.0
            @test T[i => 1, j => 1] ≈ 0.0
        end
    end

    # ── Unary minus ──

    @testset "unary - on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        M = [1.0, 2.0, 3.0]
        A = to_buffer(itensor(M, i), buf)
        with_alloc_buffer(buf) do
            B = -A
            @test ndata(NDTensors.storage(B)) isa NDTensors.UnsafeArray
            @test B ≈ itensor(-M, i)
        end
    end

    # ── similar ──

    @testset "similar on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        A = to_buffer(random_itensor(i), buf)
        with_alloc_buffer(buf) do
            B = similar(A)
            @test ndata(NDTensors.storage(B)) isa NDTensors.UnsafeArray
            @test inds(B) == inds(A)
        end
    end

    # ── isapprox ──

    @testset "isapprox on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        A = to_buffer(random_itensor(i), buf)
        B = to_buffer(random_itensor(i), buf)
        with_alloc_buffer(buf) do
            @test A ≈ A
            @test !(A ≈ B)
        end
    end

    # ── copyto! ──

    @testset "copyto! on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(2, "i")
        j = Index(2, "j")
        M = [1 2; 3 4]
        A = to_buffer(itensor(M, i, j), buf)
        N = 2 * M
        B = to_buffer(itensor(N, i, j), buf)
        with_alloc_buffer(buf) do
            copyto!(A, B)
            @test A ≈ B
        end
    end

    # ── mul! (3-arg) ──

    @testset "mul! on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(3, "i")
        a = [1.0, 2.0, 3.0]
        A = to_buffer(itensor(a, i), buf)
        with_alloc_buffer(buf) do
            B = similar(A)
            @test ndata(NDTensors.storage(B)) isa NDTensors.UnsafeArray
            mul!(B, A, 2.0)
            @test B ≈ itensor(2.0 * a, i)
        end
    end

    # ── Broadcast ──

    @testset "broadcast fill! on buffer ITensor" begin
        buf = Bumper.SlabBuffer()
        i = Index(4, "i")
        j = Index(4, "j")
        A = to_buffer(random_itensor(i, j), buf)
        A .= 1.0
        @test all(ndata(NDTensors.storage(A)) .== 1.0)
    end

    # ── Threaded buffer QN contraction ──
    if Threads.nthreads() > 1
        @testset "threaded QN buffer contraction" begin
            i = Index([QN(0) => 5, QN(1) => 3], "i")
            j = Index([QN(0) => 5, QN(1) => 3], "j")
            k = Index([QN(0) => 5, QN(1) => 3], "k")
            A = random_itensor(QN(0), i, j)
            B = random_itensor(QN(0), j', k)
            C_ref = A * B

            buf = Bumper.SlabBuffer()
            Ab = to_buffer(A, buf)
            Bb = to_buffer(B, buf)
            enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                Ab * Bb
            end
            disable_threaded_blocksparse()
            @test ndata(NDTensors.storage(C_thr)) isa NDTensors.UnsafeArray
            with_alloc_buffer(buf) do
                @test C_thr ≈ C_ref
            end
        end
    end
end

nothing
end
