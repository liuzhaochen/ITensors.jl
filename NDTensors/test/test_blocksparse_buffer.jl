# ── Fallback tests (no TBLIS) ──
@eval module $(gensym())
using NDTensors: NDTensors, Bumper, BlockSparse, BlockSparseTensor, blockoffsets, nnz, nnzblocks, data, storage, inds, similar, blockview, set_alloc_buffer!, get_alloc_buffer, with_alloc_buffer, eachnzblock, dense, tensor, contract, svd, qr, to_buffer
using LinearAlgebra: norm, Diagonal, BLAS
using Random: randn!
using ITensors: QN, Index, random_itensor, dag, prime
using Test: @test, @test_throws, @testset

# Single-thread BLAS for nested threading (block-sparse outer + BLAS inner)
if Threads.nthreads() > 1
    BLAS.set_num_threads(1)
end

@testset "BlockSparse buffer" begin
    @testset "BlockSparse(buf, blockoffsets, dim)" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        s = BlockSparse(buf, blockoffsets(A), nnz(A))
        @test data(s) isa NDTensors.UnsafeArray
        @test length(data(s)) == nnz(A)
    end

    @testset "BlockSparse(::Type{ElT}, buf, blockoffsets, dim)" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        s = BlockSparse(ComplexF64, buf, blockoffsets(A), nnz(A))
        @test eltype(s) == ComplexF64
        @test data(s) isa NDTensors.UnsafeArray
    end

    @testset "similar preserves data type" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer(); set_alloc_buffer!(buf)
        s = similar(typeof(storage(A)), blockoffsets(A), inds(A))
        @test data(s) isa Vector{Float64}
        set_alloc_buffer!(nothing)
    end

    @testset "similar without buffer (fallback)" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        s = similar(typeof(storage(A)), blockoffsets(A), inds(A))
        @test data(s) isa Vector
    end

    @testset "copy buffer BlockSparse to heap" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        s_ua = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
        d = data(s_ua); for i in eachindex(d); d[i] = randn(); end
        s_copy = copy(s_ua)
        @test data(s_copy) isa Vector
        @test data(s_copy) == data(s_ua)
    end

    @testset "copy buffer BlockSparse with buffer active" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            s = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
            d = data(s); for i in eachindex(d); d[i] = randn(); end
            s_copy = copy(s)
            @test data(s_copy) isa NDTensors.UnsafeArray
            @test data(s_copy) == data(s)
        end
    end

    @testset "BlockSparseTensor from buffer storage" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        s = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
        T = tensor(s, inds(A))
        @test storage(T) === s
        @test nnz(T) == nnz(A)
    end

    @testset "getindex/setindex on buffer BlockSparse" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        s = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
        d = data(s); for i in eachindex(d); d[i] = randn(); end
        T = tensor(s, inds(A))
        for i in 1:nnz(A)
            @test data(s)[i] != 0.0
        end
    end

    @testset "conj/real/imag on buffer-backed Complex BlockSparse" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{ComplexF64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            s = BlockSparse(ComplexF64, buf, blockoffsets(A), nnz(A))
            d = data(s); for i in eachindex(d); d[i] = complex(randn(), randn()); end
            T = tensor(s, inds(A))
            Tc = conj(T)
            @test data(storage(Tc)) isa NDTensors.UnsafeArray
            @test nnzblocks(Tc) == nnzblocks(T)
            @test ndims(Tc) == ndims(T)
            Tr = real(Tc)
            @test data(storage(Tr)) isa NDTensors.UnsafeArray
            @test ndims(Tr) == ndims(T)
            Ti = imag(Tc)
            @test data(storage(Ti)) isa NDTensors.UnsafeArray
            @test ndims(Ti) == ndims(T)
        end
    end

    @testset "scalar multiply buffer BlockSparse" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        s = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
        d = data(s); for i in eachindex(d); d[i] = randn(); end
        T = tensor(s, inds(A))
        with_alloc_buffer(buf) do
            B = 2.0 * T
            @test data(storage(B)) isa NDTensors.UnsafeArray
            @test nnzblocks(B) == nnzblocks(T)
        end
    end

    @testset "contract buffer×buffer BlockSparse" begin
        locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
        locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
        A_heap = BlockSparseTensor{Float64}(locs1, inds1...)
        B_heap = BlockSparseTensor{Float64}(locs2, inds2...)
        buf = Bumper.SlabBuffer()
        sA = BlockSparse(Float64, buf, blockoffsets(A_heap), nnz(A_heap))
        dA = data(sA); for i in eachindex(dA); dA[i] = randn(); end
        A = tensor(sA, inds(A_heap))
        sB = BlockSparse(Float64, buf, blockoffsets(B_heap), nnz(B_heap))
        dB = data(sB); for i in eachindex(dB); dB[i] = randn(); end
        B = tensor(sB, inds(B_heap))
        with_alloc_buffer(buf) do
            C = contract(A, (1, 2), B, (2, 3), (1, 3))
            @test data(storage(C)) isa NDTensors.UnsafeArray
            @test inds(C) == (inds1[1], inds2[2])
        end
    end

    @testset "SVD of buffer-backed BlockSparse" begin
        locs = [(2, 1), (1, 2)]; indsA = ([2, 2], [2, 2])
        A_heap = BlockSparseTensor{Float64}(locs, indsA...)
        d = data(storage(A_heap)); for i in eachindex(d); d[i] = randn(); end
        A_ref = Matrix(dense(A_heap))
        buf = Bumper.SlabBuffer()
        A = to_buffer(A_heap, buf)
        U, S, V = with_alloc_buffer(buf) do
            Us, Ss, Vs = svd(A)
            @test data(storage(Us)) isa NDTensors.UnsafeArray
            @test data(storage(Vs)) isa NDTensors.UnsafeArray
            @test ndims(Us) == 2 && ndims(Ss) == 2 && ndims(Vs) == 2
            Us, Ss, Vs
        end
        Uh, Sh, Vh = copy(U), copy(S), copy(V)
        US = contract(Uh, (1, -1), Sh, (-1, 2))
        USV = contract(US, (1, -1), Vh, (2, -1))
        @test Matrix(dense(USV)) ≈ A_ref
    end

    @testset "QR of buffer-backed BlockSparse" begin
        locs = [(2, 1), (1, 2)]; indsA = ([2, 2], [2, 2])
        A_heap = BlockSparseTensor{Float64}(locs, indsA...)
        d = data(storage(A_heap)); for i in eachindex(d); d[i] = randn(); end
        buf = Bumper.SlabBuffer()
        A = to_buffer(A_heap, buf)
        A_ref = Matrix(dense(A_heap))
        Q, R = with_alloc_buffer(buf) do
            Q, R = qr(A; positive=false)
            @test data(storage(Q)) isa NDTensors.UnsafeArray
            @test data(storage(R)) isa NDTensors.UnsafeArray
            @test ndims(Q) == 2 && ndims(R) == 2
            Q, R  # return buffer-backed, copy outside scope
        end
        Qh, Rh = copy(Q), copy(R)  # heap copy (no active buffer)
        @test Matrix(dense(Qh)) * Matrix(dense(Rh)) ≈ A_ref
    end

    @testset "to_buffer BlockSparse" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        d = data(storage(A)); d .= randn()
        buf = Bumper.SlabBuffer()
        Ab = to_buffer(A, buf)
        @test data(storage(Ab)) isa NDTensors.UnsafeArray
        @test data(storage(Ab)) == data(storage(A))
        @test blockoffsets(Ab) == blockoffsets(A)
    end

    @testset "contract heap×buffer BlockSparse (DMRG pattern)" begin
        locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
        locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
        H = BlockSparseTensor{Float64}(locs1, inds1...)
        psi_heap = BlockSparseTensor{Float64}(locs2, inds2...)
        buf = Bumper.SlabBuffer()
        psi = to_buffer(psi_heap, buf)
        with_alloc_buffer(buf) do
            C = contract(H, (1, 2), psi, (2, 3), (1, 3))
            @test data(storage(C)) isa Vector{Float64}
        end
    end

    @testset "dense(A) of buffer-backed BlockSparse" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            s = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
            d = data(s); for i in eachindex(d); d[i] = randn(); end
            T = tensor(s, inds(A))
            D = dense(T)
            @test ndims(D) == ndims(T)
        end
    end

    @testset "sum/prod of buffer-backed BlockSparse" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            s = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
            d = data(s); for i in eachindex(d); d[i] = randn(); end
            T = tensor(s, inds(A))
            @test sum(T) ≈ sum(dense(T))
        end
    end

    @testset "permutedims of buffer-backed BlockSparse" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A_heap = BlockSparseTensor{Float64}(locs, indsA...)
        d = data(storage(A_heap)); for i in eachindex(d); d[i] = randn(); end
        T_heap = tensor(storage(A_heap), inds(A_heap))
        Tp_ref = NDTensors.permutedims(T_heap, (2, 1))
        buf = Bumper.SlabBuffer()
        Tp = with_alloc_buffer(buf) do
            s = BlockSparse(Float64, buf, blockoffsets(A_heap), nnz(A_heap))
            d = data(s); for i in eachindex(d); d[i] = data(storage(A_heap))[i]; end
            T = tensor(s, inds(A_heap))
            Tp = NDTensors.permutedims(T, (2, 1))
            @test data(storage(Tp)) isa NDTensors.UnsafeArray
            @test nnz(T) == nnz(Tp)
            Tp  # return for heap copy outside
        end
        @test Matrix(dense(copy(Tp))) ≈ Matrix(dense(Tp_ref))
    end

    @testset "blockview of buffer-backed BlockSparse" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            s = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
            d = data(s); for i in eachindex(d); d[i] = randn(); end
            T = tensor(s, inds(A))
            for b in eachnzblock(T)
                bv = blockview(T, b)
                @test ndims(bv) == 2
            end
        end
    end

    @testset "buffer with_alloc_buffer context" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        @test get_alloc_buffer() === nothing
        with_alloc_buffer(buf) do
            s = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
            @test data(s) isa NDTensors.UnsafeArray
            @test get_alloc_buffer() === buf
        end
        @test get_alloc_buffer() === nothing
    end

    # ── Float32 ──
    @testset "BlockSparse{Float32}(buf, blockoffsets, dim)" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        s = BlockSparse(Float32, buf, blockoffsets(A), nnz(A))
        @test eltype(s) == Float32
        @test data(s) isa NDTensors.UnsafeArray{Float32}
    end

    @testset "Float32 similar with buffer" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float32}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        s = BlockSparse(Float32, buf, blockoffsets(A), nnz(A))
        @test eltype(s) == Float32
        @test data(s) isa NDTensors.UnsafeArray{Float32}
    end

    @testset "Float32 contract buffer×buffer BlockSparse" begin
        locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
        locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
        A_heap = BlockSparseTensor{Float32}(locs1, inds1...)
        B_heap = BlockSparseTensor{Float32}(locs2, inds2...)
        buf = Bumper.SlabBuffer()
        sA = BlockSparse(Float32, buf, blockoffsets(A_heap), nnz(A_heap))
        dA = data(sA); for i in eachindex(dA); dA[i] = randn(); end
        A = tensor(sA, inds(A_heap))
        sB = BlockSparse(Float32, buf, blockoffsets(B_heap), nnz(B_heap))
        dB = data(sB); for i in eachindex(dB); dB[i] = randn(); end
        B = tensor(sB, inds(B_heap))
        with_alloc_buffer(buf) do
            C = contract(A, (1, 2), B, (2, 3), (1, 3))
            @test eltype(C) == Float32
            @test data(storage(C)) isa NDTensors.UnsafeArray{Float32}
        end
    end

    # ── Scalar division ──
    @testset "scalar divide buffer BlockSparse" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        s = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
        d = data(s); for i in eachindex(d); d[i] = randn(); end
        T = tensor(s, inds(A))
        old_nnz = nnz(T)
        with_alloc_buffer(buf) do
            B = T / 2.0
            @test data(storage(B)) isa NDTensors.UnsafeArray
            @test nnzblocks(B) == nnzblocks(T)
            @test nnz(B) == old_nnz
            @test B[1, 1] ≈ T[1, 1] / 2.0
        end
    end

    # ── ComplexF32 ──
    @testset "ComplexF32 conj/real/imag on buffer-backed BlockSparse" begin
        locs = [(1, 2), (2, 1)]; indsA = ([2, 3], [4, 5])
        A = BlockSparseTensor{ComplexF32}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            s = BlockSparse(ComplexF32, buf, blockoffsets(A), nnz(A))
            d = data(s); for i in eachindex(d); d[i] = complex(randn(), randn()); end
            T = tensor(s, inds(A))
            Tc = conj(T)
            @test data(storage(Tc)) isa NDTensors.UnsafeArray
            @test nnzblocks(Tc) == nnzblocks(T)
            @test ndims(Tc) == ndims(T)
            @test eltype(Tc) == ComplexF32
            Tr = real(Tc)
            @test data(storage(Tr)) isa NDTensors.UnsafeArray
            @test eltype(Tr) == Float32
            Ti = imag(Tc)
            @test data(storage(Ti)) isa NDTensors.UnsafeArray
            @test eltype(Ti) == Float32
        end
    end

    # ── Multi-threaded contraction — fallback (no TBLIS) ──
    if Threads.nthreads() > 1
        @testset "threaded contraction (fallback) Float64" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{Float64}(locs1, inds1...)
            B_heap = BlockSparseTensor{Float64}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = randn(); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = randn(); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (fallback) Float32" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{Float32}(locs1, inds1...)
            B_heap = BlockSparseTensor{Float32}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = randn(Float32); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = randn(Float32); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test eltype(C_thr) == Float32
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (fallback) ComplexF64" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{ComplexF64}(locs1, inds1...)
            B_heap = BlockSparseTensor{ComplexF64}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = complex(randn(), randn()); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = complex(randn(), randn()); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (fallback) ComplexF64 × Float64" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{ComplexF64}(locs1, inds1...)
            B_heap = BlockSparseTensor{Float64}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = complex(randn(), randn()); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = randn(); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (fallback) Float64 × ComplexF64" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{Float64}(locs1, inds1...)
            B_heap = BlockSparseTensor{ComplexF64}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = randn(); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = complex(randn(), randn()); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (fallback) large blocks" begin
            locs = [(1, 3), (2, 1), (3, 2)]
            indsA = ([30, 40, 50], [60, 70, 80])
            indsB = ([60, 70, 80], [90, 100, 110])
            A_heap = BlockSparseTensor{Float64}(locs, indsA...)
            B_heap = BlockSparseTensor{Float64}(locs, indsB...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = randn(); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = randn(); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (fallback) diff dims 3D×2D→3D" begin
            # Dimension mismatch: 3D × 2D → 3D (common in DMRG env construction)
            # Known bug: fallback BLAS gemm layout wrong for dim-mismatch
            # https://github.com/ITensor/ITensors.jl/issues/XXX
            locs3 = [(1, 1, 1)]
            inds3 = ([2], [3], [4])
            A = BlockSparseTensor{Float64}(locs3, inds3...)
            d = data(storage(A)); d[1:24] .= randn(24)
            locs2 = [(1, 1)]
            inds2 = ([3], [5])
            B = BlockSparseTensor{Float64}(locs2, inds2...)
            d = data(storage(B)); d[1:15] .= randn(15)
            C_ref = contract(A, (1, 2, 3), B, (2, 4), (1, 3, 4))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                Ab = NDTensors.to_buffer(A, buf)
                Bb = NDTensors.to_buffer(B, buf)
                contract(Ab, (1, 2, 3), Bb, (2, 4), (1, 3, 4))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (fallback) C-perm 2D swapped" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A = BlockSparseTensor{Float64}(locs1, inds1...)
            B = BlockSparseTensor{Float64}(locs2, inds2...)
            dA = data(storage(A)); for i in eachindex(dA); dA[i] = randn(); end
            dB = data(storage(B)); for i in eachindex(dB); dB[i] = randn(); end
            C_ref = contract(A, (1, 2), B, (2, 3), (3, 1))
            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                Ab = NDTensors.to_buffer(A, buf)
                Bb = NDTensors.to_buffer(B, buf)
                contract(Ab, (1, 2), Bb, (2, 3), (3, 1))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test Matrix(dense(copy(C_ref))) ≈ Matrix(dense(copy(C_thr)))
        end

        @testset "threaded contraction (fallback) C-perm 3D reordered" begin
            locs3 = [(1, 1, 1)]
            inds3 = ([2], [3], [4])
            A = BlockSparseTensor{Float64}(locs3, inds3...)
            d = data(storage(A)); d[1:24] .= randn(24)
            locs2 = [(1, 1)]
            inds2 = ([3], [5])
            B = BlockSparseTensor{Float64}(locs2, inds2...)
            d = data(storage(B)); d[1:15] .= randn(15)
            C_ref = contract(A, (1, 2, 3), B, (2, 4), (3, 1, 4))
            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                Ab = NDTensors.to_buffer(A, buf)
                Bb = NDTensors.to_buffer(B, buf)
                contract(Ab, (1, 2, 3), Bb, (2, 4), (3, 1, 4))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test dense(copy(C_ref)) ≈ dense(copy(C_thr))
        end

        @testset "threaded contraction (fallback) QN multi-contracted order" begin
            # Regression test: contracted indices in different order between
            # A and B (A: [-1,-2], B: [-2,-1]). Previous _contract_fallback!
            # GEMM path summed over mismatched (ci1,ci2) pairs.
            # Pattern: 4D×3D→3D with labels (3,-1,4,-2)×(-2,-1,5).
            lqr = Index([QN("Sz",5)=>3, QN("Sz",3)=>8, QN("Sz",1)=>17,
                         QN("Sz",-1)=>15, QN("Sz",-3)=>7, QN("Sz",-5)=>2], "Link,qr")
            ll  = Index([QN()=>1, QN("Sz",0)=>6, QN("Sz",2)=>6,
                         QN("Sz",-2)=>6, QN("Sz",0)=>1], "Link,l=32")
            s   = Index([QN("Sz",1)=>1, QN("Sz",-1)=>1], "S=1/2,Site,n=2")
            p1  = Index([QN("Sz",6)=>1, QN("Sz",4)=>5, QN("Sz",2)=>12,
                         QN("Sz",0)=>16, QN("Sz",-2)=>12,
                         QN("Sz",-4)=>5, QN("Sz",-6)=>1], "Link,qr")

            L   = random_itensor(QN("Sz",0), dag(lqr), prime(lqr), ll, dag(s), prime(s))
            phi = random_itensor(QN("Sz",0), p1, lqr, s)
            R   = random_itensor(QN("Sz",0), dag(p1), dag(ll), prime(p1))

            tT = tensor(L * phi)  # 4D with labels (3,-1,4,-2)
            tR = tensor(R)         # 3D with labels (-2,-1,5)
            lt, lr = (3, -1, 4, -2), (-2, -1, 5)

            C_ref = contract(tT, lt, tR, lr)

            buf = Bumper.SlabBuffer{2^25}()
            NDTensors.enable_threaded_blocksparse()
            C_buf = with_alloc_buffer(buf) do
                Bumper.reset_buffer!(buf)
                contract(to_buffer(tT, buf), lt, to_buffer(tR, buf), lr)
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_buf)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_buf
        end
    end
end

nothing
end

# ── TBLIS tests ──
@eval module $(gensym())
using NDTensors: NDTensors, Bumper, BlockSparse, BlockSparseTensor, blockoffsets, nnz, nnzblocks, data, storage, inds, to_buffer, with_alloc_buffer, contract
using LinearAlgebra
using Random: randn!
using Test: @test, @testset
using TBLIS

@testset "BlockSparse buffer (TBLIS)" begin
    # Single-thread BLAS for nested threading
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
    end

    # ── Multi-threaded contraction — TBLIS ──
    if Threads.nthreads() > 1
        @testset "threaded contraction (TBLIS) Float64" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{Float64}(locs1, inds1...)
            B_heap = BlockSparseTensor{Float64}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = randn(); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = randn(); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (TBLIS) Float32" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{Float32}(locs1, inds1...)
            B_heap = BlockSparseTensor{Float32}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = randn(Float32); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = randn(Float32); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test eltype(C_thr) == Float32
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (TBLIS) ComplexF64" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{ComplexF64}(locs1, inds1...)
            B_heap = BlockSparseTensor{ComplexF64}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = complex(randn(), randn()); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = complex(randn(), randn()); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (TBLIS) ComplexF64 × Float64" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{ComplexF64}(locs1, inds1...)
            B_heap = BlockSparseTensor{Float64}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = complex(randn(), randn()); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = randn(); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (TBLIS) Float64 × ComplexF64" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{Float64}(locs1, inds1...)
            B_heap = BlockSparseTensor{ComplexF64}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = randn(); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = complex(randn(), randn()); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction (TBLIS) large blocks" begin
            locs = [(1, 3), (2, 1), (3, 2)]
            indsA = ([30, 40, 50], [60, 70, 80])
            indsB = ([60, 70, 80], [90, 100, 110])
            A_heap = BlockSparseTensor{Float64}(locs, indsA...)
            B_heap = BlockSparseTensor{Float64}(locs, indsB...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = randn(); end
            dB = data(storage(B_heap)); for i in eachindex(dB); dB[i] = randn(); end
            C_ref = contract(A_heap, (1, 2), B_heap, (2, 3), (1, 3))

            buf = Bumper.SlabBuffer()
            NDTensors.enable_threaded_blocksparse()
            C_thr = with_alloc_buffer(buf) do
                A_buf = NDTensors.to_buffer(A_heap, buf)
                B_buf = NDTensors.to_buffer(B_heap, buf)
                contract(A_buf, (1, 2), B_buf, (2, 3), (1, 3))
            end
            NDTensors.disable_threaded_blocksparse()
            @test data(storage(C_thr)) isa NDTensors.UnsafeArray
            @test C_ref ≈ C_thr
        end
    end
end

nothing
end
