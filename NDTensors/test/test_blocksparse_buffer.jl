@eval module $(gensym())
using NDTensors: NDTensors, Bumper, BlockSparse, BlockSparseTensor, blockoffsets, nnz, nnzblocks, data, storage, inds, similar, blockview, set_alloc_buffer!, get_alloc_buffer, with_alloc_buffer, storage, eachnzblock, dense, tensor, contract, svd, qr, to_buffer
using LinearAlgebra: norm, Diagonal
using Random: randn!
using Test: @test, @test_throws, @testset
using TBLIS

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
        buf = Bumper.SlabBuffer()
        A = to_buffer(A_heap, buf)
        with_alloc_buffer(buf) do
            U, S, V = svd(A)
            @test data(storage(U)) isa NDTensors.UnsafeArray
            @test data(storage(S)) isa NDTensors.UnsafeArray
            @test data(storage(V)) isa NDTensors.UnsafeArray
            @test ndims(U) == 2 && ndims(S) == 2 && ndims(V) == 2
            US = contract(U, (1, -1), S, (-1, 2))
            @test data(storage(US)) isa NDTensors.UnsafeArray
            USV = contract(US, (1, -1), V, (2, -1))
            @test data(storage(USV)) isa NDTensors.UnsafeArray
            @test dense(USV) ≈ dense(A)
        end
    end

    @testset "QR of buffer-backed BlockSparse" begin
        locs = [(2, 1), (1, 2)]; indsA = ([2, 2], [2, 2])
        A_heap = BlockSparseTensor{Float64}(locs, indsA...)
        d = data(storage(A_heap)); for i in eachindex(d); d[i] = randn(); end
        buf = Bumper.SlabBuffer()
        A = to_buffer(A_heap, buf)
        with_alloc_buffer(buf) do
            Q, R = qr(A; positive=false)
            @test data(storage(Q)) isa NDTensors.UnsafeArray
            @test data(storage(R)) isa NDTensors.UnsafeArray
            @test ndims(Q) == 2 && ndims(R) == 2
            A_mat = Matrix(dense(A)); Q_mat = Matrix(dense(Q)); R_mat = Matrix(dense(R))
            @test Q_mat * R_mat ≈ A_mat
        end
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
            # Mixed heap×buffer → heap (promote_type rule: Vector × UnsafeArray = Vector)
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
        A = BlockSparseTensor{Float64}(locs, indsA...)
        buf = Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            s = BlockSparse(Float64, buf, blockoffsets(A), nnz(A))
            d = data(s); for i in eachindex(d); d[i] = randn(); end
            T = tensor(s, inds(A))
            Tp = NDTensors.permutedims(T, (2, 1))
            @test data(storage(Tp)) isa NDTensors.UnsafeArray
            @test nnz(T) == nnz(Tp)
        end
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

    # ── Multi-threaded contraction ──
    # These tests require julia -t N with N >= 2
    if Threads.nthreads() > 1
        @testset "threaded contraction buffer Float64" begin
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

        @testset "threaded contraction buffer Float32" begin
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
            @test C_ref ≈ C_thr
        end

        @testset "threaded contraction buffer large blocks" begin
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

        @testset "threaded contraction buffer ComplexF64 × Float64" begin
            locs1 = [(1, 2), (2, 1)]; inds1 = ([2, 3], [4, 5])
            locs2 = [(1, 2), (2, 1)]; inds2 = ([4, 5], [6, 7])
            A_heap = BlockSparseTensor{ComplexF64}(locs1, inds1...)
            B_heap = BlockSparseTensor{Float64}(locs2, inds2...)
            dA = data(storage(A_heap)); for i in eachindex(dA); dA[i] = randn() + im * randn(); end
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
