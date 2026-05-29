@eval module $(gensym())
using NDTensors
using NDTensors: Dense, get_alloc_buffer, with_alloc_buffer, storage
using LinearAlgebra: svd, qr
using Test: @test, @testset

@testset "Buffer-allocated Dense" begin
    @testset "Basic buffer allocation" begin
        buf = NDTensors.Bumper.SlabBuffer()
        result = with_alloc_buffer(buf) do
            d = Dense{Float64}(get_alloc_buffer(), 10)
            @test length(d) == 10
            @test eltype(d) == Float64
            @test d isa Dense{Float64}
            @test data(d) isa NDTensors.UnsafeArray{Float64, 1}
            return d
        end
    end

    @testset "Buffer allocation with default eltype" begin
        buf = NDTensors.Bumper.SlabBuffer()
        result = with_alloc_buffer(buf) do
            d = Dense(get_alloc_buffer(), 8)
            @test length(d) == 8
            @test eltype(d) == Float64
            @test data(d) isa NDTensors.UnsafeArray{Float64, 1}
            return d
        end
    end

    @testset "Buffer allocation with undef" begin
        buf = NDTensors.Bumper.SlabBuffer()
        result = with_alloc_buffer(buf) do
            d = Dense{Float64}(get_alloc_buffer(), undef, 5)
            @test length(d) == 5
            @test eltype(d) == Float64
            @test data(d) isa NDTensors.UnsafeArray
            return d
        end
    end

    @testset "Buffer allocation with undef and default eltype" begin
        buf = NDTensors.Bumper.SlabBuffer()
        result = with_alloc_buffer(buf) do
            d = Dense(get_alloc_buffer(), undef, 5)
            @test length(d) == 5
            @test eltype(d) == Float64
            @test data(d) isa NDTensors.UnsafeArray
            return d
        end
    end

    @testset "Buffer-allocated Dense values" begin
        buf = NDTensors.Bumper.SlabBuffer()
        result = with_alloc_buffer(buf) do
            d = Dense{Float64}(get_alloc_buffer(), 10)
            @test data(d) isa NDTensors.UnsafeArray
            fill!(d, 3.14)
            for i in 1:10
                @test d[i] ≈ 3.14
            end
            d[3] = 2.71
            @test d[3] ≈ 2.71
            return d
        end
    end

    @testset "Zero-length buffer allocation" begin
        buf = NDTensors.Bumper.SlabBuffer()
        result = with_alloc_buffer(buf) do
            d = Dense{Float64}(get_alloc_buffer(), 0)
            @test length(d) == 0
            @test data(d) isa NDTensors.UnsafeArray
            return d
        end
    end

    @testset "Copy inside buffer context" begin
        buf = NDTensors.Bumper.SlabBuffer()
        result = with_alloc_buffer(buf) do
            d = Dense{Float64}(get_alloc_buffer(), 5)
            fill!(d, 1.0)
            d2 = copy(d)
            @test length(d2) == 5
            @test eltype(d2) == Float64
            @test data(d2) isa NDTensors.UnsafeArray{Float64, 1}
            for i in 1:5
                @test d2[i] ≈ 1.0
            end
            return (d, d2)
        end
    end

    @testset "Copy outside buffer context falls back to heap" begin
        buf = NDTensors.Bumper.SlabBuffer()
        d = with_alloc_buffer(buf) do
            return Dense{Float64}(get_alloc_buffer(), 5)
        end
        fill!(d, 42.0)
        # Copy with no buffer active → heap allocation
        d2 = copy(d)
        @test length(d2) == 5
        @test eltype(d2) == Float64
        @test data(d2) isa Vector{Float64}
        for i in 1:5
            @test d2[i] ≈ 42.0
        end
    end

    @testset "Buffer reset and reuse" begin
        buf = NDTensors.Bumper.SlabBuffer()
        result = with_alloc_buffer(buf) do
            d1 = Dense{Float64}(get_alloc_buffer(), 100)
            fill!(d1, 1.0)
            # Reset buffer
            NDTensors.Bumper.reset_buffer!(buf)
            d2 = Dense{Float64}(get_alloc_buffer(), 100)
            fill!(d2, 2.0)
            @test d2[1] ≈ 2.0
            # After reset, old data should have been overwritten
            @test d1[1] ≈ 2.0  # d1 points to recycled memory
            return nothing
        end
    end

    @testset "Nesting with_alloc_buffer" begin
        outer_buf = NDTensors.Bumper.SlabBuffer()
        inner_buf = NDTensors.Bumper.SlabBuffer()
        result = with_alloc_buffer(outer_buf) do
            @test get_alloc_buffer() === outer_buf
            d_outer = Dense{Float64}(get_alloc_buffer(), 3)
            fill!(d_outer, 1.0)
            with_alloc_buffer(inner_buf) do
                @test get_alloc_buffer() === inner_buf
                d_inner = Dense{Float64}(get_alloc_buffer(), 3)
                fill!(d_inner, 2.0)
                @test d_outer[1] ≈ 1.0
                @test d_inner[1] ≈ 2.0
            end
            @test get_alloc_buffer() === outer_buf
            return (d_outer,)
        end
    end

    @testset "Buffer not active returns nothing" begin
        @test get_alloc_buffer() === nothing
        d = Dense{Float64}(10)
        @test d isa Dense{Float64, Vector{Float64}}
    end

    @testset "Buffer-aware similar" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            d = Dense{Float64}(get_alloc_buffer(), 10)
            s = NDTensors.similar(d, (5,))
            @test length(s) == 5
            @test eltype(s) == Float64
            @test data(s) isa NDTensors.UnsafeArray{Float64, 1}
        end
    end

    @testset "Buffer-aware similar with eltype change" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            d = Dense{Float64}(get_alloc_buffer(), 10)
            s = NDTensors.similar(d, ComplexF64, (5,))
            @test length(s) == 5
            @test eltype(s) == ComplexF64
            @test data(s) isa NDTensors.UnsafeArray{ComplexF64, 1}
        end
    end

    @testset "Similar falls back to heap without buffer" begin
        d = Dense{Float64}(10)
        s = NDTensors.similar(d, (5,))
        @test data(s) isa Vector{Float64}
    end

    @testset "DenseTensor buffer constructor" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            dt = NDTensors.DenseTensor(buf, (2, 3))
            @test size(dt) == (2, 3)
            @test eltype(dt) == Float64
            @test data(dt) isa NDTensors.UnsafeArray
        end
    end

    @testset "DenseTensor buffer constructor with eltype" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            dt = NDTensors.DenseTensor(ComplexF64, buf, (2, 3))
            @test size(dt) == (2, 3)
            @test eltype(dt) == ComplexF64
            @test data(dt) isa NDTensors.UnsafeArray{ComplexF64}
        end
    end

    @testset "Tensor buffer constructor" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            t = NDTensors.Tensor(buf, (3, 4))
            @test size(t) == (3, 4)
            @test eltype(t) == Float64
            @test data(t) isa NDTensors.UnsafeArray
        end
    end

    @testset "Tensor buffer constructor undef" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            t = NDTensors.Tensor(buf, undef, (2, 2))
            @test size(t) == (2, 2)
            @test data(t) isa NDTensors.UnsafeArray
        end
    end

    @testset "Buffer-allocated tensor contraction" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            a = NDTensors.DenseTensor(Float64, buf, (2, 3, 4))
            b = NDTensors.DenseTensor(Float64, buf, (4, 5))
            fill!(a, 1.0)
            fill!(b, 2.0)
            c = NDTensors.contract(a, (1, 2, -1), b, (-1, 3), (1, 2, 3))
            @test size(c) == (2, 3, 5)
            @test eltype(c) == Float64
            @test data(c) isa NDTensors.UnsafeArray
        end
    end

    @testset "Buffer-allocated permutedims" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            a = NDTensors.DenseTensor(Float64, buf, (2, 3))
            fill!(a, 1.0)
            a[1, 2] = 5.0
            ap = NDTensors.permutedims(a, (2, 1))
            @test size(ap) == (3, 2)
            @test data(ap) isa NDTensors.UnsafeArray
            @test ap[2, 1] ≈ 5.0
        end
    end

    @testset "Buffer-allocated element-wise operations" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            a = NDTensors.DenseTensor(Float64, buf, (2, 3))
            fill!(a, 2.0)
            b = a .* 3.0
            @test data(b) isa NDTensors.UnsafeArray
            @test b[1, 1] ≈ 6.0
            a .+= 1.0
            @test a[1, 1] ≈ 3.0
        end
    end

    @testset "Buffer-allocated norm and conj" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            a = NDTensors.DenseTensor(Float64, buf, (2, 3))
            fill!(a, 3.0)
            @test norm(a) ≈ sqrt(2 * 3 * 3.0^2)
            c = conj(a)
            @test data(c) isa NDTensors.UnsafeArray
            @test c[1, 1] ≈ 3.0
        end
    end

    @testset "SVD of buffer-allocated 2D tensor" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            a = NDTensors.DenseTensor(Float64, buf, (5, 3))
            fill!(a, 1.0)
            for i in 1:3
                a[i, i] = Float64(4 - i + 1)
            end
            U, S, V = svd(a)
            @test size(U) == (5, 3)
            @test size(S) == (3, 3)
            @test size(V) == (3, 3)
            @test eltype(U) == Float64
            @test eltype(V) == Float64
            # Buffer-backed outputs (decompositions_buffer.jl)
            @test NDTensors.data(NDTensors.storage(U)) isa NDTensors.UnsafeArray
            @test NDTensors.data(NDTensors.storage(V)) isa NDTensors.UnsafeArray
            @test NDTensors.data(NDTensors.storage(S)) isa NDTensors.UnsafeArray
            # Verify reconstruction: A = U * S * V' via tensor contractions
            US = contract(U, (1, -1), S, (-1, 2))
            @test NDTensors.data(NDTensors.storage(US)) isa NDTensors.UnsafeArray
            USV = contract(US, (1, -1), V, (2, -1))
            @test NDTensors.data(NDTensors.storage(USV)) isa NDTensors.UnsafeArray
            @test USV ≈ a
        end
    end

    @testset "n-D SVD of buffer-allocated tensor" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            a = NDTensors.DenseTensor(Float64, buf, (2, 3, 4))
            fill!(a, 1.0)
            a[1, 1, 1] = 5.0
            U, S, V, spec = svd(a, (1,), (2, 3))
            @test size(U) == (2, 2)
            @test size(V) == (3, 4, 2)
            @test eltype(U) == Float64
            @test eltype(V) == Float64
            @test NDTensors.data(NDTensors.storage(U)) isa NDTensors.UnsafeArray
            @test NDTensors.data(NDTensors.storage(V)) isa NDTensors.UnsafeArray
        end
    end

    @testset "QR of buffer-allocated 2D tensor" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            a = NDTensors.DenseTensor(Float64, buf, (4, 3))
            fill!(a, 1.0)
            a[1,1] = 3.0; a[2,2] = 4.0; a[3,3] = 5.0
            Q, R = qr(a)
            @test size(Q) == (4, 3)
            @test size(R) == (3, 3)
            @test eltype(Q) == Float64
            @test eltype(R) == Float64
            @test Array(Q) * Array(R) ≈ Array(a)
            @test NDTensors.data(NDTensors.storage(Q)) isa NDTensors.UnsafeArray
            @test NDTensors.data(NDTensors.storage(R)) isa NDTensors.UnsafeArray
        end
    end

    @testset "n-D QR of buffer-allocated tensor" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            a = NDTensors.DenseTensor(Float64, buf, (2, 3, 4))
            fill!(a, 1.0)
            Q, R = qr(a, (1,), (2, 3))
            @test size(Q) == (2, 2)
            @test size(R) == (2, 3, 4)
            @test eltype(Q) == Float64
            @test eltype(R) == Float64
            @test NDTensors.data(NDTensors.storage(Q)) isa NDTensors.UnsafeArray
            @test NDTensors.data(NDTensors.storage(R)) isa NDTensors.UnsafeArray
        end
    end

    @testset "conj of buffer-allocated Complex Dense" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            d = Dense{ComplexF64}(get_alloc_buffer(), 10)
            fill!(d, 1.0 + 2.0im)
            c = conj(d)
            @test data(c) isa NDTensors.UnsafeArray
            @test c[1] ≈ 1.0 - 2.0im
        end
    end

    @testset "real/imag of buffer-allocated Complex Dense" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            d = Dense{ComplexF64}(get_alloc_buffer(), 10)
            fill!(d, 1.0 + 2.0im)
            r = real(d)
            i = imag(d)
            @test data(r) isa NDTensors.UnsafeArray
            @test data(i) isa NDTensors.UnsafeArray
            @test r[1] ≈ 1.0
            @test i[1] ≈ 2.0
        end
    end

    @testset "complex from buffer-allocated Real Dense" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            d = Dense{Float64}(get_alloc_buffer(), 10)
            fill!(d, 3.0)
            c = complex(d)
            @test data(c) isa NDTensors.UnsafeArray{ComplexF64}
            @test c[1] ≈ 3.0 + 0.0im
        end
    end

    @testset "negation of buffer-allocated Dense" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            d = Dense{Float64}(get_alloc_buffer(), 10)
            fill!(d, 3.0)
            n = -d
            @test data(n) isa NDTensors.UnsafeArray
            @test n[1] ≈ -3.0
        end
    end

    @testset "scalar multiply of buffer-allocated Dense" begin
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            d = Dense{Float64}(get_alloc_buffer(), 10)
            fill!(d, 3.0)
            m1 = 2.0 * d
            @test data(m1) isa NDTensors.UnsafeArray
            @test m1[1] ≈ 6.0
            m2 = d * 3.0
            @test data(m2) isa NDTensors.UnsafeArray
            @test m2[1] ≈ 9.0
        end
    end

    @testset "element-wise ops fall back to heap without buffer" begin
        d = Dense{Float64}(10)
        fill!(d, 3.0)
        @test data(-d) isa Vector{Float64}
        @test data(2.0 * d) isa Vector{Float64}
    end

    @testset "to_buffer DenseTensor" begin
        T = NDTensors.DenseTensor(Float64, (3, 4))
        fill!(data(storage(T)), 1.5)
        buf = NDTensors.Bumper.SlabBuffer()
        Tb = NDTensors.to_buffer(T, buf)
        @test data(storage(Tb)) isa NDTensors.UnsafeArray
        @test data(storage(Tb)) == data(storage(T))
    end
end

nothing
end
