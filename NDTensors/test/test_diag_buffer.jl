@eval module $(gensym())
using LinearAlgebra: norm
using NDTensors: NDTensors, Bumper, Dense, Diag, NonuniformDiag, Tensor, conj, data,
    diaglength, storage, tensor, to_buffer, with_alloc_buffer
using Test: @test, @test_throws, @testset

@testset "Diag buffer" begin
    @testset "Diag{ElT}(buf, dim)" begin
        buf = Bumper.SlabBuffer()
        d = Diag{Float64}(buf, 5)
        @test eltype(d) == Float64
        @test length(d) == 5
        @test data(d) isa NDTensors.UnsafeArray
    end

    @testset "Diag{ComplexF64}(buf, dim)" begin
        buf = Bumper.SlabBuffer()
        d = Diag{ComplexF64}(buf, 3)
        @test eltype(d) == ComplexF64
        @test length(d) == 3
        @test data(d) isa NDTensors.UnsafeArray
    end

    @testset "Diag{ComplexF32}(buf, dim)" begin
        buf = Bumper.SlabBuffer()
        d = Diag{ComplexF32}(buf, 3)
        @test eltype(d) == ComplexF32
        @test length(d) == 3
        @test data(d) isa NDTensors.UnsafeArray
    end

    @testset "promote_rule inside NDTensors" begin
        result = NDTensors.eval(quote
            UA = NDTensors.UnsafeArray{Float64, 1}
            r1 = promote_type(Dense{Float64, UA}, Diag{Float64, Float64})
            r2 = promote_type(Diag{Float64, Float64}, Dense{Float64, UA})
            isconcretetype(r1) && isconcretetype(r2) && r1 == r2
        end)
        @test result
    end

    @testset "AllocBuffer dispatch for Diag" begin
        buf = Bumper.SlabBuffer()
        d = Diag{Float64}(buf, 4)
        @test data(d) isa NDTensors.UnsafeArray
    end

    @testset "norm of buffer-backed DiagTensor" begin
        buf = Bumper.SlabBuffer()
        d = Diag{Float64}(buf, 100)
        fill!(data(d), 1.0)
        T = tensor(d, (100, 100))
        @test norm(T) ≈ sqrt(100)
    end

    @testset "conj of buffer-backed DiagTensor" begin
        buf = Bumper.SlabBuffer()
        d = Diag{ComplexF64}(buf, 10)
        fill!(data(d), 1.0 + 2.0im)
        T = tensor(d, (10, 10))
        Tc = conj(T)
        @test typeof(storage(Tc)) <: Diag
        @test data(storage(Tc)) ≈ conj(data(d))
    end

    @testset "map_diag on buffer-backed DiagTensor" begin
        buf = Bumper.SlabBuffer()
        d = Diag{Float64}(buf, 10)
        fill!(data(d), 2.0)
        D = tensor(d, (10, 10))
        S = NDTensors.map_diag(i -> 3 * i, D)
        @test ndims(S) == ndims(D)
        for i in 1:diaglength(S)
            @test S[i, i] ≈ 6.0
        end
    end

    @testset "scalar divide buffer-backed Diag" begin
        buf = Bumper.ResizeBuffer()
        with_alloc_buffer(buf) do
            d = Diag{Float64}(buf, 10)
            fill!(data(d), 6.0)
            r = d / 2.0
            @test data(r) isa NDTensors.UnsafeArray
            @test r[1] ≈ 3.0
        end
    end

    @testset "to_buffer DiagTensor" begin
        d = Diag(ones(5))
        T = tensor(d, (5, 5))
        buf = NDTensors.Bumper.SlabBuffer()
        Tb = to_buffer(T, buf)
        @test data(NDTensors.storage(Tb)) isa NDTensors.UnsafeArray
        @test data(NDTensors.storage(Tb)) == data(NDTensors.storage(T))
    end
end

nothing
end
