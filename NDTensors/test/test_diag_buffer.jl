@eval module $(gensym())
using NDTensors: NDTensors, Bumper, Dense, Diag, tensor, data
using Test: @test, @test_throws, @testset

@testset "Diag buffer" begin
    @testset "Diag{ElT}(buf, dim)" begin
        buf = Bumper.SlabBuffer()
        d = Diag{Float64}(buf, 5)
        @test eltype(d) == Float64
        @test length(d) == 5
        @test data(d) isa Bumper.UnsafeArrays.UnsafeArray
    end

    @testset "Diag{ComplexF64}(buf, dim)" begin
        buf = Bumper.SlabBuffer()
        d = Diag{ComplexF64}(buf, 3)
        @test eltype(d) == ComplexF64
        @test length(d) == 3
    end

    @testset "promote_rule inside NDTensors" begin
        # Verify that promote_type for Dense{UnsafeArray} * UniformDiag
        # gives a concrete type (not abstract DenseVector/TensorStorage)
        result = NDTensors.eval(quote
            UA = Bumper.UnsafeArrays.UnsafeArray{Float64, 1}
            r1 = promote_type(Dense{Float64, UA}, Diag{Float64, Float64})
            r2 = promote_type(Diag{Float64, Float64}, Dense{Float64, UA})
            isconcretetype(r1) && isconcretetype(r2) && r1 == r2
        end)
        @test result
    end

    @testset "AllocBuffer dispatch for Diag" begin
        buf = Bumper.SlabBuffer()
        d = Diag{Float64}(buf, 4)
        @test data(d) isa Bumper.UnsafeArrays.UnsafeArray
    end
end

nothing
end
