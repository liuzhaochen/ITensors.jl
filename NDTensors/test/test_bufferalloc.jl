@eval module $(gensym())
using NDTensors
using NDTensors: get_alloc_buffer, set_alloc_buffer!, with_alloc_buffer
using Test: @test, @test_throws, @testset

@testset "bufferalloc" begin
    @testset "No buffer by default" begin
        @test get_alloc_buffer() === nothing
    end

    @testset "set and get alloc buffer" begin
        buf = NDTensors.Bumper.SlabBuffer()
        prev = get_alloc_buffer()
        set_alloc_buffer!(buf)
        @test get_alloc_buffer() === buf
        set_alloc_buffer!(prev)
        @test get_alloc_buffer() === nothing
    end

    @testset "with_alloc_buffer sets and restores" begin
        @test get_alloc_buffer() === nothing
        buf = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(buf) do
            @test get_alloc_buffer() === buf
        end
        @test get_alloc_buffer() === nothing
    end

    @testset "with_alloc_buffer restores on exception" begin
        buf = NDTensors.Bumper.SlabBuffer()
        @test_throws ErrorException with_alloc_buffer(buf) do
            error("boom")
        end
        @test get_alloc_buffer() === nothing
    end

    @testset "with_alloc_buffer nested" begin
        outer = NDTensors.Bumper.SlabBuffer()
        inner = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(outer) do
            @test get_alloc_buffer() === outer
            with_alloc_buffer(inner) do
                @test get_alloc_buffer() === inner
            end
            @test get_alloc_buffer() === outer
        end
        @test get_alloc_buffer() === nothing
    end

    @testset "with_alloc_buffer preserves previous buffer" begin
        prev = NDTensors.Bumper.SlabBuffer()
        buf = NDTensors.Bumper.SlabBuffer()
        set_alloc_buffer!(prev)
        with_alloc_buffer(buf) do
            @test get_alloc_buffer() === buf
        end
        @test get_alloc_buffer() === prev
        set_alloc_buffer!(nothing)
    end

    @testset "set_alloc_buffer! nothing clears" begin
        buf = NDTensors.Bumper.SlabBuffer()
        set_alloc_buffer!(buf)
        @test get_alloc_buffer() === buf
        set_alloc_buffer!(nothing)
        @test get_alloc_buffer() === nothing
    end

    @testset "Multiple different buffer types" begin
        slab = NDTensors.Bumper.SlabBuffer()
        with_alloc_buffer(slab) do
            @test get_alloc_buffer() isa NDTensors.Bumper.SlabBufferImpl.SlabBuffer
        end
        alloc = NDTensors.Bumper.AllocBuffer()
        with_alloc_buffer(alloc) do
            @test get_alloc_buffer() isa NDTensors.Bumper.AllocBufferImpl.AllocBuffer
        end
    end
end

nothing
end
