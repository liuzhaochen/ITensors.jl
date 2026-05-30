@eval module $(gensym())
using GPUArraysCore: @allowscalar
using NDTensors:
    NDTensors,
    Block,
    BlockOffsets,
    BlockSparse,
    BlockSparseTensor,
    Bumper,
    Combiner,
    Dense,
    DenseTensor,
    UnsafeArray,
    contract,
    data,
    dim,
    dims,
    nblocks,
    permutedims,
    storage,
    tensor,
    with_alloc_buffer
using Test: @test, @test_throws, @testset

# Testing generic block indices
struct Index{Space}
    space::Space
end
NDTensors.dim(i::Index) = sum(b -> last(b), i.space)
NDTensors.nblocks(i::Index) = length(i.space)
NDTensors.blockdim(i::Index, block::Integer) = last(i.space[block])
function NDTensors.outer(i1::Index, i2::Index)
    return Index(
        vec(
            map(Iterators.product(i1.space, i2.space)) do (b1, b2)
                return first(b1) + first(b2) => last(b1) * last(b2)
            end
        )
    )
end
NDTensors.permuteblocks(i::Index, perm::Vector{Int}) = Index(i.space[perm])

struct QN end
Base.:+(q1::QN, q2::QN) = QN()

@testset "CombinerTensor with buffer" begin
    @testset "Dense eltype: $elt" for elt in (Float64, Float32)
        d = 2
        input_tensor_inds = (d, d, d)
        combiner_tensor_inds = (d^2, d, d)
        output_tensor_inds = (d, d^2)

        ref_input = tensor(Dense(randn(elt, input_tensor_inds)), input_tensor_inds)
        ref_combiner = tensor(Combiner([1], [1]), combiner_tensor_inds)
        ref_output = contract(ref_input, (1, -1, -2), ref_combiner, (2, -1, -2))

        buf = Bumper.ResizeBuffer()
        with_alloc_buffer(buf) do
            input_tensor = tensor(Dense{elt}(buf, dim(input_tensor_inds)), input_tensor_inds)
            copyto!(data(storage(input_tensor)), data(storage(ref_input)))
            combiner_tensor = tensor(Combiner([1], [1]), combiner_tensor_inds)

            # Combine — result should be buffer-backed
            output_tensor = contract(input_tensor, (1, -1, -2), combiner_tensor, (2, -1, -2))
            @test output_tensor isa DenseTensor
            @test dims(output_tensor) == output_tensor_inds
            @test data(storage(output_tensor)) isa UnsafeArray
            @test data(storage(output_tensor)) ≈ data(storage(ref_output))

            # Uncombine — result should also be buffer-backed
            new_input_tensor = contract(output_tensor, (1, -1), combiner_tensor, (-1, 2, 3))
            @test data(storage(new_input_tensor)) isa UnsafeArray
            @test data(storage(new_input_tensor)) ≈ data(storage(ref_input))
        end

        # Invalid combining with buffer tensor
        buf2 = Bumper.ResizeBuffer()
        with_alloc_buffer(buf2) do
            input_tensor_1d = tensor(Dense{elt}(buf2, d), (d,))
            @test_throws Any contract(input_tensor_1d, (-1,), ref_combiner, (1, -1, -2))
        end
    end

    @testset "BlockSparse eltype: $elt" for elt in (Float64, Float32)
        ind_constructors = (dim -> [dim], dim -> Index([QN() => dim]))
        @testset "ind_constructor: $ic" for ic in ind_constructors
            d = 2
            i, j, k = map(ic, (d, d, d))
            c = ic(d^2)

            input_tensor_inds = (i, j, k)
            combiner_tensor_inds = (c, j, k)
            output_tensor_inds = (c, i)

            bo = BlockOffsets{3}([Block(1, 1, 1)], [0])
            input_data = randn(elt, dim(input_tensor_inds))

            # Reference: heap BlockSparse
            ref_input = tensor(BlockSparse(copy(input_data), bo), input_tensor_inds)
            ref_combiner = tensor(Combiner([1], [1]), combiner_tensor_inds)
            ref_output = contract(ref_input, (1, -1, -2), ref_combiner, (2, -1, -2))
            ref_output = permutedims(ref_output, (2, 1))

            # Buffer-allocated BlockSparse
            buf = Bumper.ResizeBuffer()
            with_alloc_buffer(buf) do
                input_tensor = tensor(
                    BlockSparse(elt, buf, bo, dim(input_tensor_inds)),
                    input_tensor_inds
                )
                copyto!(data(storage(input_tensor)), input_data)
                combiner_tensor = tensor(Combiner([1], [1]), combiner_tensor_inds)

                # Combine — result should be buffer-backed
                output_tensor = contract(input_tensor, (1, -1, -2), combiner_tensor, (2, -1, -2))
                @test output_tensor isa BlockSparseTensor
                @test dims(output_tensor) == dims(output_tensor_inds)
                @test data(storage(output_tensor)) isa UnsafeArray
                output_tensor = permutedims(output_tensor, (2, 1))

                @allowscalar for idx in 1:length(input_tensor)
                    @test input_tensor[idx] == output_tensor[idx]
                end

                # Uncombining — result should also be buffer-backed
                new_input_tensor = contract(output_tensor, (1, -1), combiner_tensor, (-1, 2, 3))
                new_input_tensor = permutedims(new_input_tensor, (3, 1, 2))
                @test data(storage(new_input_tensor)) isa UnsafeArray
                @test NDTensors.cpu(new_input_tensor) == NDTensors.cpu(input_tensor)
            end

            # Invalid combining with buffer BlockSparse
            buf2 = Bumper.ResizeBuffer()
            with_alloc_buffer(buf2) do
                invalid_input = tensor(
                    BlockSparse(elt, buf2, BlockOffsets{1}([Block(1)], [0]), dim(k)),
                    (k,)
                )
                @test_throws Any contract(invalid_input, (-1,), ref_combiner, (1, 2, -1))
            end
        end
    end
end
end
