# [FORK] Thread-safe BlockSparse contraction for buffer-backed tensors
#
# Standard threaded contraction (contract_threaded.jl) uses Folds.foreach +
# ThreadedEx(). Inside the parallel section, each block calls:
#
#   contract!(expose(R[blockR]), ..., expose(t1[block]), ..., expose(t2[block]), ...)
#
# which goes to DenseTensor _contract! → permutedims → similar → Bumper.alloc!.
# Multiple threads calling Bumper.alloc! simultaneously corrupts the arena.
#
# This override replaces the per-block contract! with a buffer-safe wrapper:
#   - When TBLIS is available: TBLIS.mul! handles arbitrary-order tensors
#     natively, avoiding permutedims entirely → zero buffer allocation.
#   - Otherwise: temporarily clears the task-local buffer so permutedims
#     allocates on the heap (thread-safe, GC reclaims immediately).
#
# Regular (non-BlockSparse) DenseTensor contractions are NOT affected —
# this override only activates for BlockSparse output tensors with
# UnsafeArray storage when threading is enabled.

using .Expose: expose

"""
    _block_contract!(R_block, labelsR, t1_block, labels1, t2_block, labels2, α, β)

Buffer-safe per-block DenseTensor contraction. Tries TBLIS first (zero
buffer allocation, works for BlasReal element types when TBLIS.jl is
installed). Falls back to clearing the task-local buffer so permutedims
allocates on the heap (thread-safe, GC reclaims immediately).
"""
function _block_contract!(R_block, labelsR, t1_block, labels1, t2_block, labels2, α, β)
    # Try TBLIS: no permutedims, no buffer allocation.
    # The NDTensorsTBLISExt extension auto-loads when TBLIS is installed
    # and contract!(Val(:TBLIS), ...) is first called. If TBLIS is not
    # installed or types don't match, MethodError is caught below.
    if eltype(R_block) <: LinearAlgebra.BlasReal
        try
            contract!(Val(:TBLIS), R_block, labelsR, t1_block, labels1, t2_block, labels2, α, β)
            return
        catch e
            e isa MethodError || rethrow()
        end
    end
    # Fallback: clear buffer so similar → Vector (heap, thread-safe)
    prev = get_alloc_buffer()
    set_alloc_buffer!(nothing)
    try
        contract!(expose(R_block), labelsR, expose(t1_block), labels1, expose(t2_block), labels2, α, β)
    finally
        set_alloc_buffer!(prev)
    end
    return nothing
end

"""
    contract!(R::BlockSparseTensor{<:Any, <:Any, <:BlockSparse{<:Any, <:UnsafeArray}}, ...)

Override for buffer-backed BlockSparse contraction. When threading is
enabled, uses _block_contract! (buffer-safe) inside a custom parallel
loop. When single-threaded, delegates to the generic SequentialEx path.
"""
function contract!(
    R::BlockSparseTensor{ElR, NR, <:Any, <:BlockSparse{ElR, <:UnsafeArray{ElR, 1}, NR}},
    labelsR,
    tensor1::BlockSparseTensor,
    labelstensor1,
    tensor2::BlockSparseTensor,
    labelstensor2,
    contraction_plan
) where {ElR, NR}
    if isempty(contraction_plan)
        return R
    end

    if using_threaded_blocksparse() && Threads.nthreads() > 1
        # Group contractions by output block (same logic as contract_generic.jl)
        grouped = map(_ -> empty(contraction_plan), eachnzblock(R))
        for bc in contraction_plan
            push!(grouped[last(bc)], bc)
        end
        _contract_buffer_threaded!(
            R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, grouped
        )
    else
        # Single-threaded: use the standard sequential executor (safe)
        contract!(
            R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2,
            contraction_plan, SequentialEx()
        )
    end
    return R
end

"""
    _contract_buffer_threaded!(R, labelsR, t1, lt1, t2, lt2, grouped)

Execute grouped block contractions in parallel. Each spawned task uses
_block_contract! which is safe for buffer-backed tensors (TBLIS or heap
fallback, never allocates from the shared buffer).
"""
function _contract_buffer_threaded!(
    R::BlockSparseTensor,
    labelsR,
    tensor1::BlockSparseTensor,
    labelstensor1,
    tensor2::BlockSparseTensor,
    labelstensor2,
    grouped
)
    n = Threads.nthreads()
    n_groups = length(grouped)
    n_groups == 0 && return nothing

    # Chunk the groups across threads
    group_vals = values(grouped)
    chunk_size = cld(n_groups, n)
    chunks = [
        group_vals[i:min(i + chunk_size - 1, n_groups)]
        for i in 1:chunk_size:n_groups
    ]

    tasks = map(chunks) do chunk
        Threads.@spawn _execute_chunk!(
            R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, chunk
        )
    end

    for t in tasks
        fetch(t)
    end
    return nothing
end

"""
    _execute_chunk!(R, labelsR, t1, lt1, t2, lt2, groups)

Process a chunk of contraction groups sequentially on one task.
Each group reduces into one output block (β-flip).
"""
function _execute_chunk!(
    R::BlockSparseTensor,
    labelsR,
    tensor1::BlockSparseTensor,
    labelstensor1,
    tensor2::BlockSparseTensor,
    labelstensor2,
    groups
)
    for contraction_plan_group in groups
        β = zero(eltype(R))
        for block_contraction in contraction_plan_group
            blocktensor1, blocktensor2, blockR = block_contraction

            α = compute_alpha(
                eltype(R), labelsR, blockR, inds(R),
                labelstensor1, blocktensor1, inds(tensor1),
                labelstensor2, blocktensor2, inds(tensor2)
            )

            _block_contract!(
                R[blockR], labelsR,
                tensor1[blocktensor1], labelstensor1,
                tensor2[blocktensor2], labelstensor2,
                α, β
            )

            if iszero(β)
                β = one(eltype(R))
            end
        end
    end
    return nothing
end
