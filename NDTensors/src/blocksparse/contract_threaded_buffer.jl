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
#   - Float32/Float64 → direct TBLIS (zero alloc, no permutedims).
#   - ComplexF64      → 4-real TBLIS decomposition with pre-allocated
#                        buffer scratch (D, S per output block).
#   - Otherwise       → heap fallback (thread-safe, GC reclaims quickly).
#
# Each type family has its own function barrier to avoid Union types in the
# Folds.foreach closure (which prevents Julia from inlining the per-block call).

using .Expose: expose

# Module-level lazy cache for TBLIS extension (avoids dictionary lookup per block)
const _TBLIS_EXT_BUFFER = Ref{Union{Nothing, Module}}(nothing)

function _get_tblis_ext_buf()
    ext = _TBLIS_EXT_BUFFER[]
    if ext === nothing
        if isdefined(Base, :get_extension)
            ext = Base.get_extension(@__MODULE__, :NDTensorsTBLISExt)
        end
        _TBLIS_EXT_BUFFER[] = ext
    end
    return ext
end

# ===================================================================
# Float32 / Float64 — direct TBLIS, zero alloc
# ===================================================================

function _contract_blasreal!(R, labelsR, tensor1, labelstensor1,
                              tensor2, labelstensor2, grouped, executor)
    Folds.foreach(grouped.values, executor) do group
        β = zero(eltype(R))
        for bc in group
            blocktensor1, blocktensor2, blockR = bc
            α = compute_alpha(
                eltype(R), labelsR, blockR, inds(R),
                labelstensor1, blocktensor1, inds(tensor1),
                labelstensor2, blocktensor2, inds(tensor2)
            )
            contract!(Val(:TBLIS),
                R[blockR], labelsR,
                tensor1[blocktensor1], labelstensor1,
                tensor2[blocktensor2], labelstensor2,
                α, β)
            if iszero(β)
                β = one(eltype(R))
            end
        end
    end
    return nothing
end

# ===================================================================
# ComplexF64 — 4-real TBLIS decomposition with pre-allocated scratch
# ===================================================================

"""
    _block_contract_complex!(ext, R_block, labelsR, t1, labels1, t2, labels2, α, β, D, S)

Compute `R = α * T1(op) T2 + β * R` for ComplexF64 by decomposing into
real TBLIS contractions on zero-copy strided Float64 views of the complex data:

  1. Calls the extension's `tblis_compute_4real!(D, S, ...)` to compute
     D = Re(T1)∘Re(T2) - Im(T1)∘Im(T2)
     S = Re(T1)∘Im(T2) + Im(T1)∘Re(T2)
  2. Element-wise combines D, S into R with α, β.

D, S are pre-allocated Float64 arrays (from buffer or heap).
`ext` is the TBLIS extension module (loaded, checked at call site).
"""
function _block_contract_complex!(ext::Module, R_block, labelsR, t1_block, labels1,
                                   t2_block, labels2, α, β, D, S)
    ext.tblis_compute_4real!(D, S, t1_block, labels1, t2_block, labels2, labelsR)

    # Combine D, S into the complex output with α, β
    # all three arrays (D, S, aR_1d) share the same linear layout,
    # so 1D indexing avoids wrapping the UnsafeArray
    aR_1d = array(R_block)

    αr, αi = reim(α)
    βr, βi = reim(β)

    if iszero(βr) && iszero(βi)
        @inbounds for i in eachindex(D)
            d, s = D[i], S[i]
            aR_1d[i] = ComplexF64(αr * d - αi * s, αi * d + αr * s)
        end
    else
        @inbounds for i in eachindex(D)
            d, s = D[i], S[i]
            rr_old, ri_old = reim(aR_1d[i])
            aR_1d[i] = ComplexF64(
                αr * d - αi * s + βr * rr_old - βi * ri_old,
                αi * d + αr * s + βr * ri_old + βi * rr_old
            )
        end
    end
    return nothing
end

function _contract_complexf64!(R, labelsR, tensor1, labelstensor1,
                                tensor2, labelstensor2, grouped, executor,
                                ext::Module, scratch_map)
    Folds.foreach(grouped.values, executor) do group
        β = zero(eltype(R))
        scratch = scratch_map[last(first(group))]
        for bc in group
            blocktensor1, blocktensor2, blockR = bc
            α = compute_alpha(
                eltype(R), labelsR, blockR, inds(R),
                labelstensor1, blocktensor1, inds(tensor1),
                labelstensor2, blocktensor2, inds(tensor2)
            )
            _block_contract_complex!(ext,
                R[blockR], labelsR,
                tensor1[blocktensor1], labelstensor1,
                tensor2[blocktensor2], labelstensor2,
                α, β, scratch...)
            if iszero(β)
                β = one(eltype(R))
            end
        end
    end
    return nothing
end

# ===================================================================
# Fallback: heap via expose + @strided
# ===================================================================

function _contract_fallback!(R, labelsR, tensor1, labelstensor1,
                              tensor2, labelstensor2, grouped, executor)
    Folds.foreach(grouped.values, executor) do group
        β = zero(eltype(R))
        for bc in group
            blocktensor1, blocktensor2, blockR = bc
            α = compute_alpha(
                eltype(R), labelsR, blockR, inds(R),
                labelstensor1, blocktensor1, inds(tensor1),
                labelstensor2, blocktensor2, inds(tensor2)
            )
            R_block = R[blockR]
            t1_block = tensor1[blocktensor1]
            t2_block = tensor2[blocktensor2]
            # Copy buffer-backed block data to heap-backed tensors so
            # permutedims/similar inside contract! can allocate on heap
            # (the buffer is disabled to avoid multi-threaded alloc!).
            t1_h = tensor(
                Dense(Vector(data(storage(t1_block)))),
                inds(t1_block),
            )
            t2_h = tensor(
                Dense(Vector(data(storage(t2_block)))),
                inds(t2_block),
            )
            prev = get_alloc_buffer()
            set_alloc_buffer!(nothing)
            try
                contract!(expose(R_block), labelsR,
                          expose(t1_h), labelstensor1,
                          expose(t2_h), labelstensor2, α, β)
            finally
                set_alloc_buffer!(prev)
            end
            if iszero(β)
                β = one(eltype(R))
            end
        end
    end
    return nothing
end

# ===================================================================
# Pre-allocate complex scratch
# ===================================================================

"""
    _prealloc_complex_scratch(R, grouped)

Pre-allocate D, S Float64 scratch arrays for each output block.
Allocates from the active buffer (safe here because this runs
single-threaded, before @spawn).
"""
function _prealloc_complex_scratch(R::BlockSparseTensor{ElR, NR},
                                    grouped) where {ElR, NR}
    buf = get_alloc_buffer()
    scratch = Dict{Block{NR}, Tuple{Array{Float64}, Array{Float64}}}()
    for (blockR, group) in zip(keys(grouped), values(grouped))
        isempty(group) && continue
        R_block = R[blockR]
        sz = ntuple(d -> dim(R_block, d), Val(ndims(R_block)))
        D_1d = Bumper.alloc!(buf, Float64, prod(sz))
        S_1d = Bumper.alloc!(buf, Float64, prod(sz))
        D = Base.unsafe_wrap(Array{Float64}, pointer(D_1d), sz; own=false)
        S = Base.unsafe_wrap(Array{Float64}, pointer(S_1d), sz; own=false)
        scratch[blockR] = (D, S)
    end
    return scratch
end

# ===================================================================
# BlockSparse-level override
# ===================================================================

"""
    contract!(R::BlockSparseTensor{..., BlockSparse{..., UnsafeArray, ...}}, ...)

Override for buffer-backed BlockSparse contraction. Pre-allocates ComplexF64
scratch arrays from the buffer (single-threaded), then dispatches per-block
contractions through type-specific function barriers. Uses `Folds.foreach`
with `ThreadedEx`/`SequentialEx` parallel execution.
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

    # Group contractions by output block (same logic as contract_generic.jl)
    grouped = map(_ -> empty(contraction_plan), eachnzblock(R))
    for bc in contraction_plan
        push!(grouped[last(bc)], bc)
    end

    executor = (using_threaded_blocksparse() && Threads.nthreads() > 1) ?
               ThreadedEx() : SequentialEx()

    # Type-specific dispatch — compile-time via ElR
    if ElR <: LinearAlgebra.BlasReal
        if _get_tblis_ext_buf() === nothing
            _contract_fallback!(R, labelsR, tensor1, labelstensor1,
                                tensor2, labelstensor2, grouped, executor)
        else
            _contract_blasreal!(R, labelsR, tensor1, labelstensor1,
                                tensor2, labelstensor2, grouped, executor)
        end
    elseif ElR <: ComplexF64
        ext = _get_tblis_ext_buf()
        if ext === nothing
            _contract_fallback!(R, labelsR, tensor1, labelstensor1,
                                tensor2, labelstensor2, grouped, executor)
        else
            scratch_map = _prealloc_complex_scratch(R, grouped)
            _contract_complexf64!(R, labelsR, tensor1, labelstensor1,
                                  tensor2, labelstensor2, grouped, executor,
                                  ext, scratch_map)
        end
    else
        _contract_fallback!(R, labelsR, tensor1, labelstensor1,
                            tensor2, labelstensor2, grouped, executor)
    end
    return R
end
