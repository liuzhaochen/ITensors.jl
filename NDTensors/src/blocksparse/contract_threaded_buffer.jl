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

using .Expose: expose

# ===================================================================
# Main per-block dispatch
# ===================================================================

"""
    _block_contract!(R_block, labelsR, t1_block, labels1, t2_block, labels2, α, β,
                     scratch=nothing)

Dispatcher for per-block DenseTensor contraction:

- Float32/Float64 → `contract!(Val(:TBLIS), ...)` (direct TBLIS, zero alloc)
- ComplexF64      → 4-real TBLIS decomposition if `scratch=(D,S)` provided,
                     otherwise falls through to heap fallback.
- Other types     → clear buffer, fall back to `expose` + `@strided` (heap).
"""
function _block_contract!(R_block, labelsR, t1_block, labels1, t2_block, labels2, α, β,
                          scratch=nothing)
    ElR = eltype(R_block)

    _has_tblis = isdefined(Base, :get_extension) &&
                 Base.get_extension(@__MODULE__, :NDTensorsTBLISExt) !== nothing

    # --- ComplexF64: 4/2-real TBLIS with pre-allocated scratch ---
    if ElR <: ComplexF64 && scratch !== nothing
        _has_tblis || error(
            "TBLIS extension not loaded for buffer-threaded " *
            "BlockSparse ComplexF64 contraction. Use `using TBLIS`."
        )
        _block_contract_complex!(R_block, labelsR, t1_block, labels1,
                                 t2_block, labels2, α, β, scratch...)
        return
    end

    # --- BlasReal (Float32/Float64): direct TBLIS ---
    if ElR <: LinearAlgebra.BlasReal
        _has_tblis || error(
            "TBLIS extension not loaded for buffer-threaded " *
            "BlockSparse contraction. Use `using TBLIS`."
        )
        contract!(Val(:TBLIS),
            R_block, labelsR,
            t1_block, labels1,
            t2_block, labels2,
            α, β)
        return
    end

    # --- Fallback: heap via expose + @strided ---
    prev = get_alloc_buffer()
    set_alloc_buffer!(nothing)
    try
        contract!(expose(R_block), labelsR,
                  expose(t1_block), labels1,
                  expose(t2_block), labels2, α, β)
    finally
        set_alloc_buffer!(prev)
    end
    return nothing
end

# ===================================================================
# ComplexF64: 4-real TBLIS decomposition
# ===================================================================

"""
    _block_contract_complex!(R_block, labelsR, t1, labels1, t2, labels2, α, β, D, S)

Compute `R = α * T1(op) T2 + β * R` for ComplexF64 by decomposing into
4 real TBLIS contractions on zero-copy strided Float64 views of the
complex data:

  1. Calls the extension's `tblis_compute_4real!(D, S, ...)` to compute
     D = Re(T1)∘Re(T2) - Im(T1)∘Im(T2)
     S = Re(T1)∘Im(T2) + Im(T1)∘Re(T2)
  2. Element-wise combines D, S into R with α, β.

D, S are pre-allocated Float64 arrays (from buffer or heap).
"""
function _block_contract_complex!(R_block, labelsR, t1_block, labels1,
                                   t2_block, labels2, α, β, D, S)
    # Call the extension's 4-real computation
    ext = Base.get_extension(@__MODULE__, :NDTensorsTBLISExt)
    ext.tblis_compute_4real!(D, S, t1_block, labels1, t2_block, labels2, labelsR)

    # Combine D, S into the complex output with α, β
    # Rr_new = αr*D - αi*S + βr*Rr_old - βi*Ri_old
    # Ri_new = αi*D + αr*S + βr*Ri_old + βi*Rr_old
    N = ndims(R_block)
    aR = array(R_block)
    if ndims(aR) != N
        tdims = ntuple(d -> dim(R_block, d), Val(N))
        aR = Base.unsafe_wrap(Array{ComplexF64}, pointer(aR), tdims; own=false)
    end

    αr, αi = reim(α)
    βr, βi = reim(β)

    if iszero(βr) && iszero(βi)
        @inbounds for i in CartesianIndices(D)
            d, s = D[i], S[i]
            aR[i] = ComplexF64(αr * d - αi * s, αi * d + αr * s)
        end
    else
        @inbounds for i in CartesianIndices(D)
            d, s = D[i], S[i]
            rr_old, ri_old = reim(aR[i])
            aR[i] = ComplexF64(
                αr * d - αi * s + βr * rr_old - βi * ri_old,
                αi * d + αr * s + βr * ri_old + βi * rr_old
            )
        end
    end
    return nothing
end

# ===================================================================
# BlockSparse-level override
# ===================================================================

"""
    contract!(R::BlockSparseTensor{..., BlockSparse{..., UnsafeArray, ...}}, ...)

Override for buffer-backed BlockSparse contraction. Pre-allocates ComplexF64
scratch arrays from the buffer (single-threaded), then dispatches per-block
contractions through `_block_contract!` (direct TBLIS, zero-alloc). Uses
`Folds.foreach` with `ThreadedEx`/`SequentialEx` parallel execution.
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

    scratch_map = _prealloc_complex_scratch(R, grouped)
    executor = (using_threaded_blocksparse() && Threads.nthreads() > 1) ?
               ThreadedEx() : SequentialEx()

    Folds.foreach(values(grouped), executor) do group
        β = zero(eltype(R))
        first_contraction = first(group)
        scratch = scratch_map === nothing ? nothing :
                  get(scratch_map, last(first_contraction), nothing)
        for bc in group
            blocktensor1, blocktensor2, blockR = bc
            α = compute_alpha(
                eltype(R), labelsR, blockR, inds(R),
                labelstensor1, blocktensor1, inds(tensor1),
                labelstensor2, blocktensor2, inds(tensor2)
            )
            _block_contract!(
                R[blockR], labelsR,
                tensor1[blocktensor1], labelstensor1,
                tensor2[blocktensor2], labelstensor2,
                α, β, scratch,
            )
            if iszero(β)
                β = one(eltype(R))
            end
        end
    end
    return R
end

# ===================================================================
# Pre-allocate complex scratch
# ===================================================================

"""
    _prealloc_complex_scratch(R, grouped)

Pre-allocate D, S Float64 scratch arrays for each output block that needs
ComplexF64 4-real decomposition. Allocates from the active buffer (safe
here because this runs single-threaded, before @spawn).

Returns a Dict{Block{NR}, Tuple{Array{Float64}, Array{Float64}}} or
`nothing` if no scratch is needed.
"""
function _prealloc_complex_scratch(R::BlockSparseTensor{ElR}, grouped) where {ElR}
    ElR <: ComplexF64 || return nothing
    buf = get_alloc_buffer()
    buf === nothing && return nothing

    scratch = Dict{Any, Tuple{Any, Any}}()
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
    return length(scratch) > 0 ? scratch : nothing
end

