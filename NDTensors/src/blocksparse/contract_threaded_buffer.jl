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
# Fallback: direct BLAS gemm with pre-allocated buffer caches
# ===================================================================

# (no separate pre-alloc function — allocation is per-BC inside _contract_fallback!)

"""
    _contract_perm(labels, labelsR, order)

Compute dimension permutation that groups free and contracted indices.
- `order = :free_first` → (free indices..., contracted indices...)
- `order = :contracted_first` → (contracted indices..., free indices...)
"""
function _contract_perm(labels, labelsR, order)
    free = Int[]
    cont = Int[]
    for (i, l) in enumerate(labels)
        if l in labelsR
            push!(free, i)
        else
            push!(cont, i)
        end
    end
    return order == :free_first ? vcat(free, cont) : vcat(cont, free)
end

"""
    _permuted_copyto!(dest, src::DenseTensor, perm)

Copy N-dimensional DenseTensor `src` into 1D `dest` with dimension permutation.
After copy, `dest` contains the data as if `permutedims(src, perm)` were
flattened to 1D in column-major order. No temporary allocation.
"""
function _permuted_copyto!(dest, src::DenseTensor, perm)
    N = length(perm)
    src_sz = ntuple(i -> size(src, i), N)
    dest_sz = ntuple(i -> src_sz[perm[i]], N)

    dest_strides = ntuple(Val(N)) do i
        i == 1 ? 1 : prod(dest_sz[1:(i - 1)])
    end

    @inbounds for src_idx in CartesianIndices(src_sz)
        dst_lin = 1
        for d in 1:N
            dst_lin += (src_idx.I[perm[d]] - 1) * dest_strides[d]
        end
        dest[dst_lin] = src[src_idx]
    end
    return dest
end

# Trivial permutation check (avoid importing NDTensors.is_trivial_permutation)
function _is_trivial_perm(p, n)
    for i in 1:n
        p[i] != i && return false
    end
    return true
end

"""
    _scratch_to_c!(R_data, R_work, c_perm, c_sz, gemm_sz)

Copy BLAS gemm result from `R_work` (column-major in GEMM order) to
`R_data` (column-major in C's natural order), applying the dimension
permutation `c_perm`. Both are flat 1D arrays.

`c_perm[d]` = position in GEMM order of C's d-th dimension. `gemm_sz` = dimension sizes in GEMM order.
"""
function _scratch_to_c!(R_data, R_work, c_perm, c_sz, gemm_sz, accumulate::Bool=false)
    N = length(c_sz)
    g_stride = ntuple(Val(N)) do i
        i == 1 ? 1 : prod(gemm_sz[1:(i - 1)])
    end
    r_stride = ntuple(Val(N)) do i
        i == 1 ? 1 : prod(c_sz[1:(i - 1)])
    end
    if accumulate
        @inbounds for c_idx in CartesianIndices(c_sz)
            w_lin = 1
            r_lin = 1
            for d in 1:N
                w_lin += (c_idx.I[d] - 1) * g_stride[c_perm[d]]
                r_lin += (c_idx.I[d] - 1) * r_stride[d]
            end
            R_data[r_lin] += R_work[w_lin]
        end
    else
        @inbounds for c_idx in CartesianIndices(c_sz)
            w_lin = 1
            r_lin = 1
            for d in 1:N
                w_lin += (c_idx.I[d] - 1) * g_stride[c_perm[d]]
                r_lin += (c_idx.I[d] - 1) * r_stride[d]
            end
            R_data[r_lin] = R_work[w_lin]
        end
    end
    return R_data
end

"""
    _contract_fallback!(R, labelsR, tensor1, labelstensor1, ...)

Per-block direct BLAS gemm using libc-malloc'd temp caches.
Each BC: ccall(:malloc) → copy block data with permutation →
BLAS.gemm! into work scratch → transfer to R_block → ccall(:free).
No Bumper, no GC pressure.
"""
function _contract_fallback!(R, labelsR, tensor1, labelstensor1,
                              tensor2, labelstensor2, grouped, executor)
    ElR = eltype(R)

    # Hoist: dimension counts only depend on labels (same for all BCs)
    ndim1, ndim2 = length(labelstensor1), length(labelstensor2)
    nfreeA = count(l -> l in labelsR, labelstensor1)
    nfreeB = count(l -> l in labelsR, labelstensor2)
    ncontA, ncontB = ndim1 - nfreeA, ndim2 - nfreeB
    nRlabels = length(labelsR)

    # Permutations: A = (free, cont), B = (cont, free)
    permA = _contract_perm(labelstensor1, labelsR, :free_first)
    permB = _contract_perm(labelstensor2, labelsR, :contracted_first)

    # Align contracted index ordering between A and B for GEMM consistency.
    # BLAS.gemm! sums over the middle dimension by linear index, so A's
    # contracted indices must appear in the same order as B's contracted
    # indices. The upstream ContractionProperties machinery handles this
    # via PA/PB construction; here we reorder permA's contracted portion
    # to match B's order.
    if ncontA > 1
        contA = [labelstensor1[permA[nfreeA + i]] for i in 1:ncontA]
        contB = [labelstensor2[permB[i]] for i in 1:ncontB]
        if contA != contB
            permA = vcat(permA[1:nfreeA],
                         [findfirst(==(l), labelstensor1) for l in contB])
        end
    end

    trivA = _is_trivial_perm(permA, ndim1)
    trivB = _is_trivial_perm(permB, ndim2)
    permA_tup, permB_tup = Tuple(permA), Tuple(permB)

    # For output: R_work order = (A_free..., B_free...)  vs  labelsR order
    r_work_labels = Int[]
    sizehint!(r_work_labels, nRlabels)
    for i in 1:nfreeA; push!(r_work_labels, labelstensor1[permA[i]]); end
    for i in 1:nfreeB; push!(r_work_labels, labelstensor2[permB[ncontB + i]]); end
    need_c_perm = any(i -> labelsR[i] != r_work_labels[i], 1:nRlabels)
    c_perm_tup = need_c_perm ? Tuple(findfirst(==(l), r_work_labels) for l in labelsR) : nothing

    Folds.foreach(grouped.values, executor) do group
        βflag = true
        for bc in group
            blocktensor1, blocktensor2, blockR = bc
            α = compute_alpha(
                ElR, labelsR, blockR, inds(R),
                labelstensor1, blocktensor1, inds(tensor1),
                labelstensor2, blocktensor2, inds(tensor2)
            )
            R_block = R[blockR]
            t1_block = tensor1[blocktensor1]
            t2_block = tensor2[blocktensor2]

            # GEMM dimensions
            dleft = prod(ntuple(i -> size(t1_block, permA[i]), nfreeA))
            dmid  = prod(ntuple(i -> size(t1_block, permA[nfreeA + i]), ncontA))
            dright = prod(ntuple(i -> size(t2_block, permB[ncontB + i]), nfreeB))
            nA, nB, nR = dleft * dmid, dmid * dright, dleft * dright

            # Allocate scratch (per BC: malloc/use/free, thread-safe)
            A_ptr = ccall(:malloc, Ptr{Cvoid}, (Csize_t,), sizeof(ElR)*nA)
            B_ptr = ccall(:malloc, Ptr{Cvoid}, (Csize_t,), sizeof(ElR)*nB)
            R_ptr = ccall(:malloc, Ptr{Cvoid}, (Csize_t,), sizeof(ElR)*nR)

            A_buf = Base.unsafe_wrap(Array{ElR}, Ptr{ElR}(A_ptr), nA; own=false)
            B_buf = Base.unsafe_wrap(Array{ElR}, Ptr{ElR}(B_ptr), nB; own=false)
            R_buf = Base.unsafe_wrap(Array{ElR}, Ptr{ElR}(R_ptr), nR; own=false)

            # --- Input A: permute + reshape for GEMM ---
            if trivA
                copyto!(A_buf, vec(array(t1_block)))
            else
                perm_sz = ntuple(d -> size(t1_block, permA[d]), Val(ndim1))
                A_nd = reshape(A_buf, perm_sz)
                permutedims!(A_nd, array(t1_block), permA_tup)
            end
            A_2d = reshape(A_buf, (dleft, dmid))

            # --- Input B: permute + reshape for GEMM ---
            if trivB
                copyto!(B_buf, vec(array(t2_block)))
            else
                perm_sz = ntuple(d -> size(t2_block, permB[d]), Val(ndim2))
                B_nd = reshape(B_buf, perm_sz)
                permutedims!(B_nd, array(t2_block), permB_tup)
            end
            B_2d = reshape(B_buf, (dmid, dright))

            # --- GEMM ---
            R_2d = reshape(R_buf, (dleft, dright))
            BLAS.gemm!('N', 'N', α, A_2d, B_2d, zero(ElR), R_2d)

            # --- Output: copy R_buf → R_block (with perm if needed) ---
            R_data = vec(array(R_block))
            if need_c_perm
                gemm_sz = ntuple(Val(nRlabels)) do i
                    i <= nfreeA ? size(t1_block, permA[i]) :
                                  size(t2_block, permB[ncontB + i - nfreeA])
                end
                R_work_nd = reshape(R_buf, gemm_sz)
                R_block_nd = reshape(R_data, ntuple(i -> size(R_block, i), Val(nRlabels)))
                if βflag
                    permutedims!(R_block_nd, R_work_nd, c_perm_tup)
                else
                    X_ptr = ccall(:malloc, Ptr{Cvoid}, (Csize_t,), sizeof(ElR)*nR)
                    R_tmp = Base.unsafe_wrap(Array{ElR}, Ptr{ElR}(X_ptr), nR; own=false)
                    R_tmp_nd = reshape(R_tmp, gemm_sz)
                    permutedims!(R_tmp_nd, R_work_nd, c_perm_tup)
                    for i in eachindex(R_data)
                        R_data[i] += R_tmp[i]
                    end
                    ccall(:free, Cvoid, (Ptr{Cvoid},), X_ptr)
                end
            elseif βflag
                copyto!(R_data, R_buf)
            else
                for i in eachindex(R_buf)
                    R_data[i] += R_buf[i]
                end
            end
            # Free scratch
            ccall(:free, Cvoid, (Ptr{Cvoid},), A_ptr)
            ccall(:free, Cvoid, (Ptr{Cvoid},), B_ptr)
            ccall(:free, Cvoid, (Ptr{Cvoid},), R_ptr)
            βflag = false
        end  # for bc in group
    end  # Folds.foreach
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
