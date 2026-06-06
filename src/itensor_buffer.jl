# [FORK] Buffer-backed ITensor conversion
# Only provides to_buffer for converting heap ITensors to buffer-backed.
# Direct ITensor(buf, inds) constructors removed — use to_buffer instead.

"""
    to_buffer(A::ITensor, buf::NDTensors.AllocBuffer)

Copy an ITensor (Dense, Diag, or BlockSparse) into a buffer-allocated one.
The new ITensor retains the same indices and storage structure, but the
data lives in the buffer arena.
"""
function to_buffer(A::ITensor, buf::NDTensors.AllocBuffer)
    T = tensor(A)
    Tb = NDTensors.to_buffer(T, buf)
    return ITensor(NDTensors.AllowAlias(), Tb)
end

"""
    _predict_output_order(LH, ψ, R)

Predict the output index order of `noprime!(LH * ψ * R)` using only Index
metadata — no data movement or contraction. This follows the same ordering
rule as `contract_labels` → `contract_inds`: T1's free indices (in T1's
order), then T2's free indices (in T2's order), stripped of prime levels.

Returns a tuple of Index objects matching what `inds(noprime!(LH*ψ*R))`
would produce.
"""
function _predict_output_order(LH::ITensor, ψ::ITensor, R::ITensor)
    LH_inds = Tuple(inds(LH))
    ψ_inds = Tuple(inds(ψ))
    R_inds = Tuple(inds(R))

    # Step 1: LH*ψ output = LH_free (LH order) ++ ψ_free (ψ order)
    LH_free = [i for i in LH_inds if !(i in ψ_inds)]
    ψ_free  = [i for i in ψ_inds  if !(i in LH_inds)]
    Lψ_inds = (LH_free..., ψ_free...)

    # Step 2: (Lψ)*R output = Lψ_free (Lψ order) ++ R_free (R order)
    Lψ_free = [i for i in Lψ_inds if !(i in R_inds)]
    R_free  = [i for i in R_inds  if !(i in Lψ_inds)]
    result  = (Lψ_free..., R_free...)

    # Step 3: Strip primes to match ψ's prime level (noprime! equivalent)
    return noprime(result)
end

"""
    lanczos_permute(LH, R, ψ, buf)

Copy LH, R, ψ from heap into buffer, with ψ pre-permuted to match the output
order of `noprime!(LH*ψ*R)`, and LH/R pre-permuted into GEMM-friendly layout
relative to the new ψ order. This ensures every iteration's `Hv = noprime!(LH*ψ*R)`
naturally has the same storage order as ψ, so L/R's pre-permutation stays valid
(LH: free_first, R: cont_first).

Returns (LH_b, R_b, ψ_b).
"""
function lanczos_permute(LH::ITensor, R::ITensor, ψ::ITensor,
                         buf::NDTensors.AllocBuffer)
    # ── Step 1: Align ψ to target order and copy to buffer ──
    target = _predict_output_order(LH, ψ, R)
    ψ_aligned = if Tuple(inds(ψ)) == target
        to_buffer(ψ, buf)
    else
        psi_new = NDTensors.with_alloc_buffer(buf) do
            permute(ψ, target...)
        end
        psi_new
    end

    # ── Step 2: LH: free_first order, contracted in ψ_aligned's order ──
    LH_inds = Tuple(inds(LH))
    ψ_inds = Tuple(inds(ψ_aligned))
    LH_free = filter(i -> !(i in ψ_inds), LH_inds)
    LH_cont = [LH_inds[findfirst(==(ci), LH_inds)] for ci in ψ_inds if ci in LH_inds]
    LH_target = (LH_free..., LH_cont...)
    LH_perm = ntuple(length(LH_target)) do i
        findfirst(==(LH_target[i]), LH_inds)
    end

    # ── Step 3: R: cont_first order, contracted in int_inds order ──
    ψ_unique = filter(i -> !(i in LH_inds), ψ_inds)
    int_inds = (LH_free..., ψ_unique...)
    R_inds = Tuple(inds(R))
    R_cont_parts = [R_inds[findfirst(==(ci), R_inds)] for ci in int_inds if ci in R_inds]
    R_cont = (R_cont_parts...,)
    R_free = filter(i -> !(i in int_inds), R_inds)
    R_target = (R_cont..., R_free...)
    R_perm = ntuple(length(R_target)) do i
        findfirst(==(R_target[i]), R_inds)
    end

    # ── Step 4: Copy LH, R to buffer with perm ──
    LH_nd = tensor(LH)
    R_nd = tensor(R)
    LH_b = if NDTensors.storagetype(LH_nd) <: NDTensors.BlockSparse
        NDTensors.to_buffer(LH_nd, LH_perm, buf)
    else
        NDTensors.to_buffer(LH_nd, buf)
    end
    R_b = if NDTensors.storagetype(R_nd) <: NDTensors.BlockSparse
        NDTensors.to_buffer(R_nd, R_perm, buf)
    else
        NDTensors.to_buffer(R_nd, buf)
    end
    return ITensor(NDTensors.AllowAlias(), LH_b),
           ITensor(NDTensors.AllowAlias(), R_b),
           ψ_aligned
end
