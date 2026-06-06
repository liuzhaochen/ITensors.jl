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
    lanczos_permute(LH, R, ψ, buf)

Copy LH, R, ψ from heap into buffer, with LH and R pre-permuted into
GEMM-friendly layout so that subsequent `LH * ψ * R` contractions go
through the zero-copy path (trivA, trivB).

LH gets `free_first` order: (LH_unique..., LH_contracted...).
R gets `cont_first` order: (R_contracted..., R_free...).

Returns (LH_b, R_b, ψ_b) — all buffer-backed. LH_b and R_b should be
kept across Lanczos iterations; only ψ_b needs replacing each iteration.
Use Bumper.checkpoint_save/restore to manage the buffer region after LH,R.
"""
function lanczos_permute(LH::ITensor, R::ITensor, ψ::ITensor,
                         buf::NDTensors.AllocBuffer)
    # ── LH: free_first order ──
    LH_inds = Tuple(inds(LH))
    ψ_inds = Tuple(inds(ψ))
    LH_free = filter(i -> !(i in ψ_inds), LH_inds)
    LH_cont = filter(i -> i in ψ_inds, LH_inds)
    LH_target = (LH_free..., LH_cont...)
    LH_perm = ntuple(length(LH_inds)) do d
        findfirst(==(LH_inds[d]), LH_target)
    end

    # ── R: cont_first order ──
    ψ_unique = filter(i -> !(i in LH_inds), ψ_inds)
    int_inds = (LH_free..., ψ_unique...)
    R_inds = Tuple(inds(R))
    R_cont = filter(i -> i in int_inds, R_inds)
    R_free = filter(i -> !(i in int_inds), R_inds)
    R_target = (R_cont..., R_free...)
    R_perm = ntuple(length(R_inds)) do d
        findfirst(==(R_inds[d]), R_target)
    end

    # ── Apply buffer-permuted copy ──
    LH_nd = tensor(LH)
    R_nd = tensor(R)
    ψ_nd = tensor(ψ)
    LH_b = NDTensors.to_buffer(LH_nd, LH_perm, buf)
    R_b = NDTensors.to_buffer(R_nd, R_perm, buf)
    ψ_b = NDTensors.to_buffer(ψ_nd, buf)
    return ITensor(NDTensors.AllowAlias(), LH_b),
           ITensor(NDTensors.AllowAlias(), R_b),
           ITensor(NDTensors.AllowAlias(), ψ_b)
end
