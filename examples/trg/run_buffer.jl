# using Pkg
# Pkg.activate(".")
# # Ensure we use the local ITensors fork with NDTensors
# Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
# Pkg.develop(path=joinpath(@__DIR__, "..", "..", "NDTensors"))

# include(joinpath(@__DIR__, "..", "src", "trg.jl"))
# include(joinpath(@__DIR__, "..", "src", "2d_classical_ising.jl"))

using ITensors
using NDTensors: Bumper, with_alloc_buffer

"""
    trg_buffer(T; χmax, nsteps, ...)

Buffer-allocated version of TRG. The initial tensor T lives on heap.
Each TRG step wraps the factorizations and contractions in
`with_alloc_buffer`, so intermediate tensors are allocated from
the buffer arena. The result T is copied to heap at each step
boundary, then the buffer is reset.

This pattern — scope the heavy work, copy out results, then reset —
is the standard buffer usage model for iterative algorithms.
"""
function trg_buffer(
    T::ITensor;
    χmax::Int,
    nsteps::Int,
    cutoff=0.0,
    svd_alg="divide_and_conquer",
)
    sₕ, sᵥ = filterinds(T; plev=0)
    κ = 1.0
    buf = Bumper.ResizeBuffer()
    sp(x) = prime(x)

    for n in 1:nsteps
        # ── Buffer scope: all allocations go through Bumper ──
        T, sₕ, sᵥ = with_alloc_buffer(buf) do
            T = ITensors.to_buffer(T, buf)
            Fₕ, Fₕp = factorize(
                T, (sp(sₕ), sp(sᵥ));
                ortho="none", maxdim=χmax, cutoff, tags=tags(sₕ), svd_alg,
            )
            
            s̃ₕ = commonind(Fₕ, Fₕp)
            Fₕp *= δ(dag(s̃ₕ), sp(s̃ₕ))

            Fᵥ, Fᵥp = factorize(
                T, (sₕ, sp(sᵥ));
                ortho="none", maxdim=χmax, cutoff, tags=tags(sᵥ), svd_alg,
            )
            s̃ᵥ = commonind(Fᵥ, Fᵥp)
            Fᵥp *= δ(dag(s̃ᵥ), sp(s̃ᵥ))

            new_T = (Fₕ * δ(dag(sp(sₕ)), sₕ)) *
                    (Fᵥ * δ(dag(sp(sᵥ)), sᵥ)) *
                    (Fₕp * δ(dag(sₕ), sp(sₕ))) *
                    (Fᵥp * δ(dag(sᵥ), sp(sᵥ)))
            copy(new_T), s̃ₕ, s̃ᵥ
        end
        # ── End buffer scope ──

        # Normalization (on heap T, no buffer needed)
        trT = abs((T * δ(sₕ, sp(sₕ)) * δ(sᵥ, sp(sᵥ)))[])
        T = T / trT
        κ *= trT^(1 / 2^n)

        # Reclaim all temporary memory for next iteration
        Bumper.reset_buffer!(buf)
    end
    return κ, T
end

# ── Run ──
β = 1.1 * βc
d = 2
s = Index(d)
sₕ = addtags(s, "horiz")
sᵥ = addtags(s, "vert")

@show β

# Reference: original heap TRG
κ_ref = let
    # T = ising_mpo(sₕ, sᵥ, β)
    # a = @benchmark κ, _ = trg($T; χmax=20, nsteps=20)
    # @show a 
end

GC.gc()

# Buffer TRG
κ_buf = let
    # T = ising_mpo(sₕ, sᵥ, β)
    # trg_buffer(T; χmax=20, nsteps=2)
    # a = @benchmark κ, _ = trg_buffer($T; χmax=20, nsteps=20)
    # @show a 
end

κ_exact = exp(-β * ising_free_energy(β))

# println()
# println("Results:")
# println("  κ_ref  = $κ_ref  (exact = $κ_exact)")
# println("  κ_buf  = $κ_buf")
# println("  err_ref = $(abs(κ_ref - κ_exact))")
# println("  err_buf = $(abs(κ_buf - κ_exact))")
# println("  κ match: $(isapprox(κ_ref, κ_buf, rtol=1e-10))")
