# Benchmark: pre-permute vs no-pre-permute for repeated L * ψ * R contraction
# DMRG-style contraction: left environment × wavefunction × right environment
#
# Compares:
#   (a) to_buffer each iteration (re-permutes every time via internal permutedims)
#   (b) lanczos_permute once, then contract (trivA/trivB = zero internal perm)
#
# Usage:
#   julia -t 4 --project=benchmark benchmark/qn_lanczos_permute_benchmark.jl

using ITensors
using NDTensors: Bumper, with_alloc_buffer, to_buffer, copy
using ITensors: enable_threaded_blocksparse, disable_threaded_blocksparse
using LinearAlgebra, BenchmarkTools

BLAS.set_num_threads(1)
ITensors.NDTensors.Strided.disable_threads()

const NTHREADS = Threads.nthreads()
println("="^70)
println("DMRG Lanczos Permute Benchmark: L * ψ * R")
println("="^70)
println("Julia threads:  $NTHREADS")
println("BLAS threads:   $(BLAS.get_num_threads())")

# Tensor indices matching DMRG contraction pattern
const lqr = Index([QN("Sz", 5) => 3, QN("Sz", 3) => 8, QN("Sz", 1) => 17,
        QN("Sz", -1) => 15, QN("Sz", -3) => 7, QN("Sz", -5) => 2], "Link,qr")
const ll = Index([QN() => 1, QN("Sz", 0) => 6, QN("Sz", 2) => 6,
        QN("Sz", -2) => 6, QN("Sz", 0) => 1], "Link,l=32")
const s = Index([QN("Sz", 1) => 1, QN("Sz", -1) => 1], "S=1/2,Site,n=2")
const p1 = Index([QN("Sz", 6) => 1, QN("Sz", 4) => 5, QN("Sz", 2) => 12,
        QN("Sz", 0) => 16, QN("Sz", -2) => 12,
        QN("Sz", -4) => 5, QN("Sz", -6) => 1], "Link,qr")

# Left environment L, wavefunction ψ, right environment R
const L = random_itensor(ll, dag(s), prime(s), dag(lqr), prime(lqr))
const ψ = random_itensor(p1, lqr, s)
const R = random_itensor(dag(p1), dag(ll), prime(p1))

println("\nTensor structure:")
println("  L (5D): dims=$(dim(L)), nnz=$(nnz(L))")
println("  ψ (3D): dims=$(dim(ψ)), nnz=$(nnz(ψ))")
println("  R (3D): dims=$(dim(R)), nnz=$(nnz(R))")
println()

# ── Correctness ──
println("--- Correctness ---")
C_ref = L * ψ * R

# No-pre-permute: plain to_buffer each time, then contract via *
println("--- No-pre Contraction ---")
buf_c = Bumper.SlabBuffer{2^25}()
C_nopre = with_alloc_buffer(buf_c) do
    L_b = ITensors.to_buffer(L, buf_c)
    ψ_b = ITensors.to_buffer(ψ, buf_c)
    R_b = ITensors.to_buffer(R, buf_c)
    L_b * ψ_b * R_b
end
println("No-pre match: ", isapprox(C_ref, copy(C_nopre); atol=1e-10))

println("--- Pre Contraction ---")
# Pre-permute: lanczos_permute once, then contract via *
buf_p = Bumper.SlabBuffer{2^25}()
L_p, R_p, ψ_p = ITensors.lanczos_permute(L, R, ψ, buf_p)
C_pre = with_alloc_buffer(buf_p) do
    L_p * ψ_p * R_p
end
println("Pre-perm match: ", isapprox(C_ref, copy(C_pre); atol=1e-10))
# break
# ── Benchmark ──
println("\n--- Benchmark (10 contraction iterations) ---")

# No-pre: to_buffer(L, ψ, R) + L*ψ*R each iteration (fresh buf per iter)
function bench_no_pre()
    buf = Bumper.SlabBuffer{2^25}()
    L_b = ITensors.to_buffer(L, buf)
    ψ_b = ITensors.to_buffer(ψ, buf)
    R_b = ITensors.to_buffer(R, buf)
    @btime begin
        with_alloc_buffer($buf) do
            $L_b * $ψ_b * $R_b
        end
    end
    nothing
end

function bench_no_pre_thr()
    buf = Bumper.SlabBuffer{2^25}()
    L_b = ITensors.to_buffer(L, buf)
    ψ_b = ITensors.to_buffer(ψ, buf)
    R_b = ITensors.to_buffer(R, buf)
    enable_threaded_blocksparse()
    @btime begin
        with_alloc_buffer($buf) do
            $L_b * $ψ_b * $R_b
        end
    end
    disable_threaded_blocksparse()
    nothing
end

# Pre-permute: lanczos_permute once, then L*ψ*R 10x
function bench_pre()
    buf = Bumper.SlabBuffer{2^25}()
    L_p, R_p, ψ_p = ITensors.lanczos_permute(L, R, ψ, buf)
    a = with_alloc_buffer(buf) do
        a = noprime!(L_p * ψ_p * R_p)
    end
    @btime begin
        with_alloc_buffer($buf) do
            $L_p * $ ψ_p * $R_p
        end
    end
    @btime begin
        with_alloc_buffer($buf) do
            $L_p * $a * $R_p
        end
    end
    nothing
end

function bench_pre_thr()
    buf = Bumper.SlabBuffer{2^25}()
    L_p, R_p, ψ_p = ITensors.lanczos_permute(L, R, ψ, buf)
    a = with_alloc_buffer(buf) do
        a = noprime!(L_p * ψ_p * R_p)
    end
    enable_threaded_blocksparse()
    @btime begin
        with_alloc_buffer($buf) do
            $L_p * $ψ_p * $R_p
        end
    end
    @btime begin
        with_alloc_buffer($buf) do
            $L_p * $a * $R_p
        end
    end
    disable_threaded_blocksparse()
    nothing
end

# Warm-up
# bench_no_pre()
# bench_no_pre_thr()
# bench_pre()
# bench_pre_thr()

println("(each runs 10 contraction iterations, ~5 sec measurement)")
println()

for (name, fn) in [
    ("No pre-perm seq", bench_no_pre),
    ("No pre-perm thr", bench_no_pre_thr),
    ("Pre-permute seq", bench_pre),
    ("Pre-permute thr", bench_pre_thr),
]
    println("Benchmark: $name")
    b = fn()
    # show(stdout, "text/plain", b/10)
    println()
    println()
end
