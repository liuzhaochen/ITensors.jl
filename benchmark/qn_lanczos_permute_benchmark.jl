# Benchmark: Pre-permute vs no-pre-permute for repeated contraction
# Uses your real DMRG tensor structure (L*phi) × R with labels (3,-1,4,-2)×(-2,-1,5)
# Compares:
#   (a) to_buffer each iteration (re-permutes every time)
#   (b) to_buffer_permuted once, then contract (zero internal perm)
#
# Usage:
#   julia -t 4 --project=benchmark benchmark/qn_lanczos_permute_benchmark.jl

using ITensors
using NDTensors: Bumper, with_alloc_buffer, to_buffer, contract, dense, copy, permutedims
using ITensors: enable_threaded_blocksparse, disable_threaded_blocksparse
using LinearAlgebra, BenchmarkTools

BLAS.set_num_threads(1)
ITensors.NDTensors.Strided.disable_threads()

const NTHREADS = Threads.nthreads()
println("="^70)
println("DMRG Contraction Permute Benchmark")
println("="^70)
println("Julia threads:  $NTHREADS")
println("BLAS threads:   $(BLAS.get_num_threads())")

# Your exact tensor structure from DMRG
const lqr = Index([QN("Sz",5)=>3, QN("Sz",3)=>8, QN("Sz",1)=>17,
                   QN("Sz",-1)=>15, QN("Sz",-3)=>7, QN("Sz",-5)=>2], "Link,qr")
const ll  = Index([QN()=>1, QN("Sz",0)=>6, QN("Sz",2)=>6,
                   QN("Sz",-2)=>6, QN("Sz",0)=>1], "Link,l=32")
const s   = Index([QN("Sz",1)=>1, QN("Sz",-1)=>1], "S=1/2,Site,n=2")
const p1  = Index([QN("Sz",6)=>1, QN("Sz",4)=>5, QN("Sz",2)=>12,
                   QN("Sz",0)=>16, QN("Sz",-2)=>12,
                   QN("Sz",-4)=>5, QN("Sz",-6)=>1], "Link,qr")

const L  = random_itensor(QN("Sz",0), dag(lqr), prime(lqr), ll, dag(s), prime(s))
const R  = random_itensor(QN("Sz",0), dag(p1), dag(ll), prime(p1))

# LH = L * phi (environment, fixed across Lanczos iterations)
const LH = L * random_itensor(QN("Sz",0), p1, lqr, s)

const lt = (3, -1, 4, -2)
const lr = (-2, -1, 5)

LH_nd = NDTensors.tensor(LH)
R_nd = NDTensors.tensor(R)

println("\nTensor structure:")
println("  LH (4D):   $(dim(LH)) total dims, $(nnz(LH)) nnz, labels $lt")
println("  R  (3D):   $(dim(R)) total dims, $(nnz(R)) nnz, labels $lr")
println()

# Correctness
println("--- Correctness ---")
C_ref = contract(LH_nd, lt, R_nd, lr)

# No-pre-permute: to_buffer each iteration (hits internal permutedims)
buf_c = Bumper.SlabBuffer{2^25}()
C_nopre = with_alloc_buffer(buf_c) do
    contract(to_buffer(LH_nd, buf_c), lt, to_buffer(R_nd, buf_c), lr)
end
println("No-pre match: ", isapprox(copy(C_ref), copy(C_nopre); atol=1e-10))

# Pre-permuted once
# permA = free_first = (1,3,2,4): A's free labels (3,4) come before contracted (-1,-2)
# permB = (2,1,3): swap B's contracted to match A's order: (-1,-2,5)
# New labels: A=(1,2,-1,-2), B=(-1,-2,3), R=(1,2,3) → trivA, trivB
buf_p = Bumper.SlabBuffer{2^25}()
LH_p = with_alloc_buffer(buf_p) do
    to_buffer(LH_nd, (1, 3, 2, 4), buf_p)
end
R_p = with_alloc_buffer(buf_p) do
    to_buffer(R_nd, (2, 1, 3), buf_p)
end
C_pre = with_alloc_buffer(buf_p) do
    contract(LH_p, (1, 2, -1, -2), R_p, (-1, -2, 3), (1, 2, 3))
end
println("Pre-perm match: ", isapprox(copy(C_ref), copy(C_pre); atol=1e-10))

# ── Benchmark ──
println("\n--- Benchmark (10 contraction iterations) ---")

# No-pre: to_buffer + contract each iter (10x)
function bench_no_pre()
    for iter in 1:10
        buf = Bumper.SlabBuffer{2^25}()
        with_alloc_buffer(buf) do
            contract(to_buffer(LH_nd, buf), lt, to_buffer(R_nd, buf), lr)
        end
    end
    nothing
end

function bench_no_pre_thr()
    enable_threaded_blocksparse()
    for iter in 1:10
        buf = Bumper.SlabBuffer{2^25}()
        with_alloc_buffer(buf) do
            contract(to_buffer(LH_nd, buf), lt, to_buffer(R_nd, buf), lr)
        end
    end
    disable_threaded_blocksparse()
    nothing
end

# Pre-perm: permute once, then contract 10x
function bench_pre()
    buf = Bumper.SlabBuffer{2^25}()
    LH_p = to_buffer(LH_nd, (1, 3, 2, 4), buf)
    R_p = to_buffer(R_nd, (2, 1, 3), buf)
    for iter in 1:10
        with_alloc_buffer(buf) do
            contract(LH_p, (1, 2, -1, -2), R_p, (-1, -2, 3), (1, 2, 3))
        end
    end
    nothing
end

function bench_pre_thr()
    buf = Bumper.SlabBuffer{2^25}()
    LH_p = to_buffer(LH_nd, (1, 3, 2, 4), buf)
    R_p = to_buffer(R_nd, (2, 1, 3), buf)
    enable_threaded_blocksparse()
    for iter in 1:10
        with_alloc_buffer(buf) do
            contract(LH_p, (1, 2, -1, -2), R_p, (-1, -2, 3), (1, 2, 3))
        end
    end
    disable_threaded_blocksparse()
    nothing
end

# Warm-up
bench_no_pre(); bench_no_pre_thr()
bench_pre(); bench_pre_thr()

println("(each runs 10 contraction iterations, ~5 sec measurement)")
println()

for (name, fn) in [("No pre-perm seq", bench_no_pre), ("No pre-perm thr", bench_no_pre_thr),
                   ("Pre-permute seq", bench_pre), ("Pre-permute thr", bench_pre_thr)]
    println("Benchmark: $name")
    b = @benchmark $fn() seconds=5
    show(stdout, "text/plain", b); println(); println()
end
