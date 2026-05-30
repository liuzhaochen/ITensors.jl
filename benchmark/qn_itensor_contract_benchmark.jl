# QN ITensor contraction benchmark: heap vs buffer, single vs threaded
# Uses BenchmarkTools for proper statistics.
#
# QN block structure (5 sectors):
#   QN(0) => large bond dim (DMRG-like)
#   QN(±1), QN(±2) => smaller virtual dims
# This gives 5 non-zero blocks per tensor (QN-conserving with flux QN(0)).
#
# Usage:
#   julia -t 4 --project=benchmark benchmark/qn_itensor_contract_benchmark.jl

using ITensors
using ITensors: to_buffer
using NDTensors: Bumper, with_alloc_buffer
using NDTensors: enable_threaded_blocksparse, disable_threaded_blocksparse
using LinearAlgebra, BenchmarkTools

BLAS.set_num_threads(1)
ITensors.NDTensors.Strided.disable_threads()

# Trigger TBLIS extension loading if available
isdefined(Base, :get_extension) || error("Julia 1.10+ required")
try
    Base.get_extension(ITensors.NDTensors, :NDTensorsTBLISExt) === nothing && @eval import TBLIS
catch
end
has_tblis = Base.get_extension(ITensors.NDTensors, :NDTensorsTBLISExt) !== nothing
if has_tblis
    try TBLIS.set_num_threads(1) catch end
end

const NTHREADS = Threads.nthreads()
println("="^70)
println("QN ITensor Contraction Benchmark (BenchmarkTools)")
println("="^70)
println("Julia threads:  $NTHREADS")
println("BLAS threads:   $(BLAS.get_num_threads())")
println("Strided threads: $(ITensors.NDTensors.Strided.get_num_threads())")
println("TBLIS loaded:   $has_tblis")
has_tblis && println("TBLIS threads:  $(TBLIS.get_num_threads())")

# ---------------------------------------------------------------------------
# QN Index setup: 5 QN sectors, DMRG-like dimensions
# ---------------------------------------------------------------------------
# 10 QN blocks per index, same total dim=1000 as before:
#   QN(0) × 200, QN(0) × 100  (2 blocks, sum=300)
#   QN(1) × 100, QN(1) × 100  (2 blocks, sum=200)
#   QN(-1) × 100, QN(-1) × 100 (2 blocks, sum=200)
#   QN(2) × 100, QN(2) × 50   (2 blocks, sum=150)
#   QN(-2) × 100, QN(-2) × 50  (2 blocks, sum=150)
# Total: 10 blocks, dim=1000

const BLOCK_SPEC = [
    QN(0) => 200, QN(0) => 100,
    QN(1) => 100, QN(1) => 100,
    QN(-1) => 100, QN(-1) => 100,
    QN(2) => 100, QN(2) => 50,
    QN(-2) => 100, QN(-2) => 50,
]

const i = Index(BLOCK_SPEC, "i")
const j = Index(BLOCK_SPEC, "j")
const k = Index(BLOCK_SPEC, "k")

println("\nIndex structure:")
println("  i: $(dim(i)) total dim, $(nblocks(space(i))) QN blocks")
println("  j: $(dim(j)) total dim, $(nblocks(space(j))) QN blocks")
println("  k: $(dim(k)) total dim, $(nblocks(space(k))) QN blocks")

const A_HEAP = random_itensor(QN(0), i, j)
const B_HEAP = random_itensor(QN(0), dag(j), k)

println("  A nnz: $(nnz(A_HEAP)) / $(dim(i)*dim(j)) elements")
println("  B nnz: $(nnz(B_HEAP)) / $(dim(j)*dim(k)) elements")

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
println("\n--- Correctness ---")
C_ref = A_HEAP * B_HEAP

buf_seq = Bumper.SlabBuffer{2^25}()
C_buf_seq = with_alloc_buffer(buf_seq) do
    Bumper.reset_buffer!(buf_seq)
    to_buffer(A_HEAP, buf_seq) * to_buffer(B_HEAP, buf_seq)
end
buf_check = Bumper.SlabBuffer{2^25}()
with_alloc_buffer(buf_check) do
    Bumper.reset_buffer!(buf_check)
    println("Buffer sequential match: ", isapprox(C_ref, C_buf_seq))
end

buf_thr = Bumper.SlabBuffer{2^25}()
enable_threaded_blocksparse()
C_buf_thr = with_alloc_buffer(buf_thr) do
    Bumper.reset_buffer!(buf_thr)
    to_buffer(A_HEAP, buf_thr) * to_buffer(B_HEAP, buf_thr)
end
disable_threaded_blocksparse()
with_alloc_buffer(buf_check) do
    Bumper.reset_buffer!(buf_check)
    println("Buffer threaded match:   ", isapprox(C_ref, C_buf_thr))
end

enable_threaded_blocksparse()
C_heap_thr = A_HEAP * B_HEAP
disable_threaded_blocksparse()
println("Heap threaded match:     ", isapprox(C_ref, C_heap_thr))

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------
println("\n--- Benchmark ---")

function bench_heap_seq()
    A = random_itensor(QN(0), i, j)
    B = random_itensor(QN(0), dag(j), k)
    A * B
end

function bench_heap_thr()
    enable_threaded_blocksparse()
    A = random_itensor(QN(0), i, j)
    B = random_itensor(QN(0), dag(j), k)
    r = A * B
    disable_threaded_blocksparse()
    return r
end

function bench_buf_seq(buf)
    Bumper.reset_buffer!(buf)
    with_alloc_buffer(buf) do
        to_buffer(A_HEAP, buf) * to_buffer(B_HEAP, buf)
    end
end

function bench_buf_thr(buf)
    Bumper.reset_buffer!(buf)
    enable_threaded_blocksparse()
    r = with_alloc_buffer(buf) do
        to_buffer(A_HEAP, buf) * to_buffer(B_HEAP, buf)
    end
    disable_threaded_blocksparse()
    return r
end

# Warm-up + precompile
bench_heap_seq()
bench_heap_thr()
bench_buf_seq(buf_seq)
bench_buf_thr(buf_thr)

println("(each benchmark runs until ~5 seconds of measurement)")
println()

b_heap_seq = @benchmark bench_heap_seq() seconds=5
println("Heap sequential:")
show(stdout, "text/plain", b_heap_seq)
println()

b_heap_thr = @benchmark bench_heap_thr() seconds=5
println("Heap threaded:")
show(stdout, "text/plain", b_heap_thr)
println()

b_buf_seq = @benchmark bench_buf_seq($buf_seq) seconds=5
println("Buffer sequential:")
show(stdout, "text/plain", b_buf_seq)
println()

b_buf_thr = @benchmark bench_buf_thr($buf_thr) seconds=5
println("Buffer threaded:")
show(stdout, "text/plain", b_buf_thr)
println()

# Summary
println("="^70)
println("Summary (median ± σ)")
println("="^70)
println("                    │ Time (ms)        │ Allocs    │ Memory      │ vs heap seq")
println("─"^70)

function summary_row(name, trial, ref)
    m = BenchmarkTools.median(trial)
    σ = std(trial.times) / 1e6
    t = m.time / 1e6
    allocs = m.allocs
    mem = m.memory / 1024
    ratio = ref / t
    println(rpad(name, 22) * "│ $(rpad(round(t, digits=3), 6)) ± $(rpad(round(σ, digits=3), 6)) │ $(rpad(allocs, 9)) │ $(rpad(round(mem, digits=1), 8)) KiB │ $(round(ratio, digits=2))x")
end

ref = BenchmarkTools.median(b_heap_seq).time / 1e6
summary_row("Heap sequential", b_heap_seq, ref)
summary_row("Heap threaded", b_heap_thr, ref)
summary_row("Buffer sequential", b_buf_seq, ref)
summary_row("Buffer threaded", b_buf_thr, ref)
println("─"^70)
println("Higher vs heap seq = faster (speedup)")
