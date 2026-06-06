# QN ITensor interleaved-index contraction benchmark: heap vs buffer, single vs threaded
# Forces _contract_fallback! through the non-trivial perm path (A needs permutedims).
# Uses BenchmarkTools for proper statistics.
#
# QN block structure (5 sectors, 6 blocks per index):
#   QN(0) × 2 (20, 10), QN(±1) × 1 (10), QN(±2) × 1 (5,5)
# Total: 6 blocks, dim=60 per index → 3D tensors: 216K full elements, ~52.5K nonzeros
#
# Contraction A[j,i,k] × B[j,k,l] → R[i,l]:
#   A labels: (-1, 1, -2)  → permA = [2,1,3] (non-trivial, non-swap)
#   B labels: (-1,-2,  2)  → permB = [1,2,3] (trivB, zero-copy)
#   R labels: ( 1,  2)     → need_c_perm = false
#
# Usage:
#   julia -t 4 --project=benchmark benchmark/qn_itensor_interleaved_benchmark.jl

using ITensors
using ITensors: to_buffer
using NDTensors: Bumper, with_alloc_buffer
using ITensors: enable_threaded_blocksparse, disable_threaded_blocksparse
using LinearAlgebra, BenchmarkTools

BLAS.set_num_threads(1)
ITensors.NDTensors.Strided.disable_threads()

const NTHREADS = Threads.nthreads()
println("="^70)
println("QN ITensor Interleaved Contraction Benchmark")
println("="^70)
println("Julia threads:  $NTHREADS")
println("BLAS threads:   $(BLAS.get_num_threads())")
println("Strided threads: $(ITensors.NDTensors.Strided.get_num_threads())")

# ---------------------------------------------------------------------------
# QN Index setup: smaller blocks for 3D tensors
# ---------------------------------------------------------------------------
const SMALL_SPEC = [
    QN(0) => 20, QN(0) => 10,
    QN(1) => 10,
    QN(-1) => 10,
    QN(2) => 5,
    QN(-2) => 5,
]

const i_sm = Index(SMALL_SPEC, "i_sm")
const j_sm = Index(SMALL_SPEC, "j_sm")
const k_sm = Index(SMALL_SPEC, "k_sm")
const l_sm = Index(SMALL_SPEC, "l_sm")

const A3D_HEAP = random_itensor(QN(0), j_sm, i_sm, k_sm)  # free i in middle
const B3D_HEAP = random_itensor(QN(0), dag(j_sm), dag(k_sm), l_sm)  # contracted dag'd

println("\nIndex structure:")
println("  i_sm: $(dim(i_sm)) total dim, $(nblocks(space(i_sm))) QN blocks")
println("  j_sm: $(dim(j_sm)) total dim, $(nblocks(space(j_sm))) QN blocks")
println("  k_sm: $(dim(k_sm)) total dim, $(nblocks(space(k_sm))) QN blocks")
println("  l_sm: $(dim(l_sm)) total dim, $(nblocks(space(l_sm))) QN blocks")
println("  A nnz: $(nnz(A3D_HEAP)) / $(dim(j_sm)*dim(i_sm)*dim(k_sm)) elements")
println("  B nnz: $(nnz(B3D_HEAP)) / $(dim(j_sm)*dim(k_sm)*dim(l_sm)) elements")

println("\nLabel layout:")
println("  A labels: (-1, 1, -2)  → permA = [2,1,3] (non-trivial, non-swap)")
println("  B labels: (-1,-2,  2)  → permB = [1,2,3] (trivB, zero-copy)")
println("  R labels: ( 1,  2)     → need_c_perm = false")

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
println("\n--- Correctness ---")
C3d_ref = A3D_HEAP * B3D_HEAP

buf3d_check = Bumper.SlabBuffer{2^25}()
buf3d_seq = Bumper.SlabBuffer{2^25}()
C3d_buf_seq = with_alloc_buffer(buf3d_seq) do
    Bumper.reset_buffer!(buf3d_seq)
    to_buffer(A3D_HEAP, buf3d_seq) * to_buffer(B3D_HEAP, buf3d_seq)
end
with_alloc_buffer(buf3d_check) do
    Bumper.reset_buffer!(buf3d_check)
    println("Buffer sequential match: ", isapprox(C3d_ref, C3d_buf_seq))
end

buf3d_thr = Bumper.SlabBuffer{2^25}()
enable_threaded_blocksparse()
C3d_buf_thr = with_alloc_buffer(buf3d_thr) do
    Bumper.reset_buffer!(buf3d_thr)
    to_buffer(A3D_HEAP, buf3d_thr) * to_buffer(B3D_HEAP, buf3d_thr)
end
disable_threaded_blocksparse()
with_alloc_buffer(buf3d_check) do
    Bumper.reset_buffer!(buf3d_check)
    println("Buffer threaded match:   ", isapprox(C3d_ref, C3d_buf_thr))
end

enable_threaded_blocksparse()
C3d_heap_thr = A3D_HEAP * B3D_HEAP
disable_threaded_blocksparse()
println("Heap threaded match:     ", isapprox(C3d_ref, C3d_heap_thr))

println("\nBlock structure:")
println("  R nnz: $(nnz(C3d_ref))")

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------
println("\n--- Benchmark ---")

function bench_heap_seq()
    A = random_itensor(QN(0), j_sm, i_sm, k_sm)
    B = random_itensor(QN(0), dag(j_sm), dag(k_sm), l_sm)
    A * B
end

function bench_heap_thr()
    enable_threaded_blocksparse()
    A = random_itensor(QN(0), j_sm, i_sm, k_sm)
    B = random_itensor(QN(0), dag(j_sm), dag(k_sm), l_sm)
    r = A * B
    disable_threaded_blocksparse()
    return r
end

function bench_buf_seq(buf)
    Bumper.reset_buffer!(buf)
    with_alloc_buffer(buf) do
        to_buffer(A3D_HEAP, buf) * to_buffer(B3D_HEAP, buf)
    end
end

function bench_buf_thr(buf)
    Bumper.reset_buffer!(buf)
    enable_threaded_blocksparse()
    r = with_alloc_buffer(buf) do
        to_buffer(A3D_HEAP, buf) * to_buffer(B3D_HEAP, buf)
    end
    disable_threaded_blocksparse()
    return r
end

# Warm-up + precompile
bench_heap_seq(); bench_heap_thr()
bench_buf_seq(buf3d_seq); bench_buf_thr(buf3d_thr)

println("(each benchmark runs until ~5 seconds of measurement)")
println()

b_h_seq = @benchmark bench_heap_seq() seconds=5
println("Heap sequential:")
show(stdout, "text/plain", b_h_seq); println()

b_h_thr = @benchmark bench_heap_thr() seconds=5
println("Heap threaded:")
show(stdout, "text/plain", b_h_thr); println()

b_b_seq = @benchmark bench_buf_seq($buf3d_seq) seconds=5
println("Buffer sequential:")
show(stdout, "text/plain", b_b_seq); println()

b_b_thr = @benchmark bench_buf_thr($buf3d_thr) seconds=5
println("Buffer threaded:")
show(stdout, "text/plain", b_b_thr); println()

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

ref = BenchmarkTools.median(b_h_seq).time / 1e6
summary_row("Heap sequential", b_h_seq, ref)
summary_row("Heap threaded",   b_h_thr, ref)
summary_row("Buffer sequential", b_b_seq, ref)
summary_row("Buffer threaded",   b_b_thr, ref)
println("─"^70)
println("Higher vs heap seq = faster (speedup)")
