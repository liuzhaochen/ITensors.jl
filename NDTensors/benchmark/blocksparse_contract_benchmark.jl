# BlockSparse contraction benchmark: heap vs buffer, single vs threaded
# Compares all relevant paths:
#   1. Heap sequential  — baseline
#   2. Heap threaded    — original Folds.foreach + ThreadedEx
#   3. Buffer sequential — buffer-backed, single-threaded
#   4. Buffer threaded  — buffer-backed, @spawn + TBLIS/heap-fallback
#
# Usage:
#   julia -t 4 --project=benchmark NDTensors/benchmark/blocksparse_contract_benchmark.jl

using NDTensors
using NDTensors: Bumper, with_alloc_buffer, storage, contract, blockoffsets
using LinearAlgebra, Statistics

BLAS.set_num_threads(1)
NDTensors.Strided.disable_threads()

# Trigger TBLIS extension loading if available
isdefined(Base, :get_extension) || error("Julia 1.10+ required")
try
    Base.get_extension(NDTensors, :NDTensorsTBLISExt) === nothing && @eval import TBLIS
catch
end
has_tblis = Base.get_extension(NDTensors, :NDTensorsTBLISExt) !== nothing
if has_tblis
    try
        TBLIS.set_num_threads(1)
    catch
    end
end

const NTHREADS = Threads.nthreads()
println("="^70)
println("BlockSparse Contraction Benchmark")
println("="^70)
println("Julia threads:  $NTHREADS")
println("BLAS threads:   $(BLAS.get_num_threads())")
println("Strided threads: $(NDTensors.Strided.get_num_threads())")
println("TBLIS loaded:   $has_tblis")
has_tblis && println("TBLIS threads:  $(TBLIS.get_num_threads())")

# ---------------------------------------------------------------------------
# Tensor setup: block-sparse with multiple blocks of varying sizes
# ---------------------------------------------------------------------------
# Create tensors with QN-like block structure (non-zero blocks at specific
# positions). Indices are vectors of block dimensions.
# DMRG-scale block sizes for meaningful threading benchmark.
# Matched blocks: (1,3)×(1,3) = (1,2), (2,1)×(2,1) = (2,2), (3,2)×(3,2) = (3,3)

const BLOCKS_A = [(1,3), (2,1), (3,2)]
const BLOCKS_B = [(1,3), (2,1), (3,2)]
const DIMS_A = ([100, 200, 300], [150, 250, 350])
const DIMS_B = ([150, 250, 350], [200, 300, 400])

function fill_tensors!(A, B)
    fill!(storage(A), 1.0)
    fill!(storage(B), 2.0)
end

# ---------------------------------------------------------------------------
# Create reference tensors
# ---------------------------------------------------------------------------
const A_HEAP = NDTensors.BlockSparseTensor{Float64}(BLOCKS_A, DIMS_A...)
const B_HEAP = NDTensors.BlockSparseTensor{Float64}(BLOCKS_B, DIMS_B...)
fill_tensors!(A_HEAP, B_HEAP)

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
println("\n--- Correctness ---")
C_ref = contract(A_HEAP, (1, 2), B_HEAP, (2, 3), (1, 3))

# Buffer single-threaded
buf = Bumper.SlabBuffer()
C_buf_seq = with_alloc_buffer(buf) do
    contract(NDTensors.to_buffer(A_HEAP, buf), (1,2), NDTensors.to_buffer(B_HEAP, buf), (2,3), (1,3))
end
println("Buffer sequential match: ", isapprox(C_ref, C_buf_seq, rtol=1e-10))

# Buffer threaded
Bumper.reset_buffer!(buf)
NDTensors.enable_threaded_blocksparse()
C_buf_thr = with_alloc_buffer(buf) do
    contract(NDTensors.to_buffer(A_HEAP, buf), (1,2), NDTensors.to_buffer(B_HEAP, buf), (2,3), (1,3))
end
NDTensors.disable_threaded_blocksparse()
println("Buffer threaded match:   ", isapprox(C_ref, C_buf_thr, rtol=1e-10))

# Heap threaded (original path)
NDTensors.enable_threaded_blocksparse()
C_heap_thr = contract(A_HEAP, (1, 2), B_HEAP, (2, 3), (1, 3))
NDTensors.disable_threaded_blocksparse()
println("Heap threaded match:     ", isapprox(C_ref, C_heap_thr, rtol=1e-10))

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------
println("\n--- Benchmark ---")

function bench_heap_seq()
    A = NDTensors.BlockSparseTensor{Float64}(BLOCKS_A, DIMS_A...)
    B = NDTensors.BlockSparseTensor{Float64}(BLOCKS_B, DIMS_B...)
    fill_tensors!(A, B)
    contract(A, (1, 2), B, (2, 3), (1, 3))
end

function bench_heap_thr()
    NDTensors.enable_threaded_blocksparse()
    A = NDTensors.BlockSparseTensor{Float64}(BLOCKS_A, DIMS_A...)
    B = NDTensors.BlockSparseTensor{Float64}(BLOCKS_B, DIMS_B...)
    fill_tensors!(A, B)
    r = contract(A, (1, 2), B, (2, 3), (1, 3))
    NDTensors.disable_threaded_blocksparse()
    return r
end

function bench_buf_seq(buf)
    Bumper.reset_buffer!(buf)
    with_alloc_buffer(buf) do
        contract(NDTensors.to_buffer(A_HEAP, buf), (1,2), NDTensors.to_buffer(B_HEAP, buf), (2,3), (1,3))
    end
end

function bench_buf_thr(buf)
    Bumper.reset_buffer!(buf)
    NDTensors.enable_threaded_blocksparse()
    r = with_alloc_buffer(buf) do
        contract(NDTensors.to_buffer(A_HEAP, buf), (1,2), NDTensors.to_buffer(B_HEAP, buf), (2,3), (1,3))
    end
    NDTensors.disable_threaded_blocksparse()
    return r
end

# Warm-up
bench_heap_seq(); GC.gc()
let b = Bumper.SlabBuffer()
    bench_buf_seq(b); bench_buf_thr(b)
end; GC.gc()

const NSAMPLES = 11  # first is warm-up, remaining 10 measured

function run_bench(name, f)
    println("▶ $name")
    times = Float64[]
    # Warm-up + 10 measured
    for i in 1:NSAMPLES
        t = @elapsed f()
        if i > 1
            push!(times, t * 1000)  # seconds → ms
        end
    end
    t_min = minimum(times)
    t_med = median(times)
    t_max = maximum(times)
    println("  min: $(round(t_min, digits=3)) ms  med: $(round(t_med, digits=3)) ms  max: $(round(t_max, digits=3)) ms")
    println("  (over $(NSAMPLES-1) samples)")
    return times
end

# Allocate buffers before timing
buf_seq = Bumper.SlabBuffer()
buf_thr = Bumper.SlabBuffer()

t_heap_seq = run_bench("Heap sequential", bench_heap_seq)
t_heap_thr = run_bench("Heap threaded", bench_heap_thr)
t_buf_seq  = run_bench("Buffer sequential", () -> bench_buf_seq(buf_seq))
t_buf_thr  = run_bench("Buffer threaded", () -> bench_buf_thr(buf_thr))

# Summary
println("\n" * "="^70)
println("Summary")
println("="^70)
println("                    │ Time (ms)   │ vs heap seq")
println("─"^70)

function summary_row(label, times, ref)
    t = median(times)
    ratio = ref / t
    println(rpad(label, 22) * "│ $(rpad(round(t, digits=3), 10)) │ $(round(ratio, digits=2))x")
end

ref = median(t_heap_seq)
summary_row("Heap sequential", t_heap_seq, ref)
summary_row("Heap threaded", t_heap_thr, ref)
summary_row("Buffer sequential", t_buf_seq, ref)
summary_row("Buffer threaded", t_buf_thr, ref)
println("─"^70)
println("Higher vs heap seq = faster (speedup)")
println("Lower vs heap seq = slower (slowdown)")
