# DMRG-like contraction benchmark: heap vs buffer allocation
# Uses BenchmarkTools.jl for statistical rigor.

using BenchmarkTools
using NDTensors
using NDTensors: Dense, get_alloc_buffer, with_alloc_buffer
using LinearAlgebra

BLAS.set_num_threads(1)

const DIM1, DIM2, DIM3, DIM4 = 100, 100, 100, 100

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
println("="^65)
println("Correctness check: heap vs buffer contraction results")
println("="^65)

let
    a_h = NDTensors.DenseTensor(Float64, (DIM1, DIM2, DIM3))
    b_h = NDTensors.DenseTensor(Float64, (DIM3, DIM4))
    fill!(a_h, 1.0); fill!(b_h, 2.0)
    result_heap = NDTensors.contract(a_h, (1, 2, -1), b_h, (-1, 3), (1, 2, 3))

    buf = NDTensors.Bumper.SlabBuffer()
    result_buf = with_alloc_buffer(buf) do
        a_b = NDTensors.DenseTensor(Float64, get_alloc_buffer(), (DIM1, DIM2, DIM3))
        b_b = NDTensors.DenseTensor(Float64, get_alloc_buffer(), (DIM3, DIM4))
        fill!(a_b, 1.0); fill!(b_b, 2.0)
        NDTensors.contract(a_b, (1, 2, -1), b_b, (-1, 3), (1, 2, 3))
    end

    diff = norm(Array(result_heap) - Array(result_buf))
    if diff ≈ 0.0
        println("  ✓ Results match: |heap - buffer| = $diff")
    else
        println("  ✗ MISMATCH: |heap - buffer| = $diff")
    end
end
println()

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
println("="^65)
println("Single contraction benchmark: heap vs buffer allocation")
println("="^65)
println("Tensor sizes: a=($DIM1,$DIM2,$DIM3), b=($DIM3,$DIM4)")
println("Output size:  ($DIM1,$DIM2,$DIM4)")
println("BLAS threads: $(BLAS.get_num_threads())")
println("-"^65)

# Benchmark functions
function bench_heap()
    a = NDTensors.DenseTensor(Float64, (DIM1, DIM2, DIM3))
    b = NDTensors.DenseTensor(Float64, (DIM3, DIM4))
    fill!(a, 1.0)
    fill!(b, 2.0)
    NDTensors.contract(a, (1, 2, -1), b, (-1, 3), (1, 2, 3))
end

function bench_buf(buf)
    NDTensors.Bumper.reset_buffer!(buf)
    with_alloc_buffer(buf) do
        a = NDTensors.DenseTensor(Float64, get_alloc_buffer(), (DIM1, DIM2, DIM3))
        b = NDTensors.DenseTensor(Float64, get_alloc_buffer(), (DIM3, DIM4))
        fill!(a, 1.0)
        fill!(b, 2.0)
        NDTensors.contract(a, (1, 2, -1), b, (-1, 3), (1, 2, 3))
    end
end

# Warm-up
bench_heap()
GC.gc()
let
    b = NDTensors.Bumper.SlabBuffer()
    bench_buf(b)
    NDTensors.Bumper.reset_buffer!(b)
end
GC.gc()

# Benchmark heap
println("Heap allocation:")
heap_result = @benchmark bench_heap() samples=20 evals=1
display(heap_result)
println()

# Benchmark buffer
buf = NDTensors.Bumper.SlabBuffer()
buf_result = @benchmark bench_buf($buf) samples=20 evals=1
display(buf_result)
println()

# Summary
println("-"^65)
println("Summary:")
let
    h_med = median(heap_result.times) / 1e6
    b_med = median(buf_result.times) / 1e6
    h_mem = median(heap_result.memory)
    b_mem = median(buf_result.memory)
    h_gc = median(heap_result.gctimes) / 1e6
    b_gc = median(buf_result.gctimes) / 1e6
    h_min = minimum(heap_result.times) / 1e6
    b_min = minimum(buf_result.times) / 1e6

    println("                    Heap         Buffer       Reduction")
    println("Median time:       $(round(h_med, digits=3)) ms    $(round(b_med, digits=3)) ms    $(round(h_med/b_med, digits=1))x faster")
    println("Min time (no GC):  $(round(h_min, digits=3)) ms    $(round(b_min, digits=3)) ms")
    println("Memory allocated:  $(round(h_mem/1024, digits=1)) KiB  $(round(b_mem/1024, digits=1)) KiB  $(round(h_mem/max(b_mem,1), digits=0))x less")
    println("GC time:           $(round(h_gc, digits=3)) ms    $(round(b_gc, digits=3)) ms    eliminated")
    println("Variance:          ±$(round(std(heap_result.times)/1e6, digits=3)) ms   ±$(round(std(buf_result.times)/1e6, digits=3)) ms   $(round(std(heap_result.times)/max(std(buf_result.times),1), digits=0))x more stable")
end
println("="^65)
