# Benchmark: QN block-sparse ITensor operations — heap vs buffer allocation
#
# Usage:
#   julia --project examples/buffer_benchmark/qn.jl
using LinearAlgebra
using ITensors
using ITensors: to_buffer
using NDTensors: Bumper, with_alloc_buffer
using BenchmarkTools
using TBLIS
BLAS.set_num_threads(1)
NDTensors.Strided.disable_threads()
TBLIS.set_num_threads(1)

let
    # ── Setup ──
    i = Index([QN(0) => 4, QN(1) => 3, QN(-1) => 3], "i")
    j = Index([QN(0) => 4, QN(1) => 3, QN(-1) => 3], "j")
    k = Index([QN(0) => 5, QN(1) => 4, QN(-1) => 4], "k")
    l = Index([QN(0) => 4, QN(1) => 3, QN(-1) => 3], "l")

    # Heap tensors
    A_h = random_itensor(i, dag(j))
    B_h = random_itensor(j, dag(k))
    C_h = random_itensor(k, dag(l))
    Th = random_itensor(i, dag(j), k)

    # Buffer
    buf = Bumper.ResizeBuffer()
    # buf = Bumper.SlabBuffer()

    # Buffer tensors
    B_b = to_buffer(B_h, buf)
    T_b = to_buffer(Th, buf)
    # return Th
    @btime begin
        with_alloc_buffer($buf) do
            $T_b * $B_b
        end
        # Bumper.reset_buffer!()
    end
    t = with_alloc_buffer(buf) do 
        T_b*B_b
    end
    @show t,copy(t)
    #  23.935 μs (118 allocations: 9.25 KiB)

    @btime begin
        C = $Th * $B_h
    end
    #  12.674 μs (96 allocations: 8.88 KiB)
end
