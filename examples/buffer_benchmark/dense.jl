# Benchmark: Dense ITensor operations — heap vs buffer allocation
#
# Usage:  julia --project examples/buffer_benchmark/dense.jl

using ITensors
using ITensors: to_buffer
using NDTensors: Bumper, with_alloc_buffer
using BenchmarkTools
let
    N = 20
    i = Index(N, "i"); j = Index(N, "j")
    k = Index(N, "k"); l = Index(N, "l")

    A_h = random_itensor(i, j)
    B_h = random_itensor(j, k)
    C_h = random_itensor(k, l)
    T_h = random_itensor(i, j, k)
    X_h = random_itensor(i, j)
    Y_h = random_itensor(i, j)

    buf = Bumper.ResizeBuffer()
    B_b = to_buffer(B_h, buf)
    T_b = to_buffer(T_h, buf)
    @btime begin
        with_alloc_buffer($buf) do
            $T_b * $B_b
            return nothing
        end
        # Bumper.reset_buffer!()
    end

    @btime begin
        C = $T_h * $B_h
    end

end
