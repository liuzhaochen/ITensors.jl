# [FORK] Exposed{<:UnsafeArray} specializations for @strided-accelerated
# mul! and permutedims on buffer-allocated tensors.
# Same pattern as Exposed{<:Array} in array/mul.jl and array/permutedims.jl.

import ..NDTensors: UnsafeArray, Strided

function mul!(
    CM::Exposed{<:UnsafeArray},
    AM::Exposed{<:UnsafeArray},
    BM::Exposed{<:UnsafeArray},
    α,
    β,
)
    Strided.@strided mul!(unexpose(CM), unexpose(AM), unexpose(BM), α, β)
    return unexpose(CM)
end

function permutedims(E::Exposed{<:UnsafeArray}, perm)
    a_src = unexpose(E)
    return Base.permutedims(a_src, perm)
end

function permutedims!(
    Edest::Exposed{<:UnsafeArray}, Esrc::Exposed{<:UnsafeArray}, perm
)
    a_dest = unexpose(Edest)
    a_src = unexpose(Esrc)
    N = length(perm)
    for I in CartesianIndices(a_src)
        J = CartesianIndex(ntuple(i -> I[perm[i]], Val(N)))
        a_dest[J] = a_src[I]
    end
    return a_dest
end

function permutedims!(
    Edest::Exposed{<:UnsafeArray}, Esrc::Exposed{<:UnsafeArray}, perm, f
)
    a_dest = unexpose(Edest)
    a_src = unexpose(Esrc)
    ip = Base.invperm(perm)
    N = length(perm)
    for J in CartesianIndices(a_dest)
        I = CartesianIndex(ntuple(i -> J[ip[i]], Val(N)))
        a_dest[J] = f(a_dest[J], a_src[I])
    end
    return a_dest
end
