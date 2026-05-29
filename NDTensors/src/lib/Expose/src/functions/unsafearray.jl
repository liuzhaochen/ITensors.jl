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
    Strided.@strided Mperm = permutedims(unexpose(E), perm)
    return Mperm
end

function permutedims!(
    Edest::Exposed{<:UnsafeArray}, Esrc::Exposed{<:UnsafeArray}, perm
)
    a_dest = unexpose(Edest)
    a_src = unexpose(Esrc)
    Strided.@strided a_dest .= permutedims(a_src, perm)
    return a_dest
end

function permutedims!(
    Edest::Exposed{<:UnsafeArray}, Esrc::Exposed{<:UnsafeArray}, perm, f
)
    a_dest = unexpose(Edest)
    a_src = unexpose(Esrc)
    Strided.@strided a_dest .= f.(a_dest, permutedims(a_src, perm))
    return a_dest
end
