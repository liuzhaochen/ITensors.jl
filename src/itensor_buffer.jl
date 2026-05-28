# [FORK] Buffer-aware ITensor constructors
# Provides user-facing ITensor(buf, inds) API for allocating tensor
# storage from a Bumper.jl buffer arena.
# See .claude/mission.md and .claude/buffer_borrowing_model.md for context.

# -----------------------------------------------------------
# ITensor(buf, inds) — buffer-allocated (uninitialized)
# -----------------------------------------------------------

"""
    ITensor(buf, [::Type{ElT}=Float64], inds)
    ITensor(buf, [::Type{ElT}=Float64], inds::Index...)

Construct an ITensor with storage allocated from Bumper buffer `buf`.
The tensor data lives in the buffer arena and is valid only until the
buffer is reset. The storage is uninitialized (like `undef`).

# Examples
```julia
buf = ResizeBuffer()
i = Index(2)
j = Index(3)
A = ITensor(buf, i, j)              # Float64, from buffer
B = ITensor(buf, ComplexF64, i, j)  # ComplexF64, from buffer
```
"""
function ITensor(buf::NDTensors.AllocBuffer, ::Type{ElT}, inds::Indices) where {ElT <: Number}
    storage = Dense{ElT}(buf, dim(inds))
    return itensor(storage, indices(inds))
end

function ITensor(buf::NDTensors.AllocBuffer, ::Type{ElT}, inds...) where {ElT <: Number}
    return ITensor(buf, ElT, indices(inds...))
end

function ITensor(buf::NDTensors.AllocBuffer, inds::Indices)
    return ITensor(buf, Float64, inds)
end

function ITensor(buf::NDTensors.AllocBuffer, inds...)
    return ITensor(buf, Float64, indices(inds...))
end

# -----------------------------------------------------------
# ITensor(buf, undef, inds) — explicitly uninitialized
# -----------------------------------------------------------

function ITensor(
    buf::NDTensors.AllocBuffer, ::Type{ElT}, ::UndefInitializer, inds::Indices
) where {ElT <: Number}
    storage = Dense{ElT}(buf, undef, dim(inds))
    return itensor(storage, indices(inds))
end

function ITensor(
    buf::NDTensors.AllocBuffer, ::Type{ElT}, ::UndefInitializer, inds...
) where {ElT <: Number}
    return ITensor(buf, ElT, undef, indices(inds...))
end

function ITensor(buf::NDTensors.AllocBuffer, ::UndefInitializer, inds::Indices)
    return ITensor(buf, Float64, undef, inds)
end

function ITensor(buf::NDTensors.AllocBuffer, ::UndefInitializer, inds...)
    return ITensor(buf, Float64, undef, indices(inds...))
end

# -----------------------------------------------------------
# random_itensor(buf, inds) — buffer-allocated random
# -----------------------------------------------------------

function random_itensor(
    buf::NDTensors.AllocBuffer, rng::AbstractRNG, ::Type{S}, is::Indices
) where {S <: Number}
    T = ITensor(buf, S, is)
    randn!(rng, T)
    return T
end

function random_itensor(
    buf::NDTensors.AllocBuffer, ::Type{S}, is::Indices
) where {S <: Number}
    return random_itensor(buf, Random.default_rng(), S, is)
end

function random_itensor(
    buf::NDTensors.AllocBuffer, rng::AbstractRNG, ::Type{S}, is...
) where {S <: Number}
    return random_itensor(buf, rng, S, indices(is...))
end

function random_itensor(buf::NDTensors.AllocBuffer, ::Type{S}, is...) where {S <: Number}
    return random_itensor(buf, Random.default_rng(), S, indices(is...))
end

function random_itensor(buf::NDTensors.AllocBuffer, rng::AbstractRNG, is::Indices)
    return random_itensor(buf, rng, Float64, is)
end

function random_itensor(buf::NDTensors.AllocBuffer, is::Indices)
    return random_itensor(buf, Random.default_rng(), Float64, is)
end

function random_itensor(buf::NDTensors.AllocBuffer, rng::AbstractRNG, is...)
    return random_itensor(buf, rng, Float64, indices(is...))
end

function random_itensor(buf::NDTensors.AllocBuffer, is...)
    return random_itensor(buf, Random.default_rng(), Float64, indices(is...))
end

# -----------------------------------------------------------
# itensor alias (AllowAlias)
# -----------------------------------------------------------

itensor(buf::NDTensors.AllocBuffer, args...) = ITensor(buf, args...)
