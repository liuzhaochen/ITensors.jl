# [FORK] Bumper.jl-allocated Dense storage extensions
# Buffer-aware Dense constructors and operations using Bumper.jl.
# Requires Bumper + UnsafeArray imports from bufferalloc.jl.
# See .claude/dense_itensor.yaml, .claude/mission.md, and
# .claude/bumper_api.yaml for context.

# -----------------------------------------------------------
# Buffer-aware constructors
# -----------------------------------------------------------

"""
    Dense{ElT}(buf, dim::Integer)

Allocate Dense storage of length `dim` from Bumper buffer `buf`.
Returns `Dense{ElT, UnsafeArray{ElT, 1}}` — the data lives in
the buffer arena and is valid until the buffer is reset.
"""
function Dense{ElT}(buf::AllocBuffer, dim::Integer) where {ElT}
    data = Bumper.alloc!(buf, ElT, dim)
    return Dense{ElT, typeof(data)}(data)
end

"""
    Dense(buf, dim::Integer)

Allocate `Dense{Float64}` storage from Bumper buffer `buf`.
"""
function Dense(buf::AllocBuffer, dim::Integer)
    return Dense{default_eltype()}(buf, dim)
end

"""
    Dense{ElT}(buf, ::UndefInitializer, dim::Integer)

Uninitialized Dense storage from buffer. Equivalent to `Dense{ElT}(buf, dim)`.
"""
function Dense{ElT}(buf::AllocBuffer, ::UndefInitializer, dim::Integer) where {ElT}
    return Dense{ElT}(buf, dim)
end

"""
    Dense(buf, ::UndefInitializer, dim::Integer)

Uninitialized Dense storage from buffer, default eltype.
"""
function Dense(buf::AllocBuffer, ::UndefInitializer, dim::Integer)
    return Dense{default_eltype()}(buf, dim)
end

# -----------------------------------------------------------
# Buffer-aware copy — requires buffer for UnsafeArray
# -----------------------------------------------------------

"""
    copy(D::Dense{ElT, <:UnsafeArray})

Copy buffer-allocated Dense. When a buffer is active, the copy lives
in the buffer; otherwise falls back to heap allocation for extracting
data out of the buffer arena.
"""
function copy(D::Dense{ElT, UA}) where {ElT, UA <: UnsafeArray}
    buf = get_alloc_buffer()
    if buf !== nothing
        new_data = Bumper.alloc!(buf, ElT, length(data(D)))
        copyto!(new_data, data(D))
        return Dense{ElT, typeof(new_data)}(new_data)
    end
    return Dense(copy(Vector(data(D))))
end

# -----------------------------------------------------------
# Buffer-aware element-wise ops — all require buffer for UnsafeArray
# -----------------------------------------------------------

function conj(::AllowAlias, S::Dense{<:Any, <:UnsafeArray})
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    out = Dense{eltype(S)}(buf, length(S))
    out .= conj.(data(S))
    return out
end

function Base.real(S::Dense{<:Any, <:UnsafeArray})
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    out = Dense{real(eltype(S))}(buf, length(S))
    out .= real.(data(S))
    return out
end

function Base.imag(S::Dense{<:Any, <:UnsafeArray})
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    out = Dense{real(eltype(S))}(buf, length(S))
    out .= imag.(data(S))
    return out
end

function Base.complex(S::Dense{<:Any, <:UnsafeArray})
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    out = Dense{complex(eltype(S))}(buf, length(S))
    out .= complex.(data(S))
    return out
end

function -(S::Dense{<:Any, <:UnsafeArray})
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    out = Dense{eltype(S)}(buf, length(S))
    out .= .-data(S)
    return out
end

function (S::Dense{<:Any, <:UnsafeArray} * x::Number)
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    out = Dense{eltype(S)}(buf, length(S))
    out .= data(S) .* x
    return out
end

(x::Number * S::Dense{<:Any, <:UnsafeArray}) = S * x

function Base.:/(S::Dense{<:Any, <:UnsafeArray}, x::Number)
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    out = Dense{eltype(S)}(buf, length(S))
    out .= data(S) ./ x
    return out
end
