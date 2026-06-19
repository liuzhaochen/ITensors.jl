# [FORK] Bumper.jl buffer context manager
# Provides task-local storage for an active allocation buffer,
# and re-exports Bumper + UnsafeArray for buffer-aware storage types.
# See .claude/dense_itensor.yaml and .claude/mission.md for context.

const ALLOC_BUFFER_KEY = gensym(:alloc_buffer)

"""
    get_alloc_buffer()

Return the currently active allocation buffer for this task,
or `nothing` if no buffer is set.
"""
get_alloc_buffer() = get(task_local_storage(), ALLOC_BUFFER_KEY, nothing)

"""
    set_alloc_buffer!(buf)

Set the active allocation buffer for the current task.
"""
set_alloc_buffer!(buf) = task_local_storage(ALLOC_BUFFER_KEY, buf)

"""
    with_alloc_buffer(f, buf)

Execute `f` with `buf` as the active allocation buffer,
restoring the previous buffer state on exit.
"""
function with_alloc_buffer(f, buf)
    prev = get_alloc_buffer()
    set_alloc_buffer!(buf)
    try
        f()
    finally
        set_alloc_buffer!(prev)
    end
end

"""
    AllocBuffer

Union type of all Bumper.jl buffer types: `SlabBuffer`, `ResizeBuffer`,
and `AllocBuffer`. Useful for dispatch on buffer-aware constructors.
"""
const AllocBuffer = Union{
    Bumper.SlabBufferImpl.SlabBuffer,
    Bumper.ResizeBufferImpl.ResizeBuffer,
    Bumper.AllocBufferImpl.AllocBuffer,
}

# -----------------------------------------------------------
# UnsafeArray: always requires buffer. No heap fallback.
# -----------------------------------------------------------

const _NO_BUFFER_MSG = "Cannot allocate UnsafeArray: no active buffer. " *
    "Wrap the code in `with_alloc_buffer(buf) do ... end` or call " *
    "NDTensors.set_alloc_buffer!(buf) first."

# NDTensors.similar for UnsafeArray — used by storage allocation chain.
# Requires an active buffer; errors otherwise.
function NDTensors.similar(::Type{UA}, dims::Dims) where {T, UA <: UnsafeArray{T}}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    return Bumper.alloc!(buf, T, dims...)
end

# Base.similar for UnsafeArray — used by permutedims and other Base functions.
# Instance-level: intercepts Base's similar(a::AbstractArray{T}, dims::Tuple).
function Base.similar(a::UnsafeArray{T, N}, dims::Dims) where {T, N}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    return Bumper.alloc!(buf, T, dims...)
end

# Type-level catch for direct similar(::Type{<:UnsafeArray}, dims) calls.
function Base.similar(::Type{UA}, dims::Dims) where {T, N, UA <: UnsafeArray{T, N}}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    return Bumper.alloc!(buf, T, dims...)
end

# generic_zeros for UnsafeArray — uses buffer when active, errors otherwise.
function generic_zeros(
    arraytype::Type{<:UnsafeArray{T}}, dims::Integer
) where {T}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    data = Bumper.alloc!(buf, T, dims)
    fill!(data, zero(T))
    return data
end

# Fix promote_type for UnsafeArray: promote to Vector or UnsafeArray,
# never to abstract DenseVector/DenseMatrix.
function Base.promote_type(
    ::Type{UA}, ::Type{UA}
) where {T, N, UA <: UnsafeArray{T, N}}
    return UA
end
function Base.promote_type(
    ::Type{<:UnsafeArray{T}}, ::Type{A}
) where {T, A <: AbstractArray{T}}
    return A
end
function Base.promote_type(
    ::Type{A}, ::Type{<:UnsafeArray{T}}
) where {T, A <: AbstractArray{T}}
    return A
end
