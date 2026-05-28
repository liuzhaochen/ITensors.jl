# [FORK] Bumper.jl buffer context manager
# Provides task-local storage for an active allocation buffer,
# and re-exports Bumper + UnsafeArray for buffer-aware storage types.
# See .claude/dense_itensor.yaml and .claude/mission.md for context.

import Bumper: Bumper
import Bumper.UnsafeArrays: UnsafeArray

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
# Fallback for non-constructible DenseArray types
# -----------------------------------------------------------
# promote_type(Vector{T}, UnsafeArray{T,1}) = DenseVector{T}
# which is abstract and can't be constructed via similar.
# Make it concrete by filling in default type parameters.

# UnsafeArray can only be created via Bumper.alloc!, not via similar(undef).
# Fall back to Vector when generic_zeros tries to create one.
function generic_zeros(
    arraytype::Type{<:Bumper.UnsafeArrays.UnsafeArray}, dims::Integer
)
    return generic_zeros(Vector{eltype(arraytype)}, dims)
end

# Fix promote_type for UnsafeArray: promote to Vector or UnsafeArray,
# never to abstract DenseVector/DenseMatrix.
function Base.promote_type(
    ::Type{UA}, ::Type{UA}
) where {T, N, UA <: Bumper.UnsafeArrays.UnsafeArray{T, N}}
    return UA
end
function Base.promote_type(
    ::Type{<:Bumper.UnsafeArrays.UnsafeArray{T}}, ::Type{A}
) where {T, A <: AbstractArray{T}}
    return A
end
function Base.promote_type(
    ::Type{A}, ::Type{<:Bumper.UnsafeArrays.UnsafeArray{T}}
) where {T, A <: AbstractArray{T}}
    return A
end
