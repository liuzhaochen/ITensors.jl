# [FORK] Buffer-aware similar dispatches for Dense storage
# When a Bumper buffer is active, `similar` allocates Dense storage
# from the buffer instead of the heap. Dense{<:UnsafeArray} always
# requires an active buffer — no heap fallback.
# See .claude/bumper_api.yaml and .claude/mission.md for context.

# --- Instance-level methods (dispatches on a Dense instance) ---

function similar(storage::Dense{<:Any, <:UnsafeArray})
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    return Dense{eltype(storage)}(buf, length(storage))
end

function similar(storage::Dense{<:Any, <:UnsafeArray}, eltype::Type)
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    return Dense{eltype}(buf, length(storage))
end

function similar(storage::Dense{<:Any, <:UnsafeArray}, dims::Tuple)
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    return Dense{eltype(storage)}(buf, dim(dims))
end

function similar(storage::Dense{<:Any, <:UnsafeArray}, eltype::Type, dims::Tuple)
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    return Dense{eltype}(buf, dim(dims))
end

# --- Generic instance-level (heap fallback, Dense{<:Vector} or abstract) ---

function similar(storage::Dense)
    buf = get_alloc_buffer()
    if buf !== nothing
        return Dense{eltype(storage)}(buf, length(storage))
    end
    return Dense{eltype(storage)}(length(storage))
end

function similar(storage::Dense, eltype::Type)
    buf = get_alloc_buffer()
    if buf !== nothing
        return Dense{eltype}(buf, length(storage))
    end
    return Dense{eltype}(length(storage))
end

function similar(storage::Dense, dims::Tuple)
    buf = get_alloc_buffer()
    if buf !== nothing
        return Dense{eltype(storage)}(buf, dim(dims))
    end
    return Dense{eltype(storage)}(dim(dims))
end

function similar(storage::Dense, eltype::Type, dims::Tuple)
    buf = get_alloc_buffer()
    if buf !== nothing
        return Dense{eltype}(buf, dim(dims))
    end
    return Dense{eltype}(dim(dims))
end

# --- Type-level methods ---
# Used by contraction path: similar(storagetype(tensortype), dims).

function similar(storagetype::Type{<:Dense}, dims::Tuple)
    buf = get_alloc_buffer()
    if buf !== nothing && isconcretetype(datatype(storagetype))
        return Dense{eltype(storagetype)}(buf, dim(dims))
    end
    return Dense{eltype(storagetype)}(dim(dims))
end

function similar(storagetype::Type{<:Dense}, dims::Dims)
    buf = get_alloc_buffer()
    if buf !== nothing && isconcretetype(datatype(storagetype))
        return Dense{eltype(storagetype)}(buf, prod(dims))
    end
    return Dense{eltype(storagetype)}(prod(dims))
end

function similar(storagetype::Type{<:Dense}, eltype::Type, dims::Tuple)
    buf = get_alloc_buffer()
    if buf !== nothing && isconcretetype(datatype(storagetype))
        return Dense{eltype}(buf, dim(dims))
    end
    return Dense{eltype}(dim(dims))
end
