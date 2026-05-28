# [FORK] Buffer-aware similar dispatches for Dense storage
# When a Bumper buffer is active, `similar` allocates Dense storage
# from the buffer instead of the heap. Falls back to heap-allocated
# Vector-backed Dense when no buffer is active (avoids dispatching
# similar on UnsafeArray, which doesn't support index-based construction).
# See .claude/bumper_api.yaml and .claude/mission.md for context.

# --- Instance-level methods (dispatches on a Dense instance) ---

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

# --- Type-level methods (dispatches on Type{<:Dense}) ---
# Used by contraction_output → NDTensors.similar(tensortype, indsR)
# → similar(storagetype(tensortype), dims).
#
# Only use buffer allocation when the storage type has a concrete data
# type parameter. When promote_type(Vector, UnsafeArray) gives an
# abstract DenseVector/DenseMatrix, fall back to heap allocation
# (since we cannot construct UnsafeArray-backed storage that would
# satisfy the abstract type parameter constraint).

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
