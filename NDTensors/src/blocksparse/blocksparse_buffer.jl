# [FORK] Buffer-aware BlockSparse storage extensions
# When a Bumper buffer is active, BlockSparse data is allocated from
# the buffer instead of the heap. The blockoffsets (small metadata)
# remain heap-allocated — only the contiguous data vector is buffered.
# See .claude/dense_itensor.yaml and .claude/mission.md for context.

# BlockSparse{<:UnsafeArray} constructors always require an active buffer.
function BlockSparse(
    ::Type{<:UnsafeArray{T}}, ::UndefInitializer,
    blockoffsets::BlockOffsets, dim::Integer; vargs...
) where {T}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    return BlockSparse(Bumper.alloc!(buf, T, dim), blockoffsets)
end

function BlockSparse(
    ::Type{<:UnsafeArray{T}},
    blockoffsets::BlockOffsets,
    dim::Integer;
    vargs...
) where {T}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    data = Bumper.alloc!(buf, T, dim)
    fill!(data, zero(T))
    return BlockSparse(data, blockoffsets; vargs...)
end

# DiagBlockSparse also propagates UnsafeArray from inputs.
function DiagBlockSparse(
    ::Type{<:UnsafeArray{T}}, ::UndefInitializer,
    boffs::BlockOffsets, diaglength::Integer
) where {T}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    return DiagBlockSparse(Bumper.alloc!(buf, T, diaglength), boffs)
end

"""
    BlockSparse(buf, blockoffsets::BlockOffsets, dim::Integer)

Create a `BlockSparse` with data allocated from Bumper buffer `buf`.
The element type is inferred from the context or defaults to `Float64`.
"""
function BlockSparse(
    buf::AllocBuffer, blockoffsets::BlockOffsets{N}, dim::Integer
) where {N}
    data = Bumper.alloc!(buf, Float64, dim)
    return BlockSparse(data, blockoffsets)
end

function BlockSparse(
    ::Type{ElT}, buf::AllocBuffer, blockoffsets::BlockOffsets{N}, dim::Integer
) where {ElT, N}
    data = Bumper.alloc!(buf, ElT, dim)
    return BlockSparse(data, blockoffsets)
end

function copy(B::BlockSparse{ElT, <:UnsafeArray}) where {ElT}
    buf = get_alloc_buffer()
    if buf !== nothing
        new_data = Bumper.alloc!(buf, ElT, length(data(B)))
        copyto!(new_data, data(B))
        return BlockSparse(new_data, copy(blockoffsets(B)))
    end
    return BlockSparse(copy(Vector(data(B))), copy(blockoffsets(B)))
end

function Base.conj(::AllowAlias, B::BlockSparse{ElT, <:UnsafeArray}) where {ElT}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    result = BlockSparse(ElT, buf, blockoffsets(B), length(data(B)))
    map!(conj, data(result), data(B))
    return result
end

function Base.real(B::BlockSparse{Complex{ElT}, <:UnsafeArray}) where {ElT}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    result = BlockSparse(ElT, buf, blockoffsets(B), length(data(B)))
    map!(real, data(result), data(B))
    return result
end

function Base.imag(B::BlockSparse{Complex{ElT}, <:UnsafeArray}) where {ElT}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    result = BlockSparse(ElT, buf, blockoffsets(B), length(data(B)))
    map!(imag, data(result), data(B))
    return result
end

function Base.:*(
    x::Number, T::Tensor{ElT, N, <:BlockSparse{ElT, <:UnsafeArray}}
) where {ElT, N}
    B = storage(T)
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    new_B = BlockSparse(ElT, buf, blockoffsets(B), length(data(B)))
    map!(z -> x * z, data(new_B), data(B))
    return Tensor(AllowAlias(), new_B, inds(T))
end

Base.:*(
    T::Tensor{ElT, N, <:BlockSparse{ElT, <:UnsafeArray}}, x::Number
) where {ElT, N} = x * T

function Base.:/(
    T::Tensor{ElT, N, <:BlockSparse{ElT, <:UnsafeArray}}, x::Number
) where {ElT, N}
    B = storage(T)
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    new_B = BlockSparse(ElT, buf, blockoffsets(B), length(data(B)))
    map!(z -> z / x, data(new_B), data(B))
    return Tensor(AllowAlias(), new_B, inds(T))
end

# -----------------------------------------------------------
# Buffer-aware block extraction
# -----------------------------------------------------------

"""
    blockview(T::BlockSparseTensor{<:Any, <:BlockSparse{<:Any, <:UnsafeArray}}, bof::BlockOffset)

Extract a block as a pointer-offset `UnsafeArray` instead of a `SubArray{UnsafeArray}`.
This avoids deeply nested `ReshapedArray{SubArray{UnsafeArray}}` types in the
subsequent `_contract!` path — giving BLAS a clean `UnsafeArray{ElT, N}` type
instead of a wrapper chain that the compiler struggles to optimize.

Without this override, the generic `blockview` at blocksparsetensor.jl:346 uses
`@view data[...]` which creates a `SubArray{UnsafeArray}`, and downstream
`array()` → `reshape()` produces `ReshapedArray{SubArray{UnsafeArray}}`.
"""
function blockview(
    T::Tensor{ElT, N, BlockSparse{ElT, UnsafeArray{ElT, 1}, N}},
    bof::BlockOffset
) where {ElT, N}
    blockT, offsetT = bof
    blockdimsT = blockdims(T, blockT)
    ptr = pointer(data(storage(T)), offsetT + 1)
    return tensor(Dense(UnsafeArray(ptr, (prod(blockdimsT),))), blockdimsT)
end

"""
    similar(
        storagetype::Type{<:BlockSparse{ElT, UnsafeArray{ElT, 1}, N}},
        blockoffsets::BlockOffsets, dims::Tuple
    )

Allocate BlockSparse{UnsafeArray} storage. Requires an active buffer.
Always returns `BlockSparse{ElT, UnsafeArray{ElT, 1}, N}` — a single
concrete type, no Union, no heap fallback.
"""
function similar(
    storagetype::Type{<:BlockSparse{ElT, UnsafeArray{ElT, 1}, N}},
    blockoffsets::BlockOffsets,
    dims::Tuple
) where {ElT, N}
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    data = Bumper.alloc!(buf, ElT, nnz(blockoffsets, dims))
    return BlockSparse(data, blockoffsets)
end

"""
    to_buffer(T::BlockSparseTensor{ElT}, buf::AllocBuffer)

Copy a heap-backed `BlockSparseTensor` into a buffer-allocated one.
Returns a new `BlockSparseTensor` with `UnsafeArray` storage.
Block offsets (metadata) remain heap-allocated.
"""
function to_buffer(T::BlockSparseTensor{ElT, N}, buf::AllocBuffer) where {ElT, N}
    bo = copy(blockoffsets(T))
    d = data(storage(T))
    buf_data = Bumper.alloc!(buf, ElT, length(d))
    copyto!(buf_data, d)
    return tensor(BlockSparse(buf_data, bo), inds(T))
end
