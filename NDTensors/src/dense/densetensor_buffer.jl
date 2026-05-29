# [FORK] Buffer-aware DenseTensor constructors
# Convenience wrappers for creating DenseTensors with
# buffer-allocated Dense storage.
# See .claude/bumper_api.yaml and .claude/mission.md for context.

"""
    DenseTensor(buf, inds::Tuple)

Create a `DenseTensor` with storage allocated from Bumper buffer `buf`.
"""
function DenseTensor(buf::AllocBuffer, inds::Tuple)
    return tensor(Dense(buf, dim(inds)), inds)
end

"""
    DenseTensor(::Type{ElT}, buf, inds::Tuple)

Create a `DenseTensor{ElT}` with storage allocated from Bumper buffer `buf`.
"""
function DenseTensor(::Type{ElT}, buf::AllocBuffer, inds::Tuple) where {ElT}
    return tensor(Dense{ElT}(buf, dim(inds)), inds)
end

"""
    to_buffer(T::DenseTensor{ElT, N}, buf::AllocBuffer)

Copy a heap-backed `DenseTensor` into a buffer-allocated one.
Returns a new `DenseTensor` with `UnsafeArray` storage.
"""
function to_buffer(T::DenseTensor{ElT, N}, buf::AllocBuffer) where {ElT, N}
    d = data(storage(T))
    buf_data = Bumper.alloc!(buf, ElT, length(d))
    copyto!(buf_data, d)
    return tensor(Dense{ElT, typeof(buf_data)}(buf_data), inds(T))
end
