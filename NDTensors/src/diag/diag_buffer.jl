# [FORK] Buffer-aware Diag storage extensions
# When a Bumper buffer is active, Diag can use UnsafeArray backing
# for non-uniform diagonal data.

# -----------------------------------------------------------
# Buffer-aware NonuniformDiag constructor
# -----------------------------------------------------------

"""
    Diag{ElT}(buf, dim::Integer)

Create a `NonuniformDiag` with diagonal data allocated from Bumper
buffer `buf`. Returns `Diag{ElT, UnsafeArray{ElT, 1}}`.
"""
function Diag{ElT}(buf::AllocBuffer, dim::Integer) where {ElT}
    data = Bumper.alloc!(buf, ElT, dim)
    return Diag{ElT, typeof(data)}(data)
end

function Base.:/(D::NonuniformDiag{<:Any, <:UnsafeArray}, x::Number)
    buf = get_alloc_buffer()
    if buf === nothing
        error(_NO_BUFFER_MSG)
    end
    out = Diag{eltype(D)}(buf, length(D))
    out .= data(D) ./ x
    return out
end

"""
    to_buffer(T::DiagTensor{ElT}, buf::AllocBuffer)

Copy a heap-backed `DiagTensor` into a buffer-allocated one.
Returns a new `DiagTensor` with `UnsafeArray` storage.
Only works for non-uniform (vector-backed) diagonal tensors.
"""
function to_buffer(T::Tensor{ElT, N, <:NonuniformDiag{ElT}}, buf::AllocBuffer) where {ElT, N}
    d = data(storage(T))
    buf_data = Bumper.alloc!(buf, ElT, length(d))
    copyto!(buf_data, d)
    return tensor(Diag{ElT, typeof(buf_data)}(buf_data), inds(T))
end
