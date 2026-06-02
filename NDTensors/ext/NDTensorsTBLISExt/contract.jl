# TBLIS-accelerated DenseTensor contraction
#
# Uses tblis_tensor / tblis_tensor_mult from the modern TBLIS.jl API.
# Supports Float32, Float64 natively (fast).
#
# ComplexF64 via tblis_compute_4real! — TBLIS complex kernels are ~16x slower
# than BLAS. Instead, decompose into 4 real TBLIS calls on zero-copy strided
# Float64 views of the complex data. Pre-allocated D, S arrays avoid heap allocs.

using TBLIS: tblis_tensor, tblis_tensor_mult

# ---------------------------------------------------------------------------
# Label conversion
# ---------------------------------------------------------------------------
function _tblis_labels(labels)
    isempty(labels) && return ""
    return prod(labels) do label
        Char((label < 0 ? 123 : 96) + label)
    end
end

# ---------------------------------------------------------------------------
# Strided array helper
# ---------------------------------------------------------------------------
"""
    _tblis_strided_array(block::DenseTensor)

Return the tensor data as a strided N-D array matching the logical tensor
dimensions. BlockSparse blocks store data as 1D flat vectors, but TBLIS
needs N-D arrays for correct stride computation. This is a zero-copy view
via `Base.unsafe_wrap` — no data copy.
"""
function _tblis_strided_array(block::DenseTensor{ElT, N}) where {ElT, N}
    a = array(block)
    ndims(a) == N && return a
    tdims = ntuple(d -> dim(block, d), N)
    return Base.unsafe_wrap(Array{ElT}, pointer(a), tdims; own=false)
end

# ---------------------------------------------------------------------------
# Zero-copy strided Float64 view of real/imag parts of ComplexF64 data
# ---------------------------------------------------------------------------
# ComplexF64 stores (r,i) interleaved. A strided Float64 view with stride 2
# in the first dimension extracts real or imag without copying.

struct _RealView{T, N} <: DenseArray{T, N}
    ptr::Ptr{T}
    dims::NTuple{N, Int}
    strides::NTuple{N, Int}
end

Base.size(v::_RealView) = v.dims
Base.size(v::_RealView, d) = v.dims[d]
Base.strides(v::_RealView) = v.strides
Base.elsize(::Type{<:_RealView{T}}) where {T} = sizeof(T)
Base.unsafe_convert(::Type{Ptr{T}}, v::_RealView{T}) where {T} = v.ptr

function _real_imag_views(a::AbstractArray{ComplexF64})
    sz = size(a)
    str = ntuple(d -> 2 * Base.stride(a, d), ndims(a))
    ptr = pointer(a)
    return _RealView{Float64, ndims(a)}(ptr, sz, str),
           _RealView{Float64, ndims(a)}(ptr + sizeof(Float64), sz, str)
end

# ---------------------------------------------------------------------------
# Float32 / Float64 — direct TBLIS (fast)
# ---------------------------------------------------------------------------
function contract!(
    ::Val{:TBLIS},
    R::DenseTensor{ElT},
    labelsR,
    T1::DenseTensor{ElT},
    labelsT1,
    T2::DenseTensor{ElT},
    labelsT2,
    α::ElT,
    β::ElT,
) where {ElT <: Union{Float32, Float64}}
    R_t = tblis_tensor(_tblis_strided_array(R), β)
    T1_t = tblis_tensor(_tblis_strided_array(T1), α)
    T2_t = tblis_tensor(_tblis_strided_array(T2))

    labelsT1_s = _tblis_labels(labelsT1)
    labelsT2_s = _tblis_labels(labelsT2)
    labelsR_s = _tblis_labels(labelsR)

    tblis_tensor_mult(T1_t, labelsT1_s, T2_t, labelsT2_s, R_t, labelsR_s)
    return R
end

# ---------------------------------------------------------------------------
# ComplexF64 — 4-real computation into pre-allocated D, S
# ---------------------------------------------------------------------------
# Called from _block_contract_complex! in contract_threaded_buffer.jl.
# D, S are Float64 arrays pre-allocated from the buffer (thread-safe because
# allocation happens before @spawn, in the single-threaded parent).
#
# Compute into D: D = Re(T1)∘Re(T2) - Im(T1)∘Im(T2)
# Compute into S: S = Re(T1)∘Im(T2) + Im(T1)∘Re(T2)
#
# The caller then combines D, S into the complex output with α, β.
function tblis_compute_4real!(
    D::StridedArray{Float64},
    S::StridedArray{Float64},
    T1::DenseTensor{ComplexF64},
    labelsT1,
    T2::DenseTensor{ComplexF64},
    labelsT2,
    labelsR,
)
    aT1 = _tblis_strided_array(T1)
    aT2 = _tblis_strided_array(T2)

    T1r, T1i = _real_imag_views(aT1)
    T2r, T2i = _real_imag_views(aT2)

    labT1 = _tblis_labels(labelsT1)
    labT2 = _tblis_labels(labelsT2)
    labR = _tblis_labels(labelsR)

    # D = T1r·T2r (overwrite)
    tblis_tensor_mult(tblis_tensor(T1r, 1.0), labT1,
                      tblis_tensor(T2r), labT2,
                      tblis_tensor(D, 0.0), labR)
    # D -= T1i·T2i (accumulate with α=-1)
    tblis_tensor_mult(tblis_tensor(T1i, -1.0), labT1,
                      tblis_tensor(T2i), labT2,
                      tblis_tensor(D, 1.0), labR)

    # S = T1r·T2i (overwrite)
    tblis_tensor_mult(tblis_tensor(T1r, 1.0), labT1,
                      tblis_tensor(T2i), labT2,
                      tblis_tensor(S, 0.0), labR)
    # S += T1i·T2r (accumulate)
    tblis_tensor_mult(tblis_tensor(T1i, 1.0), labT1,
                      tblis_tensor(T2r), labT2,
                      tblis_tensor(S, 1.0), labR)

    return nothing
end

# ---------------------------------------------------------------------------
# ComplexF64 × BlasReal — 2-real TBLIS (Im(BlasReal) = 0)
# ---------------------------------------------------------------------------
function tblis_compute_4real!(
    D::StridedArray{Float64},
    S::StridedArray{Float64},
    T1::DenseTensor{ComplexF64},
    labelsT1,
    T2::DenseTensor{<:LinearAlgebra.BlasReal},
    labelsT2,
    labelsR,
)
    aT1 = _tblis_strided_array(T1)
    aT2 = _tblis_strided_array(T2)
    T1r, T1i = _real_imag_views(aT1)

    labT1 = _tblis_labels(labelsT1)
    labT2 = _tblis_labels(labelsT2)
    labR = _tblis_labels(labelsR)

    # D = T1r·T2 (overwrite)
    tblis_tensor_mult(tblis_tensor(T1r, 1.0), labT1,
                      tblis_tensor(aT2), labT2,
                      tblis_tensor(D, 0.0), labR)
    # S = T1i·T2 (overwrite)
    tblis_tensor_mult(tblis_tensor(T1i, 1.0), labT1,
                      tblis_tensor(aT2), labT2,
                      tblis_tensor(S, 0.0), labR)

    return nothing
end

# ---------------------------------------------------------------------------
# BlasReal × ComplexF64 — 2-real TBLIS (Im(BlasReal) = 0)
# ---------------------------------------------------------------------------
function tblis_compute_4real!(
    D::StridedArray{Float64},
    S::StridedArray{Float64},
    T1::DenseTensor{<:LinearAlgebra.BlasReal},
    labelsT1,
    T2::DenseTensor{ComplexF64},
    labelsT2,
    labelsR,
)
    aT2 = _tblis_strided_array(T2)
    aT1 = _tblis_strided_array(T1)
    T2r, T2i = _real_imag_views(aT2)

    labT1 = _tblis_labels(labelsT1)
    labT2 = _tblis_labels(labelsT2)
    labR = _tblis_labels(labelsR)

    # D = T1·T2r (overwrite)
    tblis_tensor_mult(tblis_tensor(aT1, 1.0), labT1,
                      tblis_tensor(T2r), labT2,
                      tblis_tensor(D, 0.0), labR)
    # S = T1·T2i (overwrite)
    tblis_tensor_mult(tblis_tensor(aT1, 1.0), labT1,
                      tblis_tensor(T2i), labT2,
                      tblis_tensor(S, 0.0), labR)

    return nothing
end

# ComplexF32 repeats the same pattern. Currently not implemented
# (no benchmark demand); add when needed.
