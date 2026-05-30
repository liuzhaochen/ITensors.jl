# TBLIS-accelerated DenseTensor contraction
#
# Uses tblis_tensor / tblis_tensor_mult from the modern TBLIS.jl API.
# Supports Float32, Float64, ComplexF32, ComplexF64 natively.

using TBLIS: tblis_tensor, tblis_tensor_mult

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
    # 1D block data from BlockSparse: reshape to logical dimensions
    tdims = ntuple(d -> dim(block, d), N)
    return Base.unsafe_wrap(Array{ElT}, pointer(a), tdims; own=false)
end

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
) where {ElT <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    # tblis_tensor wraps a strided array as a TBLIS tensor, storing the
    # scaling factors α/β internally. No data copy — zero-cost view.
    # Use _tblis_strided_array to ensure N-D strides for block-sparse blocks.
    R_t = tblis_tensor(_tblis_strided_array(R), β)
    T1_t = tblis_tensor(_tblis_strided_array(T1), α)
    T2_t = tblis_tensor(_tblis_strided_array(T2))

    function label_to_char(label)
        char_start = label < 0 ? Char(123) : Char(96)
        return char_start + label
    end

    labelsT1_s = prod(label_to_char.(labelsT1))
    labelsT2_s = prod(label_to_char.(labelsT2))
    labelsR_s = prod(label_to_char.(labelsR))

    tblis_tensor_mult(T1_t, labelsT1_s, T2_t, labelsT2_s, R_t, labelsR_s)
    return R
end
