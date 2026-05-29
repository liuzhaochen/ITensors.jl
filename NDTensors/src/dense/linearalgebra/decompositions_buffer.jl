# [FORK] Buffer-aware SVD/QR for UnsafeArray-backed DenseTensor
# When a Bumper buffer is active, the output tensors (U, V from svd;
# Q, R from qr) are backed by UnsafeArrays allocated from the buffer.
# Downstream operations on these results can then also benefit from
# buffer allocation.
#
# These methods dispatch on Tensor{ElT, 2, Dense{ElT, <:UnsafeArray}},
# which is more specific than the original DenseTensor{ElT, 2, IndsT}
# methods. They only activate when the input tensor has UnsafeArray
# storage; regular Vector-backed tensors use the original methods.

# -----------------------------------------------------------
# Helper: create buffer-backed Dense from heap Vector
# -----------------------------------------------------------

"""
    _bufferize_dense(data::Vector{ElT}) -> Dense

If a Bumper buffer is active in the task-local context, copy `data`
into a `Dense{ElT}` allocated from the buffer. Otherwise, wrap `data`
directly (no copy, heap). Returns `Dense{ElT, <:UnsafeArray}` when
buffered, `Dense{ElT, Vector{ElT}}` when not.
"""
function _bufferize_dense(vecdata::AbstractVector{ElT}) where {ElT}
    buf = get_alloc_buffer()
    buf === nothing && return Dense(vecdata)
    result = Dense{ElT}(buf, length(vecdata))
    copyto!(data(result), vecdata)
    return result
end

function _bufferize_diag(vecdata::AbstractVector{ElT}) where {ElT}
    buf = get_alloc_buffer()
    buf === nothing && return Diag(vecdata)
    result = Diag{ElT}(buf, length(vecdata))
    copyto!(data(result), vecdata)
    return result
end

# -----------------------------------------------------------
# Buffer-aware 2D SVD (extends linearalgebra.jl svd)
# -----------------------------------------------------------

function svd(
    T::Tensor{ElT, 2, Dense{ElT, UA}};
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    use_absolute_cutoff=nothing,
    use_relative_cutoff=nothing,
    alg=nothing,
    min_blockdim=nothing,
) where {ElT, UA <: UnsafeArray}
    IndsT = indstype(T)

    # Use matrix(T) directly — UnsafeArray works with LAPACK via pointer.
    # Explicit fill of UnsafeArray-backed tensors requires setindex loops
    # (broadcast assignment may segfault due to UnsafeArray write semantics).
    M = matrix(T)

    # SVD algorithm selection (same logic as original svd)
    alg = replace_nothing(alg, default_svd_alg(T))
    if alg == "divide_and_conquer"
        MUSV = svd_catch_error(M; alg=LinearAlgebra.DivideAndConquer())
        if isnothing(MUSV)
            alg = "qr_iteration"
            MUSV = svd_catch_error(M; alg=LinearAlgebra.QRIteration())
            if isnothing(MUSV)
                alg = "recursive"
                MUSV = svd_recursive(M)
            end
        end
    elseif alg == "qr_iteration"
        MUSV = svd_catch_error(M; alg=LinearAlgebra.QRIteration())
        if isnothing(MUSV)
            alg = "recursive"
            MUSV = svd_recursive(M)
        end
    elseif alg == "recursive"
        MUSV = svd_recursive(M)
    elseif alg == "qr_algorithm" || alg == "jacobi_algorithm"
        MUSV = svd_catch_error(M; alg)
    else
        error(
            "svd algorithm $alg is not currently supported. " *
            "Please see the documentation for currently supported algorithms.",
        )
    end

    if isnothing(MUSV)
        if any(isnan, expose(T))
            println("SVD failed, the matrix you were trying to SVD contains NaNs.")
        else
            println(lapack_svd_error_message(alg))
        end
        return nothing
    end

    MU, MS, MV = MUSV
    conj!(MV)

    # Truncation
    P = MS .^ 2
    if any(!isnothing, (maxdim, cutoff))
        P, truncerr, _ = truncate!!(
            P; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
        )
    else
        truncerr = 0.0
    end
    spec = Spectrum(P, truncerr)
    dS = length(P)
    if dS < length(MS)
        MU = expose(MU)[:, 1:dS]
        MS = MS[1:dS]
        MV = expose(MV)[:, 1:dS]
    end

    # Construct output indices
    u = eltype(IndsT)(dS)
    v = eltype(IndsT)(dS)
    Uinds = IndsT((ind(T, 1), u))
    Sinds = IndsT((u, v))
    Vinds = IndsT((ind(T, 2), v))

    # Buffer-aware output wrapping
    U = tensor(_bufferize_dense(vec(MU)), Uinds)
    S = tensor(_bufferize_diag(MS), Sinds)
    V = tensor(_bufferize_dense(vec(MV)), Vinds)
    return U, S, V, spec
end

# -----------------------------------------------------------
# Buffer-aware 2D QR (extends linearalgebra.jl)
# -----------------------------------------------------------

function qx(qx::Function, T::Tensor{ElT, 2, Dense{ElT, UA}}) where {ElT, UA <: UnsafeArray}
    QM, XM = qx(expose(matrix(T)))
    q, r = inds(T)
    q = dim(q) < dim(r) ? sim(q) : sim(r)
    IndsT = indstype(T)
    Qinds = IndsT((ind(T, 1), q))
    Xinds = IndsT((q, ind(T, 2)))
    QM = convert(typeof(XM), QM)
    QM = convert(typeof(XM), QM)

    Q = tensor(_bufferize_dense(vec(QM)), Qinds)
    X = tensor(_bufferize_dense(vec(XM)), Xinds)
    return Q, X
end
