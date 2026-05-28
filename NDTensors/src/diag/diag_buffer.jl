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
