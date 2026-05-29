# [FORK] Buffer-backed ITensor conversion
# Only provides to_buffer for converting heap ITensors to buffer-backed.
# Direct ITensor(buf, inds) constructors removed — use to_buffer instead.

"""
    to_buffer(A::ITensor, buf::NDTensors.AllocBuffer)

Copy an ITensor (Dense, Diag, or BlockSparse) into a buffer-allocated one.
The new ITensor retains the same indices and storage structure, but the
data lives in the buffer arena.
"""
function to_buffer(A::ITensor, buf::NDTensors.AllocBuffer)
    T = tensor(A)
    Tb = NDTensors.to_buffer(T, buf)
    return ITensor(NDTensors.AllowAlias(), Tb)
end
