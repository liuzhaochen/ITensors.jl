# [FORK] Buffer-aware Tensor constructors
# Convenience constructors for creating Tensors with buffer-allocated
# Dense storage. The hot path (contraction outputs) is handled by
# buffer-aware similar in tensorstorage_buffer.jl.
# See .claude/bumper_api.yaml and .claude/mission.md for context.

"""
    Tensor(buf, inds::Tuple)

Create a `Tensor` with storage allocated from Bumper buffer `buf`.
Uses the default element type and `Dense` storage.
"""
function Tensor(buf::AllocBuffer, inds::Tuple)
    store = Dense(buf, dim(inds))
    return Tensor(AllowAlias(), store, inds)
end

"""
    Tensor(buf, ::UndefInitializer, inds::Tuple)

Create an uninitialized `Tensor` with buffer-allocated storage.
"""
function Tensor(buf::AllocBuffer, ::UndefInitializer, inds::Tuple)
    store = Dense(buf, undef, dim(inds))
    return Tensor(AllowAlias(), store, inds)
end
