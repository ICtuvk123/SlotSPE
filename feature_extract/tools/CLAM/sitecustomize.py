"""Local Python startup compatibility patches for CLAM scripts."""

try:
    import torch.utils._pytree as _torch_pytree

    if (
        not hasattr(_torch_pytree, "register_pytree_node")
        and hasattr(_torch_pytree, "_register_pytree_node")
    ):

        def register_pytree_node(node_type, flatten_fn, unflatten_fn, *args, **kwargs):
            return _torch_pytree._register_pytree_node(
                node_type,
                flatten_fn,
                unflatten_fn,
            )

        _torch_pytree.register_pytree_node = register_pytree_node
except Exception:
    pass
