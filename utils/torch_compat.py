import torch

_PATCHED = False


def patch_transformers_custom_op_registration() -> None:
    """
    Patch torch custom-op registration to tolerate known transformers/torch mismatch.

    Some transformers builds register fallback MoE custom ops with postponed type
    annotations that older torch versions reject during schema inference. For
    non-MoE models in this project, skipping that registration is safe and avoids
    startup crashes.
    """
    global _PATCHED
    if _PATCHED:
        return

    original_custom_op = torch.library.custom_op
    original_register_fake = torch.library.register_fake
    original_register_autograd = torch.library.register_autograd

    def _safe_custom_op(name, fn=None, /, *args, **kwargs):
        if fn is None:
            def _decorator(inner_fn):
                return _safe_custom_op(name, inner_fn, *args, **kwargs)
            return _decorator

        try:
            return original_custom_op(name, fn, *args, **kwargs)
        except ValueError as e:
            msg = str(e)
            if "infer_schema(func)" in msg and "unsupported type torch.Tensor" in msg:
                return fn
            raise

    def _safe_register_fake(op_name, fn=None, /, *args, **kwargs):
        if fn is None:
            def _decorator(inner_fn):
                return _safe_register_fake(op_name, inner_fn, *args, **kwargs)
            return _decorator
        try:
            return original_register_fake(op_name, fn, *args, **kwargs)
        except (RuntimeError, AttributeError) as e:
            # Happens when custom_op registration was skipped due to schema incompatibility.
            msg = str(e)
            if "does not exist" in msg or "has no attribute" in msg:
                return fn
            raise

    def _safe_register_autograd(op_name, *args, **kwargs):
        try:
            return original_register_autograd(op_name, *args, **kwargs)
        except (RuntimeError, AttributeError) as e:
            msg = str(e)
            if "does not exist" in msg or "has no attribute" in msg:
                return None
            raise

    torch.library.custom_op = _safe_custom_op
    torch.library.register_fake = _safe_register_fake
    torch.library.register_autograd = _safe_register_autograd
    _PATCHED = True
