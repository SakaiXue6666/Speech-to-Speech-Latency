"""Alignment module entrypoints with lazy imports."""


def step2_segale(*args, **kwargs):
    from .segale import step2_segale as impl

    return impl(*args, **kwargs)


__all__ = ["step2_segale"]
