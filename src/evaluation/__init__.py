"""Evaluation module entrypoints with lazy imports."""


def step3_longyaal(*args, **kwargs):
    from .eval import step3_longyaal as impl

    return impl(*args, **kwargs)


__all__ = ["step3_longyaal"]
