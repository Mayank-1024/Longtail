"""Label guard: the pipeline never reads the dataset's 3D labels (plan 2.1 and C1).

Labels are evaluation ground truth. Reading them raises LabelAccessError unless the caller is inside
``with eval_ground_truth():``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# ASSUMPTION A-15: the guard covers cuboids and semantic segmentation
# ASSUMPTION A-36: the guard is a context variable; threads started inside it do not inherit access
_depth: ContextVar[int] = ContextVar("longtail_label_access_depth", default=0)


class LabelAccessError(RuntimeError):
    """Raised when dataset labels are read outside eval_ground_truth()."""


@contextmanager
def eval_ground_truth() -> Iterator[None]:
    """Allow label reads for the duration of the block. Use only in eval/ ground-truth code."""
    token = _depth.set(_depth.get() + 1)
    try:
        yield
    finally:
        _depth.reset(token)


def labels_allowed() -> bool:
    return _depth.get() > 0


def require_label_access(what: str) -> None:
    if not labels_allowed():
        raise LabelAccessError(
            f"{what} are evaluation ground truth; read them only inside eval_ground_truth()"
        )
