"""Model backends. A backend exposes a model as a stack of layers that can be
loaded in contiguous slices, plus an embedding front and a head/norm tail."""

from .base import ModelBackend, PatchRuntime  # noqa: F401
