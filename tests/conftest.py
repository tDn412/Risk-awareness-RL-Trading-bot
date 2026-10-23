"""Pytest configuration.

torch must be imported before pandas/pyarrow on Windows; otherwise loading its DLLs
fails with ``WinError 1114`` when a later test module imports torch.
"""

import torch  # noqa: F401
