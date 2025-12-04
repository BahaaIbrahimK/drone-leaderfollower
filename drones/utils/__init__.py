import numpy as np

# Provide backwards-compatible aliases for older NumPy names relied on by RLlib 1.x.
_np_dict = np.__dict__
if "object" not in _np_dict:
    np.object = object  # type: ignore[attr-defined]
if "bool" not in _np_dict:
    np.bool = bool  # type: ignore[attr-defined]
if "int" not in _np_dict:
    np.int = int  # type: ignore[attr-defined]
if "float" not in _np_dict:
    np.float = float  # type: ignore[attr-defined]
