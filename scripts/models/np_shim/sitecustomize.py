# runtime-only aliases for the scipy/numpy mismatch in the vllm_g4 env (env itself untouched)
import numpy as _np
if not hasattr(_np, "long"):
    _np.long = _np.int64
if not hasattr(_np, "ulong"):
    _np.ulong = _np.uint64
