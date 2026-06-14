# transitional alias-shim (refactor #251 Stage A). Canonical home:
# sglang.srt.mem_cache.aginfer.sglang_adapter. Removed in Stage D (pure-policy) / kept
# as dev-research compat (value-model). Do not add logic here.
import sys as _sys
from sglang.srt.mem_cache.aginfer import sglang_adapter as _m
_sys.modules[__name__] = _m
