"""Auto-loaded by Python in every process when PYTHONPATH includes
this directory. Triggers the gap-trace patch on the scheduler.

Only activates if SGLANG_GAP_TRACE_LOG is set.
"""
import os

if os.environ.get("SGLANG_GAP_TRACE_LOG"):
    # Defer until sglang is importable; the actual install of the
    # monkey-patch happens lazily when the scheduler module is loaded.
    import importlib
    import sys

    _orig_find_spec = None

    class _Loader:
        """Trigger scheduler_gap_patch.install() right after
        sglang.srt.managers.scheduler is imported (which is what
        defines the Scheduler class)."""

        def find_spec(self, name, path, target=None):
            if name == "sglang.srt.managers.scheduler":
                # Let the real loader load it first, then patch.
                import importlib.util
                spec = importlib.util.find_spec(name)
                if spec is not None:
                    orig_exec = spec.loader.exec_module

                    def patched_exec(module):
                        orig_exec(module)
                        try:
                            import scheduler_gap_patch  # noqa: F401
                        except Exception as e:
                            print(f"[sitecustomize] gap patch failed: {e}")

                    spec.loader.exec_module = patched_exec
                return spec
            return None

    sys.meta_path.insert(0, _Loader())
