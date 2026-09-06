"""Compatibility launcher for legacy experiment configurations."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("experiments.configs.v1_legacy", run_name="__main__")
else:
    import sys
    from experiments.configs import v1_legacy as _implementation

    sys.modules[__name__] = _implementation
