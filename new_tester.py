"""Compatibility launcher for the revised-diagonal experiment generation."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("experiments.runner", run_name="__main__")
else:
    import sys
    from experiments import runner as _implementation

    sys.modules[__name__] = _implementation
