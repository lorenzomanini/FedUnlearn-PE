"""Compatibility launcher for revised-diagonal configurations."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("experiments.configs.revised_diagonal", run_name="__main__")
else:
    import sys
    from experiments.configs import revised_diagonal as _implementation

    sys.modules[__name__] = _implementation
