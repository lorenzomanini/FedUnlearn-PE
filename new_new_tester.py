"""Compatibility launcher for the spectral-WIP experiment generation."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("experiments.spectral_runner", run_name="__main__")
else:
    import sys
    from experiments import spectral_runner as _implementation

    sys.modules[__name__] = _implementation
