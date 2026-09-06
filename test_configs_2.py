"""Compatibility launcher for spectral-WIP configurations."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("experiments.configs.spectral_wip", run_name="__main__")
else:
    import sys
    from experiments.configs import spectral_wip as _implementation

    sys.modules[__name__] = _implementation
