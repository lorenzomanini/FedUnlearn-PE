"""Compatibility launcher for the `test_<n>` result processor."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("analysis.results", run_name="__main__")
else:
    import sys
    from analysis import results as _implementation

    sys.modules[__name__] = _implementation
