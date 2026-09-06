"""Compatibility launcher for the `iter_<n>` result processor."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("analysis.experiments_results", run_name="__main__")
else:
    import sys
    from analysis import experiments_results as _implementation

    sys.modules[__name__] = _implementation
