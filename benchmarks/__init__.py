"""BindFM Benchmarks Package — lazy imports"""

def __getattr__(name):
    if name in ("BindFMEvaluator", "BenchmarkResult"):
        import benchmarks.evaluate as _e
        return getattr(_e, name)
    raise AttributeError(f"module 'benchmarks' has no attribute {name!r}")
