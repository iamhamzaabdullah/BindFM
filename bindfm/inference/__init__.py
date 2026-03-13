"""BindFM Inference Package — lazy imports"""

def __getattr__(name):
    if name in ("BindFMPredictor", "AffinityResult", "StructureResult", "GeneratedBinder"):
        import inference.api as _a
        return getattr(_a, name)
    raise AttributeError(f"module 'inference' has no attribute {name!r}")
