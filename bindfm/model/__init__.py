"""
BindFM model package.
Imports are lazy to avoid circular dependency issues and
allow syntax checking without a full torch install.
"""

def __getattr__(name):
    if name in ("BindFM", "BindFMConfig"):
        from model.bindfm import BindFM, BindFMConfig
        return {"BindFM": BindFM, "BindFMConfig": BindFMConfig}[name]
    if name in ("AtomFeatures", "MolecularGraph", "BindingPair",
                "ATOM_FEAT_DIM", "EntityType", "BondFeatures"):
        import model.tokenizer as _t
        return getattr(_t, name)
    if name in ("DualEntityEncoder", "AtomEncoder"):
        import model.encoder as _e
        return getattr(_e, name)
    if name in ("PairFormerTrunk",):
        import model.trunk as _tr
        return getattr(_tr, name)
    if name in ("AffinityHead", "StructureFlowMatchingHead",
                "GenerativeHead", "BindFMLoss"):
        import model.heads as _h
        return getattr(_h, name)
    raise AttributeError(f"module 'model' has no attribute {name!r}")
