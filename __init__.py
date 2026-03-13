"""
BindFM — Universal Binding Foundation Model
============================================

Architecture:
  tokenizer.py     — Universal atomic vocabulary and molecular graph representation
  encoder.py       — SE(3)-equivariant atom encoder (shared across all modalities)
  trunk.py         — PairFormer binding interaction trunk
  heads.py         — Affinity, structure, and generative output heads
  bindfm.py        — Complete assembled model

Training:
  training/train.py — 4-stage curriculum training pipeline

Key properties:
  - Built entirely from scratch. Zero pretrained weights.
  - Atom-level representation: proteins, nucleic acids, small molecules
    all represented as the same atomic graph.
  - Universal affinity head: covers protein-small mol, protein-protein,
    protein-nucleic acid, nucleic-small mol, nucleic-nucleic binding.
  - SE(3)-equivariant: correct physics under rotation and translation.
  - Three co-trained heads: affinity + structure + generation.
"""

from model.bindfm import BindFM, BindFMConfig
from model.tokenizer import (
    AtomFeatures, BondFeatures, MolecularGraph, BindingPair,
    MolecularGraphBuilder, EntityType, ModificationType,
    StructuralContext, Hybridization, Chirality, BondType,
    ELEMENTS, ELEMENT_TO_IDX, N_ELEMENTS,
)
from model.encoder import AtomEncoder, DualEntityEncoder
from model.trunk import PairFormerTrunk
from model.heads import AffinityHead, StructureFlowMatchingHead, GenerativeHead

__version__ = "0.1.0-dev"
__all__ = [
    "BindFM", "BindFMConfig",
    "AtomFeatures", "BondFeatures", "MolecularGraph", "BindingPair",
    "MolecularGraphBuilder", "EntityType", "ModificationType",
    "AtomEncoder", "DualEntityEncoder", "PairFormerTrunk",
    "AffinityHead", "StructureFlowMatchingHead", "GenerativeHead",
]
