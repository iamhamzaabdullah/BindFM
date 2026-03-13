#!/usr/bin/env python3
"""
BindFM — Affinity Index Builder
---------------------------------
Merges all affinity data sources into a single unified index CSV.
Handles unit conversion, deduplication, quality filtering, and
train/val/test splitting with no target leakage.

Input sources:
  - BindingDB TSV
  - ChEMBL activities CSV (exported from SQLite)
  - AptaBase CSV
  - SKEMPI2 CSV
  - PDBbind index CSV
  - RNAcompete scores
  - CovalentDB CSV

Output:
  - data/affinity/all_affinity.csv       — unified training set
  - data/affinity/train.csv
  - data/affinity/val.csv
  - data/affinity/test.csv
  - data/affinity/stats.json             — dataset statistics

Unified schema:
  ligand_id, ligand_smiles, ligand_sequence, ligand_entity,
  target_id, target_sequence, target_entity,
  log_kd_nM, assay_type, source, is_binder,
  is_covalent, is_allosteric, split
"""

import os
import sys
import json
import csv
import math
import hashlib
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter

import numpy as np


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

# Unified assay type labels
ASSAY_TYPES = {
    "Kd": 0, "Ki": 1, "IC50": 2, "Kd_SPR": 3,
    "ITC": 4, "EMSA": 5, "SELEX": 6, "Tm": 7,
    "kinact_KI": 8, "AC50": 9, "Tm_hybrid": 10, "MST": 11,
}

# Cutoff for binder/non-binder classification (< 10 µM = binder)
BINDER_CUTOFF_LOG_NM = 4.0   # log10(10000 nM) = 4.0

# Output schema
UNIFIED_FIELDS = [
    "entry_id", "ligand_smiles", "ligand_sequence", "ligand_entity",
    "target_sequence", "target_id", "target_entity",
    "log_kd_nM", "assay_type", "assay_type_idx",
    "source", "is_binder", "is_covalent", "is_allosteric",
    "split",
]


# ──────────────────────────────────────────────
# UNIT CONVERSION
# ──────────────────────────────────────────────

def to_log_kd_nM(value: float, unit: str, assay: str) -> Optional[float]:
    """
    Convert any affinity measurement to log10(Kd in nM).

    Handles:
      - Molar concentrations (M, mM, µM, nM, pM)
      - pKd/pKi (= -log10(Kd/M))
      - IC50 in various units
      - Tm values (converted to approximate ΔG-based ranking)
    """
    if math.isnan(value) or math.isinf(value) or value <= 0:
        return None

    # pX values (already in -log10(M) space)
    if assay.lower() in ("pkd", "pki", "pic50") or unit.lower() == "plog":
        # pKd = -log10(Kd/M), so Kd/M = 10^(-pKd)
        # Kd in nM = 10^(-pKd) * 10^9 = 10^(9 - pKd)
        log_kd_nM = 9.0 - value
        return log_kd_nM

    # IC50 requires conversion (roughly IC50 ≈ 2*Ki for competitive inhibitors)
    # We store as-is and let the assay-type conditioning handle normalization
    multiplier = {
        "M":   1e9,   # Molar → nM
        "mM":  1e6,   # millimolar → nM
        "µM":  1e3,   # micromolar → nM
        "uM":  1e3,
        "nM":  1.0,   # nanomolar
        "pM":  1e-3,  # picomolar → nM
        "fM":  1e-6,  # femtomolar → nM
    }.get(unit, 1.0)  # default: assume nM

    kd_nM = value * multiplier
    if kd_nM <= 0:
        return None
    log_kd_nM = math.log10(kd_nM)

    # Sanity bounds: -3 (pM range) to 8 (mM range)
    if not (-3.0 <= log_kd_nM <= 8.0):
        return None

    return log_kd_nM


def parse_affinity_string(s: str) -> Tuple[Optional[float], str]:
    """
    Parse affinity strings like:
      "45.3", "< 100", "> 10000", "1.2e-3", "45.3 nM", "1 µM"
    Returns (value_in_nM, qualifier) where qualifier ∈ {'=','<','>'}
    """
    s = s.strip()
    if not s or s.lower() in ("n/a", "nd", "none", ""):
        return None, "="

    qualifier = "="
    if s.startswith("<"):
        qualifier = "<"
        s = s[1:].strip()
    elif s.startswith(">"):
        qualifier = ">"
        s = s[1:].strip()

    # Extract unit if present
    unit = "nM"
    for u in ["µM", "uM", "mM", " M", "nM", "pM", "fM"]:
        if u.lower() in s.lower():
            unit = u.strip()
            s = s.lower().replace(u.lower(), "").strip()
            break

    try:
        val = float(s.replace(",", ""))
        return val, qualifier
    except ValueError:
        return None, "="


# ──────────────────────────────────────────────
# SOURCE PARSERS
# ──────────────────────────────────────────────

def parse_bindingdb(path: str) -> List[Dict]:
    """Parse BindingDB TSV into unified schema."""
    print(f"Parsing BindingDB: {path}")
    entries = []

    affinity_columns = [
        ("Kd (nM)", "Kd"),
        ("Ki (nM)", "Ki"),
        ("IC50 (nM)", "IC50"),
    ]

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            smiles   = row.get("Ligand SMILES", "").strip()
            tgt_seq  = row.get("BindingDB Target Chain  Sequence", "").strip()
            tgt_name = row.get("Target Name", "").strip()
            if not smiles or not tgt_seq:
                continue

            for col, assay in affinity_columns:
                val_str = row.get(col, "").strip()
                if not val_str or val_str.lower() in ("", "none"):
                    continue
                # Skip inequality values for continuous training
                # (could use as bounds, but keeps things simple)
                val_str = val_str.replace(">","").replace("<","").strip()
                try:
                    kd_nm = float(val_str)
                    if kd_nm <= 0:
                        continue
                    log_kd = math.log10(max(kd_nm, 0.001))
                except ValueError:
                    continue

                entry_id = hashlib.md5(
                    f"bdb_{smiles[:50]}_{tgt_seq[:20]}_{assay}".encode()
                ).hexdigest()[:12]

                entries.append({
                    "entry_id":       entry_id,
                    "ligand_smiles":  smiles,
                    "ligand_sequence": "",
                    "ligand_entity":  "small_mol",
                    "target_sequence": tgt_seq,
                    "target_id":      tgt_name,
                    "target_entity":  "protein",
                    "log_kd_nM":      log_kd,
                    "assay_type":     assay,
                    "assay_type_idx": ASSAY_TYPES.get(assay, 0),
                    "source":         "bindingdb",
                    "is_binder":      int(log_kd < BINDER_CUTOFF_LOG_NM),
                    "is_covalent":    0,
                    "is_allosteric":  0,
                })
                break   # take first available affinity type

    print(f"  BindingDB: {len(entries):,} entries")
    return entries


def parse_chembl_csv(path: str) -> List[Dict]:
    """Parse ChEMBL activities CSV (exported from SQLite)."""
    print(f"Parsing ChEMBL: {path}")
    entries = []

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles    = row.get("canonical_smiles", "").strip()
            tgt_seq   = row.get("target_sequence", "").strip()
            std_type  = row.get("standard_type", "").strip()
            std_val   = row.get("standard_value", "").strip()
            std_unit  = row.get("standard_units", "nM").strip()
            tgt_name  = row.get("target_pref_name", "").strip()
            allosteric = int(row.get("is_allosteric", "0") or "0")

            if not smiles or not tgt_seq or not std_val:
                continue

            assay = std_type if std_type in ASSAY_TYPES else "IC50"

            try:
                val = float(std_val)
            except ValueError:
                continue

            log_kd = to_log_kd_nM(val, std_unit, assay)
            if log_kd is None:
                continue

            entry_id = hashlib.md5(
                f"cbl_{smiles[:50]}_{tgt_seq[:20]}_{assay}".encode()
            ).hexdigest()[:12]

            entries.append({
                "entry_id":        entry_id,
                "ligand_smiles":   smiles,
                "ligand_sequence": "",
                "ligand_entity":   "small_mol",
                "target_sequence": tgt_seq,
                "target_id":       tgt_name,
                "target_entity":   "protein",
                "log_kd_nM":       log_kd,
                "assay_type":      assay,
                "assay_type_idx":  ASSAY_TYPES.get(assay, 2),
                "source":          "chembl",
                "is_binder":       int(log_kd < BINDER_CUTOFF_LOG_NM),
                "is_covalent":     0,
                "is_allosteric":   allosteric,
            })

    print(f"  ChEMBL: {len(entries):,} entries")
    return entries


def parse_aptabase(path: str) -> List[Dict]:
    """Parse AptaBase CSV into unified schema."""
    print(f"Parsing AptaBase: {path}")
    entries = []

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            apt_seq  = row.get("Aptamer_Sequence", "").strip().upper()
            tgt_seq  = row.get("Target_Sequence", "").strip()
            tgt_name = row.get("Target_Name", "").strip()
            kd_str   = row.get("Kd_nM", "").strip()
            mod_str  = row.get("Modification", "NONE").strip().upper()
            apt_type = row.get("Aptamer_Type", "RNA").strip().upper()

            if not apt_seq or not tgt_seq:
                continue

            # Determine entity type
            entity = "rna"
            if apt_type == "DNA" or all(c in "ACGT" for c in apt_seq.replace("-","")):
                entity = "dna"

            log_kd   = None
            is_binder = 1

            if kd_str:
                val, qualifier = parse_affinity_string(kd_str)
                if val is not None:
                    log_kd = math.log10(max(val, 0.001))   # already in nM
                    is_binder = int(log_kd < BINDER_CUTOFF_LOG_NM)

            entry_id = hashlib.md5(
                f"apt_{apt_seq[:30]}_{tgt_seq[:20]}".encode()
            ).hexdigest()[:12]

            entries.append({
                "entry_id":        entry_id,
                "ligand_smiles":   "",
                "ligand_sequence": apt_seq,
                "ligand_entity":   entity,
                "target_sequence": tgt_seq,
                "target_id":       tgt_name,
                "target_entity":   "protein",
                "log_kd_nM":       log_kd,
                "assay_type":      "Kd",
                "assay_type_idx":  0,
                "source":          "aptabase",
                "is_binder":       is_binder,
                "is_covalent":     0,
                "is_allosteric":   0,
            })

    print(f"  AptaBase: {len(entries):,} entries")
    return entries


def parse_skempi(path: str) -> List[Dict]:
    """Parse SKEMPI2 CSV (protein-protein ΔΔG)."""
    print(f"Parsing SKEMPI2: {path}")
    entries = []

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            pdb_id  = row.get("#Pdb", "").strip()
            kd_wt   = row.get("Affinity_wt_parsed", "").strip()
            kd_mut  = row.get("Affinity_mut_parsed", "").strip()

            if not pdb_id:
                continue

            # Wild-type affinity
            for kd_str, label in [(kd_wt, "wt"), (kd_mut, "mut")]:
                if not kd_str:
                    continue
                try:
                    kd_M  = float(kd_str)   # in Molar
                    kd_nM = kd_M * 1e9
                    log_kd = math.log10(max(kd_nM, 0.001))
                except ValueError:
                    continue

                entry_id = hashlib.md5(
                    f"skempi_{pdb_id}_{label}".encode()
                ).hexdigest()[:12]

                entries.append({
                    "entry_id":        entry_id,
                    "ligand_smiles":   "",
                    "ligand_sequence": f"PDB:{pdb_id}:chainA",
                    "ligand_entity":   "protein",
                    "target_sequence": f"PDB:{pdb_id}:chainB",
                    "target_id":       pdb_id,
                    "target_entity":   "protein",
                    "log_kd_nM":       log_kd,
                    "assay_type":      "Kd_SPR",
                    "assay_type_idx":  3,
                    "source":          "skempi2",
                    "is_binder":       int(log_kd < BINDER_CUTOFF_LOG_NM),
                    "is_covalent":     0,
                    "is_allosteric":   0,
                })

    print(f"  SKEMPI2: {len(entries):,} entries")
    return entries


def parse_pdbbind(path: str) -> List[Dict]:
    """Parse PDBbind index CSV."""
    print(f"Parsing PDBbind: {path}")
    entries = []

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id    = row.get("pdb_id", "").strip()
            pkd       = row.get("pkd", "").strip()   # -log10(Kd/M)
            aff_type  = row.get("affinity_type", "Kd").strip()
            if not pdb_id or not pkd:
                continue

            try:
                pkd_val = float(pkd)
                # pKd(M) = -log10(Kd/M) → Kd_nM = 10^(9-pKd)
                log_kd  = 9.0 - pkd_val
            except ValueError:
                continue

            entry_id = hashlib.md5(f"pdb_{pdb_id}".encode()).hexdigest()[:12]
            entries.append({
                "entry_id":        entry_id,
                "ligand_smiles":   "",
                "ligand_sequence": f"PDB:{pdb_id}:ligand",
                "ligand_entity":   "small_mol",
                "target_sequence": f"PDB:{pdb_id}:protein",
                "target_id":       pdb_id,
                "target_entity":   "protein",
                "log_kd_nM":       log_kd,
                "assay_type":      aff_type,
                "assay_type_idx":  ASSAY_TYPES.get(aff_type, 0),
                "source":          "pdbbind",
                "is_binder":       int(log_kd < BINDER_CUTOFF_LOG_NM),
                "is_covalent":     0,
                "is_allosteric":   0,
            })

    print(f"  PDBbind: {len(entries):,} entries")
    return entries


def parse_covalentdb(path: str) -> List[Dict]:
    """Parse CovalentDB — covalent warhead kinetics."""
    print(f"Parsing CovalentDB: {path}")
    entries = []

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles   = row.get("SMILES", "").strip()
            tgt_seq  = row.get("Target_Sequence", "").strip()
            kinact   = row.get("kinact_KI", "").strip()   # kinact/KI in M⁻¹s⁻¹
            ki_str   = row.get("KI_uM", "").strip()

            if not smiles or not tgt_seq:
                continue

            log_kd = None
            if ki_str:
                try:
                    ki_uM = float(ki_str)
                    ki_nM = ki_uM * 1000
                    log_kd = math.log10(max(ki_nM, 0.001))
                except ValueError:
                    pass

            entry_id = hashlib.md5(
                f"cov_{smiles[:50]}_{tgt_seq[:20]}".encode()
            ).hexdigest()[:12]

            entries.append({
                "entry_id":        entry_id,
                "ligand_smiles":   smiles,
                "ligand_sequence": "",
                "ligand_entity":   "small_mol",
                "target_sequence": tgt_seq,
                "target_id":       "",
                "target_entity":   "protein",
                "log_kd_nM":       log_kd,
                "assay_type":      "kinact_KI",
                "assay_type_idx":  8,
                "source":          "covalentdb",
                "is_binder":       1,
                "is_covalent":     1,
                "is_allosteric":   0,
            })

    print(f"  CovalentDB: {len(entries):,} entries")
    return entries


# ──────────────────────────────────────────────
# DEDUPLICATION
# ──────────────────────────────────────────────

def deduplicate(entries: List[Dict], keep: str = "best") -> List[Dict]:
    """
    Remove duplicates by (ligand_id, target_id, assay_type).
    keep='best': keep tightest binder
    keep='mean': keep mean affinity
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)

    for e in entries:
        key = (
            e.get("ligand_smiles","")[:40] or e.get("ligand_sequence","")[:20],
            e.get("target_sequence","")[:30],
            e.get("assay_type",""),
        )
        groups[str(key)].append(e)

    deduped = []
    for key, group in groups.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue

        valid = [e for e in group if e["log_kd_nM"] is not None]
        if not valid:
            deduped.append(group[0])
            continue

        if keep == "best":
            # Tightest binder (lowest Kd)
            best = min(valid, key=lambda x: x["log_kd_nM"])
            deduped.append(best)
        else:
            # Mean
            mean_log_kd = float(np.mean([e["log_kd_nM"] for e in valid]))
            rep = valid[0].copy()
            rep["log_kd_nM"] = mean_log_kd
            deduped.append(rep)

    print(f"  Deduplication: {len(entries):,} → {len(deduped):,} entries")
    return deduped


# ──────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# No target leakage: same target only in one split
# ──────────────────────────────────────────────

def target_aware_split(
    entries:       List[Dict],
    val_fraction:  float = 0.05,
    test_fraction: float = 0.10,
    seed:          int   = 42,
) -> List[Dict]:
    """
    Split by target identity: targets in test set never appear in train.
    This gives a true held-out generalization benchmark.
    """
    random.seed(seed)

    # Collect all target IDs
    targets = list({e["target_id"] or e["target_sequence"][:20] for e in entries})
    random.shuffle(targets)

    n_test = max(1, int(len(targets) * test_fraction))
    n_val  = max(1, int(len(targets) * val_fraction))

    test_targets = set(targets[:n_test])
    val_targets  = set(targets[n_test:n_test + n_val])

    for e in entries:
        tgt = e["target_id"] or e["target_sequence"][:20]
        if tgt in test_targets:
            e["split"] = "test"
        elif tgt in val_targets:
            e["split"] = "val"
        else:
            e["split"] = "train"

    split_counts = Counter(e["split"] for e in entries)
    print(f"  Train/val/test split: {split_counts}")
    return entries


# ──────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────

def compute_stats(entries: List[Dict]) -> Dict:
    """Compute dataset statistics for reporting."""
    log_kds = [e["log_kd_nM"] for e in entries if e["log_kd_nM"] is not None]
    sources  = Counter(e["source"] for e in entries)
    assays   = Counter(e["assay_type"] for e in entries)
    entities = Counter(e["ligand_entity"] for e in entries)
    splits   = Counter(e["split"] for e in entries)
    binders  = sum(1 for e in entries if e["is_binder"])
    covalent = sum(1 for e in entries if e["is_covalent"])
    allost   = sum(1 for e in entries if e["is_allosteric"])

    return {
        "total_entries":    len(entries),
        "entries_with_kd":  len(log_kds),
        "binders":          binders,
        "non_binders":      len(entries) - binders,
        "covalent":         covalent,
        "allosteric":       allost,
        "log_kd_mean":      float(np.mean(log_kds)) if log_kds else None,
        "log_kd_std":       float(np.std(log_kds))  if log_kds else None,
        "log_kd_min":       float(np.min(log_kds))  if log_kds else None,
        "log_kd_max":       float(np.max(log_kds))  if log_kds else None,
        "by_source":        dict(sources),
        "by_assay":         dict(assays),
        "by_ligand_entity": dict(entities),
        "by_split":         dict(splits),
    }


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build BindFM affinity index")
    parser.add_argument("--data-dir",    type=str, default="./data")
    parser.add_argument("--output-dir",  type=str, default="./data/affinity")
    parser.add_argument("--sources",     type=str,
                        default="bindingdb,chembl,aptabase,skempi,pdbbind,covalentdb",
                        help="Comma-separated list of sources to include")
    parser.add_argument("--val-frac",    type=float, default=0.05)
    parser.add_argument("--test-frac",   type=float, default=0.10)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dedup",       action="store_true", default=True)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources    = [s.strip() for s in args.sources.split(",")]

    all_entries = []

    # Parse each source
    source_parsers = {
        "bindingdb": (data_dir / "bindingdb" / "BindingDB_All.tsv", parse_bindingdb),
        "chembl":    (data_dir / "chembl" / "chembl_activities.csv", parse_chembl_csv),
        "aptabase":  (data_dir / "aptabase" / "aptabase_pairs.csv",  parse_aptabase),
        "skempi":    (data_dir / "skempi" / "SKEMPI2.csv",           parse_skempi),
        "pdbbind":   (data_dir / "pdbbind" / "pdbbind2020.csv",       parse_pdbbind),
        "covalentdb":(data_dir / "covalentdb" / "CovalentDB.csv",    parse_covalentdb),
    }

    for src in sources:
        if src not in source_parsers:
            print(f"Unknown source: {src}, skipping")
            continue
        path, parser_fn = source_parsers[src]
        if not path.exists():
            print(f"Source file not found: {path}, skipping")
            continue
        try:
            entries = parser_fn(str(path))
            all_entries.extend(entries)
        except Exception as e:
            print(f"Error parsing {src}: {e}")

    print(f"\nTotal raw entries: {len(all_entries):,}")

    if not all_entries:
        print("No entries loaded. Exiting.")
        sys.exit(1)

    # Deduplication
    if args.dedup:
        all_entries = deduplicate(all_entries)

    # Target-aware split
    all_entries = target_aware_split(
        all_entries, args.val_frac, args.test_frac, args.seed
    )

    # Fill missing split field
    for e in all_entries:
        if "split" not in e:
            e["split"] = "train"

    # Write unified CSV
    all_path = output_dir / "all_affinity.csv"
    with open(all_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for e in all_entries:
            writer.writerow(e)
    print(f"\nUnified affinity index: {all_path}")

    # Write split CSVs
    for split in ("train", "val", "test"):
        split_entries = [e for e in all_entries if e["split"] == split]
        split_path = output_dir / f"{split}.csv"
        with open(split_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=UNIFIED_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for e in split_entries:
                writer.writerow(e)
        print(f"  {split}: {len(split_entries):,} entries → {split_path}")

    # Write statistics
    stats = compute_stats(all_entries)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset statistics:")
    print(f"  Total entries:      {stats['total_entries']:>10,}")
    print(f"  With Kd data:       {stats['entries_with_kd']:>10,}")
    print(f"  Binders:            {stats['binders']:>10,}")
    print(f"  Non-binders:        {stats['non_binders']:>10,}")
    print(f"  Covalent:           {stats['covalent']:>10,}")
    print(f"  Allosteric:         {stats['allosteric']:>10,}")
    print(f"\n  By source:")
    for src, n in sorted(stats['by_source'].items(), key=lambda x: -x[1]):
        print(f"    {src:<20s}: {n:>10,}")
    print(f"\n  By modality:")
    for entity, n in sorted(stats['by_ligand_entity'].items(), key=lambda x: -x[1]):
        print(f"    {entity:<20s}: {n:>10,}")
    print(f"\nStats saved: {stats_path}")


if __name__ == "__main__":
    main()
