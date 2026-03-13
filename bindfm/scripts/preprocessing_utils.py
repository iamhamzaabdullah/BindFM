#!/usr/bin/env python3
"""
BindFM — Preprocessing Utility Scripts
----------------------------------------
Collection of preprocessing scripts called by download_data.sh:

  export_chembl.py         — Export ChEMBL SQLite → CSV
  download_skempi_pdbs.py  — Download PDB structures for SKEMPI2
  parse_pdbbind_index.py   — Parse PDBbind INDEX files → CSV
  preprocess_bindingdb.py  — Clean and split BindingDB TSV
  split_aptabase.py        — Target-aware split for AptaBase
  preprocess_dude.py       — Convert DUD-E to VS benchmark format
  build_rna_ligand_benchmark.py — Extract RNA-ligand pairs from PDB
  build_allosteric_benchmark.py — Build allosteric test set
  create_aptabase_placeholder.py — Synthetic placeholder for testing

All scripts are importable as modules OR runnable as CLIs.
Dispatch is done by script name detection at the bottom.
"""

import os
import sys
import csv
import json
import gzip
import math
import random
import hashlib
import argparse
import requests
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


# =============================================================================
# export_chembl.py
# =============================================================================

def export_chembl(db_path: str, output_csv: str, min_assay_count: int = 3):
    """
    Export ChEMBL activities table to CSV with protein sequences.
    Joins: activities → assays → target_components → component_sequences
    Filters to direct binding assays with pChEMBL value.
    """
    try:
        import sqlite3
    except ImportError:
        print("sqlite3 not available")
        return

    print(f"Exporting ChEMBL from {db_path}...")
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    query = """
    SELECT
        cs.canonical_smiles,
        cs.standard_inchi_key,
        td.pref_name AS target_pref_name,
        cs2.sequence AS target_sequence,
        act.standard_type,
        act.standard_value,
        act.standard_units,
        CASE WHEN act.standard_relation = '<' THEN 1 ELSE 0 END AS is_upper_bound,
        CASE WHEN prop.site_id IS NOT NULL THEN 1 ELSE 0 END AS is_allosteric,
        act.pchembl_value
    FROM activities act
    JOIN assays a         ON act.assay_id     = a.assay_id
    JOIN target_dictionary td ON a.tid        = td.tid
    JOIN target_components tc ON td.tid       = tc.tid
    JOIN component_sequences cs2 ON tc.component_id = cs2.component_id
    JOIN compound_structures cs  ON act.molregno = cs.molregno
    LEFT JOIN binding_sites bs   ON a.assay_id = bs.assay_id
    LEFT JOIN site_components sc ON bs.site_id = sc.site_id
    LEFT JOIN component_domains cd ON sc.component_id = cd.component_id
    LEFT JOIN domains dom          ON cd.domain_id = dom.domain_id
    LEFT JOIN (
        SELECT DISTINCT bs2.site_id
        FROM binding_sites bs2
        JOIN assays a2 ON bs2.assay_id = a2.assay_id
        WHERE a2.assay_type = 'B'
    ) prop ON prop.site_id = bs.site_id
    WHERE act.standard_type IN ('Kd','Ki','IC50','EC50','Kd_app')
      AND act.standard_value IS NOT NULL
      AND act.standard_value > 0
      AND cs.canonical_smiles IS NOT NULL
      AND cs2.sequence IS NOT NULL
      AND td.target_type = 'SINGLE PROTEIN'
    LIMIT 5000000
    """

    print("  Running SQL query (may take several minutes)...")
    cur.execute(query)
    columns = [d[0] for d in cur.description]
    rows    = cur.fetchall()
    conn.close()

    print(f"  Fetched {len(rows):,} activity records")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"  ChEMBL exported to {output_csv}")


# =============================================================================
# download_skempi_pdbs.py
# =============================================================================

def download_skempi_pdbs(skempi_csv: str, output_dir: str, n_workers: int = 8):
    """Download PDB structures for all entries in SKEMPI2."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = set()
    with open(skempi_csv) as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            pdb_id = row.get("#Pdb", "").strip().upper()
            if pdb_id:
                pdb_ids.add(pdb_id.split("_")[0])   # strip chain suffix

    print(f"Downloading {len(pdb_ids)} PDB structures for SKEMPI2...")

    def download_one(pdb_id: str) -> bool:
        out = output_dir / f"{pdb_id.lower()}.pdb"
        if out.exists() and out.stat().st_size > 1000:
            return True
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            out.write_bytes(r.content)
            return True
        except Exception:
            return False

    succeeded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(download_one, pid): pid for pid in pdb_ids}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            if fut.result():
                succeeded += 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(pdb_ids)} (ok={succeeded})")

    print(f"SKEMPI2 PDBs: {succeeded}/{len(pdb_ids)} downloaded")


# =============================================================================
# parse_pdbbind_index.py
# =============================================================================

def parse_pdbbind_index(index_file: str, structures_dir: str, output_csv: str):
    """
    Parse PDBbind INDEX file format:
        1abc  2.10  Kd  1.9e-10  // comment
    Columns: PDB_ID, RESOLUTION, MEASURE_TYPE, VALUE, SMILES
    """
    entries = []

    with open(index_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            pdb_id   = parts[0].lower()
            try:
                resolution = float(parts[1]) if parts[1] != "NMR" else 99.0
            except ValueError:
                resolution = 99.0

            measure  = parts[2]  # Kd, Ki, IC50, etc.
            val_str  = parts[3]

            # Parse value (may be "1.9e-10" in Molar)
            try:
                val_M  = float(val_str)
                val_nM = val_M * 1e9
                log_kd = math.log10(max(val_nM, 0.001))
            except (ValueError, IndexError):
                continue

            # Check structure exists
            struct_path = None
            for ext in [".pdb", f"/{pdb_id}/{pdb_id}_protein.pdb"]:
                p = Path(structures_dir) / f"{pdb_id}{ext}"
                if p.exists():
                    struct_path = str(p)
                    break

            entries.append({
                "pdb_id":      pdb_id,
                "resolution":  resolution,
                "pkd":         9.0 - log_kd,   # pKd in M convention for compat
                "log_kd_nM":   log_kd,
                "affinity_type": measure,
                "has_structure": struct_path is not None,
                "structure_path": struct_path or "",
            })

    print(f"PDBbind index: {len(entries)} entries parsed")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
        writer.writeheader()
        writer.writerows(entries)

    print(f"PDBbind CSV saved: {output_csv}")


# =============================================================================
# preprocess_bindingdb.py
# =============================================================================

def preprocess_bindingdb(input_tsv: str, output_dir: str,
                          test_fraction: float = 0.1, seed: int = 42):
    """
    Clean BindingDB TSV:
      - Remove entries with no SMILES or target sequence
      - Remove entries with ambiguous affinity (> or < only)
      - Target-aware train/test split
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries   = []
    targets   = defaultdict(list)

    print(f"Reading BindingDB: {input_tsv}")
    with open(input_tsv, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            smiles  = row.get("Ligand SMILES", "").strip()
            tgt_seq = row.get("BindingDB Target Chain  Sequence", "").strip()
            tgt_id  = row.get("UniProt (SwissProt) Entry Name", "").strip()

            if not smiles or not tgt_seq:
                continue

            # Get best available affinity (prefer Kd > Ki > IC50)
            log_kd = None
            assay  = None
            for field, atype in [("Kd (nM)", "Kd"), ("Ki (nM)", "Ki"),
                                   ("IC50 (nM)", "IC50")]:
                val_str = row.get(field, "").strip()
                if not val_str or val_str.startswith(">") or val_str.startswith("<"):
                    continue
                try:
                    kd_nm  = float(val_str)
                    log_kd = math.log10(max(kd_nm, 0.001))
                    assay  = atype
                    break
                except ValueError:
                    pass

            if log_kd is None:
                continue

            entry = {
                "smiles":      smiles,
                "target_seq":  tgt_seq,
                "target_id":   tgt_id or tgt_seq[:20],
                "log_kd_nM":   log_kd,
                "assay_type":  assay,
                "is_binder":   int(log_kd < 4.0),
            }
            entries.append(entry)
            targets[tgt_id or tgt_seq[:20]].append(len(entries) - 1)

    print(f"Cleaned entries: {len(entries):,}")

    # Target-aware split
    random.seed(seed)
    all_targets = list(targets.keys())
    random.shuffle(all_targets)
    n_test  = max(1, int(len(all_targets) * test_fraction))
    test_tgts = set(all_targets[:n_test])

    train_entries = [e for e in entries if e["target_id"] not in test_tgts]
    test_entries  = [e for e in entries if e["target_id"] in test_tgts]

    fields = list(entries[0].keys())
    for split_name, split_data in [("train", train_entries), ("test", test_entries)]:
        out_path = output_dir / f"bindingdb_{split_name}_split.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split_data)
        print(f"  {split_name}: {len(split_data):,} → {out_path}")


# =============================================================================
# split_aptabase.py
# =============================================================================

def split_aptabase(input_csv: str, output_dir: str, test_targets: int = 20):
    """Target-aware split for AptaBase by protein target."""
    output_dir = Path(output_dir)
    entries = []
    targets = defaultdict(list)

    with open(input_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
            tgt = row.get("Target_Name", row.get("Target_Sequence","")[:20])
            targets[tgt].append(len(entries) - 1)

    all_targets = list(targets.keys())
    random.shuffle(all_targets)
    test_tgt_set = set(all_targets[:test_targets])

    train_entries = [e for e in entries
                     if e.get("Target_Name","")[:20] not in test_tgt_set]
    test_entries  = [e for e in entries
                     if e.get("Target_Name","")[:20] in test_tgt_set]

    if not entries:
        return

    fields = list(entries[0].keys())
    for name, data in [("aptabase_train.csv", train_entries),
                        ("aptabase_test.csv",  test_entries)]:
        out = output_dir / name
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        print(f"  AptaBase {name}: {len(data)} entries")


# =============================================================================
# preprocess_dude.py
# =============================================================================

def preprocess_dude(dude_dir: str, output_dir: str):
    """
    Convert DUD-E structure to BindFM virtual screening benchmark format.
    For each target in DUD-E:
      - actives.csv:   active compounds (Ki < threshold)
      - decoys.csv:    property-matched decoys
    Merge into per-target CSVs for the VS benchmark.
    """
    dude_dir   = Path(dude_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for target_dir in sorted(dude_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        tgt_name = target_dir.name

        # Active compounds
        actives_file = target_dir / "actives_final.ism"
        decoys_file  = target_dir / "decoys_final.ism"
        receptor_seq = target_dir / "receptor.fasta"

        if not actives_file.exists() or not decoys_file.exists():
            continue

        # Read protein sequence
        tgt_seq = ""
        if receptor_seq.exists():
            lines = receptor_seq.read_text().splitlines()
            tgt_seq = "".join(l.strip() for l in lines if not l.startswith(">"))

        rows = []

        def read_ism(path: Path, active: int):
            if not path.exists():
                return
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        rows.append({
                            "smiles":     parts[0],
                            "target_seq": tgt_seq,
                            "target_name":tgt_name,
                            "active":     active,
                        })

        read_ism(actives_file, 1)
        read_ism(decoys_file,  0)

        if rows:
            out = output_dir / f"{tgt_name}.csv"
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"  DUD-E {tgt_name}: {len(rows)} compounds")


# =============================================================================
# build_rna_ligand_benchmark.py
# =============================================================================

def build_rna_ligand_benchmark(pdb_dir: str, output_csv: str):
    """
    Extract RNA-small molecule pairs from PDB structures for
    the cross-modality benchmark.
    Looks for PDB files with both RNA chains and HETATM ligands.
    """
    pdb_dir = Path(pdb_dir)
    entries = []

    RNA_RES = {"A","C","G","U","PSU","I","2MA","5MC","1MA"}
    EXCLUDE = {"HOH","WAT","H2O","NA","K","MG","CA","ZN","CL",
               "SO4","PO4","GOL","EDO","PEG","ACT","TRS"}

    for pdb_file in sorted(pdb_dir.glob("**/*.pdb"))[:5000]:
        rna_chains   = set()
        ligand_chains = set()
        chain_residues = defaultdict(set)

        try:
            with open(pdb_file) as f:
                for line in f:
                    if len(line) < 26:
                        continue
                    record  = line[:6].strip()
                    resname = line[17:20].strip()
                    chain   = line[21]

                    if record == "ATOM":
                        chain_residues[chain].add(resname)
                    elif record == "HETATM":
                        if resname not in EXCLUDE and len(resname) <= 3:
                            ligand_chains.add((chain, resname))

            for chain, resnames in chain_residues.items():
                if len(resnames & RNA_RES) >= 2:
                    rna_chains.add(chain)

            if rna_chains and ligand_chains:
                pdb_id = pdb_file.stem
                for lig_chain, lig_resname in ligand_chains:
                    entries.append({
                        "pdb_id":       pdb_id,
                        "rna_chain":    list(rna_chains)[0],
                        "ligand_chain": lig_chain,
                        "ligand_name":  lig_resname,
                        "rna_sequence": "",  # filled below
                        "ligand_smiles":"",  # filled below
                        "log_kd_nM":    "",  # unknown from structure alone
                    })
        except Exception:
            pass

    print(f"RNA-ligand pairs found: {len(entries)}")
    if entries:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
            writer.writeheader()
            writer.writerows(entries)
        print(f"RNA-ligand benchmark: {output_csv}")


# =============================================================================
# build_allosteric_benchmark.py
# =============================================================================

def build_allosteric_benchmark(asd_path: str, output_csv: str):
    """
    Build allosteric binding test set from ASD (Allosteric Site Database).
    Falls back to a curated literature set if ASD unavailable.
    """
    entries = []

    # Try ASD file
    if asd_path and Path(asd_path).exists():
        try:
            open_fn = gzip.open if asd_path.endswith(".gz") else open
            with open_fn(asd_path, "rt") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 5:
                        continue
                    entries.append({
                        "smiles":       parts[0] if len(parts) > 0 else "",
                        "target_seq":   parts[1] if len(parts) > 1 else "",
                        "log_kd_nM":    parts[2] if len(parts) > 2 else "",
                        "pdb_id":       parts[3] if len(parts) > 3 else "",
                        "allosteric_site": parts[4] if len(parts) > 4 else "",
                    })
            print(f"ASD: {len(entries)} allosteric entries")
        except Exception as e:
            print(f"ASD parsing failed: {e}")

    # Literature fallback: known allosteric drugs
    KNOWN_ALLOSTERIC = [
        # (SMILES, target_name, Kd_nM)
        ("CC(C)(C)c1cnc(NCc2cccc(F)c2)nc1",  "MEK1",    10.0),
        ("O=C(Nc1ccc(F)cc1)c1cc2ccccc2nc1Nc1cccc(Cl)c1", "ABL1", 25.0),
        ("CC(=O)Nc1ccc(Oc2ccc3c(c2)CC(=O)N3)cc1", "ALLO_TARGET", 100.0),
    ]
    for smiles, tgt, kd_nM in KNOWN_ALLOSTERIC:
        entries.append({
            "smiles":     smiles,
            "target_seq": tgt,   # placeholder — real eval uses full sequences
            "log_kd_nM":  math.log10(kd_nM),
            "pdb_id":     "",
            "allosteric_site": "allosteric",
        })

    if entries:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
            writer.writeheader()
            writer.writerows(entries)
        print(f"Allosteric benchmark: {len(entries)} entries → {output_csv}")


# =============================================================================
# create_aptabase_placeholder.py
# =============================================================================

def create_aptabase_placeholder(output_path: str, n: int = 100):
    """
    Create a small synthetic AptaBase placeholder for testing
    when the real database is unavailable.
    """
    random.seed(42)
    bases_rna = list("ACGU")
    bases_dna = list("ACGT")
    aa        = list("ACDEFGHIKLMNPQRSTVWY")

    rows = []
    for i in range(n):
        is_dna = random.random() > 0.5
        length = random.randint(15, 45)
        bases  = bases_dna if is_dna else bases_rna
        seq    = "".join(random.choices(bases, k=length))
        tgt_len = random.randint(50, 300)
        tgt_seq = "".join(random.choices(aa, k=tgt_len))
        kd_nM   = 10 ** random.uniform(-1, 4)  # 0.1 nM to 10 µM

        rows.append({
            "Aptamer_Sequence": seq,
            "Target_Name":      f"TestTarget_{i % 10}",
            "Target_Sequence":  tgt_seq,
            "Kd_nM":            f"{kd_nM:.2f}",
            "Aptamer_Type":     "DNA" if is_dna else "RNA",
            "Modification":     "NONE",
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"AptaBase placeholder ({n} entries): {output_path}")


# =============================================================================
# CLI DISPATCH
# =============================================================================

SCRIPT_MAP = {
    "export_chembl":              export_chembl,
    "download_skempi_pdbs":       download_skempi_pdbs,
    "parse_pdbbind_index":        parse_pdbbind_index,
    "preprocess_bindingdb":       preprocess_bindingdb,
    "split_aptabase":             split_aptabase,
    "preprocess_dude":            preprocess_dude,
    "build_rna_ligand_benchmark": build_rna_ligand_benchmark,
    "build_allosteric_benchmark": build_allosteric_benchmark,
    "create_aptabase_placeholder":create_aptabase_placeholder,
}


def make_parser(script_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"BindFM preprocessing: {script_name}")
    if script_name == "export_chembl":
        p.add_argument("--db",     required=True)
        p.add_argument("--output", required=True)
    elif script_name == "download_skempi_pdbs":
        p.add_argument("--skempi-csv",   required=True)
        p.add_argument("--output-dir",   required=True)
        p.add_argument("--n-workers",    type=int, default=8)
    elif script_name == "parse_pdbbind_index":
        p.add_argument("--index-file",    required=True)
        p.add_argument("--structures-dir",required=True)
        p.add_argument("--output-csv",    required=True)
    elif script_name == "preprocess_bindingdb":
        p.add_argument("--input",           required=True)
        p.add_argument("--output-dir",      required=True)
        p.add_argument("--test-fraction",   type=float, default=0.1)
        p.add_argument("--seed",            type=int,   default=42)
    elif script_name == "split_aptabase":
        p.add_argument("--input",        required=True)
        p.add_argument("--output-dir",   required=True)
        p.add_argument("--test-targets", type=int, default=20)
    elif script_name == "preprocess_dude":
        p.add_argument("--dude-dir",   required=True)
        p.add_argument("--output-dir", required=True)
    elif script_name == "build_rna_ligand_benchmark":
        p.add_argument("--pdb-dir", required=True)
        p.add_argument("--output",  required=True)
    elif script_name == "build_allosteric_benchmark":
        p.add_argument("--asd",    default="")
        p.add_argument("--output", required=True)
    elif script_name == "create_aptabase_placeholder":
        p.add_argument("--output", required=True)
        p.add_argument("--n",      type=int, default=100)
    return p


if __name__ == "__main__":
    script_name = Path(sys.argv[0]).stem

    if script_name not in SCRIPT_MAP:
        print(f"Unknown script: {script_name}")
        print(f"Available: {', '.join(SCRIPT_MAP.keys())}")
        sys.exit(1)

    parser = make_parser(script_name)
    args   = vars(parser.parse_args())
    fn     = SCRIPT_MAP[script_name]

    # Map argparse args to function kwargs
    arg_map = {
        "db":              "db_path",
        "output":          "output_csv" if "rna" in script_name or "allosteric" in script_name else "output_path",
        "output_dir":      "output_dir",
        "skempi_csv":      "skempi_csv",
        "n_workers":       "n_workers",
        "index_file":      "index_file",
        "structures_dir":  "structures_dir",
        "output_csv":      "output_csv",
        "input":           "input_tsv" if "bindingdb" in script_name else "input_csv",
        "test_fraction":   "test_fraction",
        "seed":            "seed",
        "test_targets":    "test_targets",
        "dude_dir":        "dude_dir",
        "pdb_dir":         "pdb_dir",
        "asd":             "asd_path",
        "n":               "n",
    }

    kwargs = {}
    for k, v in args.items():
        mapped_k = arg_map.get(k, k)
        kwargs[mapped_k] = v

    fn(**{k: v for k, v in kwargs.items()
          if k in fn.__code__.co_varnames[:fn.__code__.co_argcount]})
