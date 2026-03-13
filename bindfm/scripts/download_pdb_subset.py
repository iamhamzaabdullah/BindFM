#!/usr/bin/env python3
"""
BindFM — PDB Subset Downloader
--------------------------------
Downloads a curated subset of PDB structures relevant to BindFM training.
Uses the RCSB PDB Search API to select structures by:
  - Resolution cutoff
  - Presence of protein + nucleic acid (for aptamer training)
  - Presence of protein + ligand
  - Protein-protein complexes
  - Minimum chain length

Much faster than full PDB mirror (~5-50K structures vs 220K).
Sufficient for Stage 0 geometry pretraining and Stage 1 structural.

Usage:
    python3 scripts/download_pdb_subset.py \
        --output-dir ./data/pdb \
        --max-structures 50000 \
        --resolution-cutoff 3.5 \
        --n-workers 8
"""

import os
import sys
import json
import time
import gzip
import argparse
import requests
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Set


# ──────────────────────────────────────────────
# RCSB SEARCH API QUERIES
# ──────────────────────────────────────────────

RCSB_SEARCH_URL  = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_CIF_URL     = "https://files.rcsb.org/download/{pdb_id}.cif"

def make_query_protein_nucleic(resolution: float) -> Dict:
    """Protein + RNA/DNA complexes — key for aptamer training."""
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": resolution,
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater",
                        "value": 0,
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_nucleic_acid",
                        "operator": "greater",
                        "value": 0,
                    }
                },
            ]
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": 10000},
            "results_verbosity": "minimal",
        }
    }


def make_query_protein_ligand(resolution: float) -> Dict:
    """Protein + small molecule complexes."""
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": resolution,
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater",
                        "value": 0,
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater",
                        "value": 0,
                    }
                },
            ]
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": 10000},
            "results_verbosity": "minimal",
        }
    }


def make_query_protein_protein(resolution: float) -> Dict:
    """Protein-protein complexes (at least 2 protein chains)."""
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": resolution,
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": 2,
                    }
                },
            ]
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": 10000},
            "results_verbosity": "minimal",
        }
    }


def search_rcsb(query: Dict, max_results: int = 10000) -> List[str]:
    """Execute RCSB search query and return list of PDB IDs."""
    pdb_ids = []
    start   = 0
    rows    = min(1000, max_results)

    while len(pdb_ids) < max_results:
        query["request_options"]["paginate"] = {"start": start, "rows": rows}
        try:
            resp = requests.post(
                RCSB_SEARCH_URL,
                json=query,
                timeout=30,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  RCSB search error at offset {start}: {e}")
            break

        results = data.get("result_set", [])
        if not results:
            break

        for r in results:
            pdb_ids.append(r["identifier"].lower())

        total = data.get("total_count", 0)
        start += len(results)
        print(f"  Fetched {len(pdb_ids)} / {min(total, max_results)} PDB IDs...")

        if start >= total or start >= max_results:
            break
        time.sleep(0.1)  # rate limit

    return pdb_ids[:max_results]


# ──────────────────────────────────────────────
# DOWNLOAD WORKERS
# ──────────────────────────────────────────────

def download_pdb(pdb_id: str, output_dir: Path) -> Optional[str]:
    """
    Download a single PDB file.
    Returns path on success, None on failure.
    """
    out_path = output_dir / f"{pdb_id}.pdb"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return str(out_path)

    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return str(out_path)
    except requests.RequestException:
        # Try compressed version
        try:
            url_gz = url.replace(".pdb", ".pdb.gz")
            resp   = requests.get(url_gz, timeout=30)
            resp.raise_for_status()
            with gzip.open(url_gz, "rb") as gz:
                with open(out_path, "wb") as f:
                    f.write(gz.read())
            return str(out_path)
        except Exception:
            return None


def download_batch(
    pdb_ids:     List[str],
    output_dir:  Path,
    n_workers:   int = 8,
    desc:        str = "",
) -> Dict[str, int]:
    """Download a batch of PDB files in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    succeeded = failed = skipped = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(download_pdb, pid, output_dir): pid
            for pid in pdb_ids
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            pid    = futures[future]
            result = future.result()
            if result is None:
                failed  += 1
            else:
                path = Path(result)
                if path.stat().st_size > 1000:
                    succeeded += 1
                else:
                    skipped += 1

            if (i + 1) % 500 == 0 or (i + 1) == len(pdb_ids):
                print(f"  {desc}: {i+1}/{len(pdb_ids)} "
                      f"(ok={succeeded}, fail={failed})")

    return {"succeeded": succeeded, "failed": failed, "skipped": skipped}


# ──────────────────────────────────────────────
# CHAIN DETECTION: classify each PDB by modality
# ──────────────────────────────────────────────

PROTEIN_RESIDUES  = set("ACDEFGHIKLMNPQRSTVWY")
RNA_RESIDUES      = {"A","C","G","U","I","PSU","2MA","2MG"}
DNA_RESIDUES      = {"DA","DC","DG","DT","DI"}
SMALL_MOL_EXCLUDE = {"HOH","WAT","H2O","NA","K","MG","CA","ZN","CL","SO4",
                     "PO4","GOL","EDO","PEG","ACT","TRS","HEP","FMT"}

def classify_pdb_chains(pdb_path: str) -> Dict:
    """
    Read PDB file and classify chains by molecular type.
    Returns dict: {chain_id: entity_type} + metadata.
    """
    chains = {}
    residues_per_chain = {}

    try:
        with open(pdb_path) as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                if len(line) < 26:
                    continue
                chain_id = line[21]
                resname  = line[17:20].strip()
                if chain_id not in residues_per_chain:
                    residues_per_chain[chain_id] = set()
                residues_per_chain[chain_id].add(resname)
    except Exception:
        return {}

    for chain_id, resnames in residues_per_chain.items():
        rna_count  = len(resnames & RNA_RESIDUES)
        dna_count  = len(resnames & DNA_RESIDUES)
        sm_resnames = resnames - SMALL_MOL_EXCLUDE - RNA_RESIDUES - DNA_RESIDUES
        is_protein  = any(r in PROTEIN_RESIDUES for r in resnames)

        if rna_count > dna_count and rna_count >= 2:
            chains[chain_id] = "rna"
        elif dna_count > 0 and dna_count >= 2:
            chains[chain_id] = "dna"
        elif is_protein and len(resnames) >= 10:
            chains[chain_id] = "protein"
        elif len(sm_resnames) >= 1 and not is_protein:
            chains[chain_id] = "small_mol"

    return chains


# ──────────────────────────────────────────────
# INDEX BUILDER
# ──────────────────────────────────────────────

def build_complex_index(
    pdb_dir:    Path,
    output_json:Path,
    max_pairs:  int = 500000,
) -> List[Dict]:
    """
    Scan downloaded PDB files, classify chains, and build
    the complex index JSON expected by BindFMComplexDataset.

    For each PDB, creates entries for each valid chain pair
    that constitutes a binding interaction.
    """
    print(f"\nBuilding complex index from {pdb_dir}...")
    entries  = []
    pdb_files = list(pdb_dir.glob("**/*.pdb"))
    print(f"Found {len(pdb_files)} PDB files.")

    # Modality pair priority (we want diversity)
    pair_counts = {
        "protein-small_mol": 0,
        "protein-protein":   0,
        "protein-rna":       0,
        "protein-dna":       0,
        "rna-small_mol":     0,
        "dna-small_mol":     0,
        "rna-rna":           0,
    }

    for pdb_path in pdb_files:
        if len(entries) >= max_pairs:
            break

        pdb_id = pdb_path.stem.lower()
        chains = classify_pdb_chains(str(pdb_path))
        if len(chains) < 2:
            continue

        chain_ids  = list(chains.keys())
        entity_map = chains

        # Generate all valid chain pairs
        for i in range(len(chain_ids)):
            for j in range(i + 1, len(chain_ids)):
                ca, cb = chain_ids[i], chain_ids[j]
                ea, eb = entity_map[ca], entity_map[cb]

                # Skip water-only or ion-only
                if ea not in ("protein","rna","dna","small_mol"):
                    continue
                if eb not in ("protein","rna","dna","small_mol"):
                    continue

                # Sort so protein is always chain_b (target) convention
                # Exception: protein-protein keeps alphabetical
                if ea == "small_mol" or (ea == "rna" and eb == "protein"):
                    ca, cb = cb, ca
                    ea, eb = eb, ea

                pair_key = f"{ea}-{eb}"
                if pair_key not in pair_counts:
                    pair_key = f"{eb}-{ea}"
                if pair_key in pair_counts:
                    pair_counts[pair_key] += 1

                entries.append({
                    "pdb_id":    pdb_id,
                    "pdb_path":  str(pdb_path),
                    "chain_a":   ca,
                    "chain_b":   cb,
                    "entity_a":  ea,
                    "entity_b":  eb,
                    "modality":  f"{ea}-{eb}",
                    "resolution": None,  # filled by annotate_resolution()
                    "log_kd":    None,   # filled by merge_affinity_data()
                })

    print(f"\nComplex index summary:")
    for key, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
        print(f"  {key:<25s}: {count:>6,}")
    print(f"  TOTAL: {len(entries):,}")

    with open(output_json, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nIndex saved to {output_json}")

    return entries


def annotate_resolution(
    index:    List[Dict],
    rcsb_url: str = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}",
) -> List[Dict]:
    """
    Fetch resolution metadata from RCSB REST API for each entry.
    Adds 'resolution' field to each entry.
    Batches requests to avoid rate limiting.
    """
    print("Annotating resolution from RCSB API...")
    seen_ids  = {}
    annotated = 0

    for entry in index:
        pdb_id = entry["pdb_id"].upper()
        if pdb_id in seen_ids:
            entry["resolution"] = seen_ids[pdb_id]
            continue

        try:
            resp = requests.get(
                rcsb_url.format(pdb_id=pdb_id),
                timeout=10
            )
            data = resp.json()
            res  = (data.get("rcsb_entry_info", {})
                        .get("resolution_combined", [None]))
            resolution = res[0] if isinstance(res, list) and res else None
        except Exception:
            resolution = None

        seen_ids[pdb_id] = resolution
        entry["resolution"] = resolution
        annotated += 1

        if annotated % 500 == 0:
            print(f"  Annotated {annotated} structures...")
        time.sleep(0.05)   # gentle rate limiting

    return index


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download PDB structures for BindFM")
    parser.add_argument("--output-dir",        type=str, required=True)
    parser.add_argument("--max-structures",    type=int, default=50000)
    parser.add_argument("--resolution-cutoff", type=float, default=3.5)
    parser.add_argument("--n-workers",         type=int, default=8)
    parser.add_argument("--has-ligand",        action="store_true")
    parser.add_argument("--has-nucleic-acid",  action="store_true")
    parser.add_argument("--build-index-only",  action="store_true",
                        help="Skip download, only build index from existing PDBs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    complex_dir = out_dir / "complexes"
    monomer_dir = out_dir / "monomers"
    complex_dir.mkdir(parents=True, exist_ok=True)
    monomer_dir.mkdir(parents=True, exist_ok=True)

    all_pdb_ids: Set[str] = set()

    if not args.build_index_only:
        per_query = max(1, args.max_structures // 3)

        # Query 1: protein + nucleic acid
        print("\n[1/3] Querying protein-nucleic acid complexes...")
        q1  = make_query_protein_nucleic(args.resolution_cutoff)
        ids = search_rcsb(q1, per_query)
        all_pdb_ids.update(ids)
        print(f"      Found {len(ids)} protein-NA complexes")

        # Query 2: protein + ligand
        print("\n[2/3] Querying protein-ligand complexes...")
        q2  = make_query_protein_ligand(args.resolution_cutoff)
        ids = search_rcsb(q2, per_query)
        all_pdb_ids.update(ids)
        print(f"      Found {len(ids)} protein-ligand complexes")

        # Query 3: protein-protein complexes
        print("\n[3/3] Querying protein-protein complexes...")
        q3  = make_query_protein_protein(args.resolution_cutoff)
        ids = search_rcsb(q3, per_query)
        all_pdb_ids.update(ids)
        print(f"      Found {len(ids)} protein-protein complexes")

        all_ids = list(all_pdb_ids)[:args.max_structures]
        print(f"\nTotal unique PDB IDs: {len(all_ids)}")

        # Save ID list
        id_list_path = out_dir / "pdb_ids.txt"
        with open(id_list_path, "w") as f:
            f.write("\n".join(all_ids))
        print(f"PDB ID list saved: {id_list_path}")

        # Download
        print(f"\nDownloading {len(all_ids)} PDB files with {args.n_workers} workers...")
        stats = download_batch(all_ids, complex_dir, args.n_workers, "PDB")
        print(f"\nDownload complete: {stats}")

    # Build complex index
    index_path = out_dir / "complexes.json"
    print(f"\nBuilding complex index...")
    entries = build_complex_index(complex_dir, index_path)

    # Annotate resolution (optional — can be slow for large sets)
    if len(entries) < 10000:
        print("Annotating resolution metadata...")
        entries = annotate_resolution(entries)
        with open(index_path, "w") as f:
            json.dump(entries, f, indent=2)

    print(f"\nDone. Complex index: {index_path} ({len(entries)} entries)")
    print(f"Next step: python3 scripts/build_affinity_index.py")


if __name__ == "__main__":
    main()
