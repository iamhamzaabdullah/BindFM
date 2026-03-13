#!/usr/bin/env bash
# =============================================================================
# BindFM — Master Training Data Download Script
# =============================================================================
# Downloads all data sources required for all 4 curriculum stages.
#
# Usage:
#   chmod +x scripts/download_data.sh
#   ./scripts/download_data.sh --data-dir ./data --stages 0,1,2,3
#
# Individual source flags:
#   --pdb          Download PDB structures (Stage 0+1)
#   --qm9          Download QM9 small molecules (Stage 0)
#   --bindingdb    Download BindingDB (Stage 2)
#   --chembl       Download ChEMBL (Stage 2)
#   --aptabase     Download AptaBase (Stage 2)
#   --skempi       Download SKEMPI2 (Stage 2)
#   --pdbbind      Download PDBbind 2020 (Stage 1+2)
#   --rnacompete   Download RNAcompete (Stage 2)
#   --covalentdb   Download CovalentDB (Stage 2)
#   --benchmarks   Download benchmark test sets
#   --all          Download everything (default)
#
# Requirements:
#   wget, curl, python3, rsync (for PDB mirror)
#   ~500 GB free disk space for full dataset
#   Smaller subsets available with --max-pdb N
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR="./data"
STAGES="0,1,2,3"
DOWNLOAD_ALL=true
MAX_PDB_STRUCTURES=0        # 0 = all, N = cap at N (for dev/testing)
N_WORKERS=8                 # parallel download workers
VERIFY_CHECKSUMS=true

# Individual source flags (set to false by --all override if selective)
DL_PDB=false
DL_QM9=false
DL_BINDINGDB=false
DL_CHEMBL=false
DL_APTABASE=false
DL_SKEMPI=false
DL_PDBBIND=false
DL_RNACOMPETE=false
DL_COVALENTDB=false
DL_BENCHMARKS=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)     DATA_DIR="$2";    shift 2;;
        --stages)       STAGES="$2";     shift 2;;
        --max-pdb)      MAX_PDB_STRUCTURES="$2"; DOWNLOAD_ALL=false; DL_PDB=true; shift 2;;
        --pdb)          DL_PDB=true;     DOWNLOAD_ALL=false; shift;;
        --qm9)          DL_QM9=true;     DOWNLOAD_ALL=false; shift;;
        --bindingdb)    DL_BINDINGDB=true; DOWNLOAD_ALL=false; shift;;
        --chembl)       DL_CHEMBL=true;  DOWNLOAD_ALL=false; shift;;
        --aptabase)     DL_APTABASE=true; DOWNLOAD_ALL=false; shift;;
        --skempi)       DL_SKEMPI=true;  DOWNLOAD_ALL=false; shift;;
        --pdbbind)      DL_PDBBIND=true; DOWNLOAD_ALL=false; shift;;
        --rnacompete)   DL_RNACOMPETE=true; DOWNLOAD_ALL=false; shift;;
        --covalentdb)   DL_COVALENTDB=true; DOWNLOAD_ALL=false; shift;;
        --benchmarks)   DL_BENCHMARKS=true; DOWNLOAD_ALL=false; shift;;
        --all)          DOWNLOAD_ALL=true; shift;;
        --no-verify)    VERIFY_CHECKSUMS=false; shift;;
        --workers)      N_WORKERS="$2";  shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if $DOWNLOAD_ALL; then
    DL_PDB=true; DL_QM9=true; DL_BINDINGDB=true; DL_CHEMBL=true
    DL_APTABASE=true; DL_SKEMPI=true; DL_PDBBIND=true
    DL_RNACOMPETE=true; DL_COVALENTDB=true; DL_BENCHMARKS=true
fi

# ── Utilities ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
section() { echo -e "\n${GREEN}════════════════════════════════════════${NC}"; \
            echo -e "${GREEN} $1${NC}"; \
            echo -e "${GREEN}════════════════════════════════════════${NC}"; }

check_dep() {
    command -v "$1" &>/dev/null || { warn "$1 not found. Some downloads may fail."; return 1; }
    return 0
}

safe_download() {
    local url="$1" dest="$2" desc="$3"
    if [[ -f "$dest" ]]; then
        info "$desc already exists, skipping."
        return 0
    fi
    info "Downloading $desc..."
    wget -q --show-progress -O "$dest.tmp" "$url" \
        && mv "$dest.tmp" "$dest" \
        && info "$desc done." \
        || { warn "Failed to download $desc from $url"; rm -f "$dest.tmp"; return 1; }
}

verify_file() {
    local file="$1" min_bytes="$2" desc="$3"
    if [[ ! -f "$file" ]]; then
        warn "$desc not found: $file"
        return 1
    fi
    local size
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file")
    if [[ "$size" -lt "$min_bytes" ]]; then
        warn "$desc seems too small ($size bytes < $min_bytes expected)"
        return 1
    fi
    info "$desc verified (${size} bytes)"
    return 0
}

# ── Setup directories ──────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"/{pdb/{monomers,complexes,ligands},qm9,bindingdb,chembl,\
aptabase,skempi/structures,pdbbind/{structures,index},rnacompete,\
covalentdb,benchmarks/{virtual_screening},selex,affinity,logs}

LOG_FILE="$DATA_DIR/logs/download_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

info "BindFM Data Download"
info "Data directory: $DATA_DIR"
info "Log: $LOG_FILE"


# =============================================================================
# STAGE 0+1 — PDB STRUCTURES
# =============================================================================

if $DL_PDB; then
    section "PDB Structures"

    # Option A: rsync mirror (fastest, most complete)
    if check_dep rsync; then
        info "Mirroring PDB via rsync (this will take several hours for full PDB)..."
        info "PDB mmCIF mirror → converting to PDB format via python script"

        if [[ "$MAX_PDB_STRUCTURES" -gt 0 ]]; then
            warn "MAX_PDB_STRUCTURES=$MAX_PDB_STRUCTURES — using PDB RCSB search API subset"
            python3 scripts/download_pdb_subset.py \
                --output-dir "$DATA_DIR/pdb" \
                --max-structures "$MAX_PDB_STRUCTURES" \
                --resolution-cutoff 3.5 \
                --has-ligand \
                --has-nucleic-acid \
                --n-workers "$N_WORKERS"
        else
            info "Full PDB rsync (~700 GB, ~220K structures)..."
            rsync -rlpt -v -z \
                --include="*.ent.gz" \
                --exclude="*" \
                rsync.rcsb.org::ftp_data/structures/divided/pdb/ \
                "$DATA_DIR/pdb/raw_gz/" \
                || warn "rsync PDB mirror failed — try download_pdb_subset.py instead"

            # Decompress
            info "Decompressing PDB files..."
            find "$DATA_DIR/pdb/raw_gz" -name "*.ent.gz" | \
                xargs -P "$N_WORKERS" -I{} bash -c \
                'gunzip -c "$1" > "${1%.gz}" 2>/dev/null' _ {}
        fi
    else
        # Option B: RCSB search API for relevant structures only
        warn "rsync not found. Using RCSB search API for curated subset."
        python3 scripts/download_pdb_subset.py \
            --output-dir "$DATA_DIR/pdb" \
            --max-structures "${MAX_PDB_STRUCTURES:-50000}" \
            --resolution-cutoff 3.5 \
            --n-workers "$N_WORKERS"
    fi

    # Download PDBbind ligand SDF files (small mol coordinates in complexes)
    info "Downloading PDB ligand SDF library..."
    safe_download \
        "https://files.rcsb.org/pub/pdb/data/monomers/all_ccd.cif" \
        "$DATA_DIR/pdb/ligands/all_ccd.cif" \
        "PDB CCD (chemical component dictionary)"
fi


# =============================================================================
# STAGE 0 — QM9 SMALL MOLECULE GEOMETRIES
# =============================================================================

if $DL_QM9; then
    section "QM9 Dataset (small molecule geometry pretraining)"

    QM9_URL="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.csv"
    QM9_SDF="https://figshare.com/ndownloader/files/3195389"

    safe_download "$QM9_URL" "$DATA_DIR/qm9/qm9.csv" "QM9 CSV"
    safe_download "$QM9_SDF" "$DATA_DIR/qm9/qm9.sdf.tar.bz2" "QM9 SDF geometries"

    if [[ -f "$DATA_DIR/qm9/qm9.sdf.tar.bz2" ]]; then
        info "Extracting QM9..."
        tar -xjf "$DATA_DIR/qm9/qm9.sdf.tar.bz2" -C "$DATA_DIR/qm9/" \
            && info "QM9 extracted." \
            || warn "QM9 extraction failed"
    fi

    # Also download PCQM4Mv2 for larger scale geometry pretraining
    info "Downloading PCQM4Mv2 (larger geometry dataset)..."
    safe_download \
        "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip" \
        "$DATA_DIR/qm9/pcqm4mv2.zip" \
        "PCQM4Mv2"

    if [[ -f "$DATA_DIR/qm9/pcqm4mv2.zip" ]]; then
        unzip -q "$DATA_DIR/qm9/pcqm4mv2.zip" -d "$DATA_DIR/qm9/" \
            && info "PCQM4Mv2 extracted." \
            || warn "PCQM4Mv2 extraction failed"
    fi
fi


# =============================================================================
# STAGE 2 — BINDINGDB
# =============================================================================

if $DL_BINDINGDB; then
    section "BindingDB (~2.8M binding affinity measurements)"

    BINDINGDB_URL="https://www.bindingdb.org/bind/downloads/BindingDB_All_202401_tsv.zip"

    safe_download "$BINDINGDB_URL" \
        "$DATA_DIR/bindingdb/BindingDB_All.zip" \
        "BindingDB full TSV"

    if [[ -f "$DATA_DIR/bindingdb/BindingDB_All.zip" ]]; then
        info "Extracting BindingDB..."
        unzip -q "$DATA_DIR/bindingdb/BindingDB_All.zip" \
            -d "$DATA_DIR/bindingdb/" \
            && info "BindingDB extracted." \
            || warn "BindingDB extraction failed — may need manual download from bindingdb.org"
    fi

    # Split into train/test
    info "Creating BindingDB train/test split..."
    python3 scripts/preprocess_bindingdb.py \
        --input "$DATA_DIR/bindingdb/BindingDB_All.tsv" \
        --output-dir "$DATA_DIR/bindingdb/" \
        --test-fraction 0.1 \
        --seed 42 \
        || warn "BindingDB preprocessing failed — run manually after download"
fi


# =============================================================================
# STAGE 2 — ChEMBL
# =============================================================================

if $DL_CHEMBL; then
    section "ChEMBL Activity Data"

    # ChEMBL SQLite download (most complete)
    CHEMBL_VERSION="34"
    CHEMBL_URL="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_${CHEMBL_VERSION}/chembl_${CHEMBL_VERSION}_sqlite.tar.gz"

    safe_download "$CHEMBL_URL" \
        "$DATA_DIR/chembl/chembl_${CHEMBL_VERSION}_sqlite.tar.gz" \
        "ChEMBL ${CHEMBL_VERSION} SQLite"

    if [[ -f "$DATA_DIR/chembl/chembl_${CHEMBL_VERSION}_sqlite.tar.gz" ]]; then
        info "Extracting ChEMBL..."
        tar -xzf "$DATA_DIR/chembl/chembl_${CHEMBL_VERSION}_sqlite.tar.gz" \
            -C "$DATA_DIR/chembl/" \
            && info "ChEMBL extracted."

        # Export relevant tables to CSV for faster loading
        info "Exporting ChEMBL activity table to CSV..."
        python3 scripts/export_chembl.py \
            --db "$DATA_DIR/chembl/chembl_${CHEMBL_VERSION}/chembl_${CHEMBL_VERSION}.db" \
            --output "$DATA_DIR/chembl/chembl_activities.csv" \
            || warn "ChEMBL export failed — run scripts/export_chembl.py manually"
    fi
fi


# =============================================================================
# STAGE 2 — APTABASE (Aptamer Database)
# =============================================================================

if $DL_APTABASE; then
    section "AptaBase — Aptamer-Target Binding Data"

    # AptaBase is available via their web API or bulk download
    info "Downloading AptaBase..."
    safe_download \
        "http://aptabase.bioapps.biozentrum.uni-wuerzburg.de/downloads/aptabase_full.csv" \
        "$DATA_DIR/aptabase/aptabase_pairs.csv" \
        "AptaBase full dataset" \
        || {
            warn "AptaBase direct download unavailable."
            warn "Manual steps:"
            warn "  1. Visit https://aptabase.bioapps.biozentrum.uni-wuerzburg.de"
            warn "  2. Download full dataset"
            warn "  3. Save to: $DATA_DIR/aptabase/aptabase_pairs.csv"
            # Create placeholder for testing
            python3 scripts/create_aptabase_placeholder.py \
                --output "$DATA_DIR/aptabase/aptabase_pairs.csv"
        }

    # Also download Ellington lab SELEX database
    info "Downloading Ellington Lab Aptamer Database..."
    safe_download \
        "https://aptamer.icmb.utexas.edu/downloads/aptamer_db_full.csv" \
        "$DATA_DIR/aptabase/ellington_aptamers.csv" \
        "Ellington Aptamer DB" \
        || warn "Ellington aptamer DB unavailable — will use AptaBase only"

    # Split train/test by target protein identity (no data leakage)
    python3 scripts/split_aptabase.py \
        --input "$DATA_DIR/aptabase/aptabase_pairs.csv" \
        --output-dir "$DATA_DIR/aptabase/" \
        --test-targets 20 \
        || warn "AptaBase split failed"
fi


# =============================================================================
# STAGE 2 — SKEMPI2 (Protein-Protein Interaction Thermodynamics)
# =============================================================================

if $DL_SKEMPI; then
    section "SKEMPI2 — Protein-Protein Binding ΔΔG"

    safe_download \
        "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv" \
        "$DATA_DIR/skempi/SKEMPI2.csv" \
        "SKEMPI2 ΔΔG database"

    # Download associated PDB structures
    if [[ -f "$DATA_DIR/skempi/SKEMPI2.csv" ]]; then
        info "Downloading SKEMPI2 PDB structures..."
        python3 scripts/download_skempi_pdbs.py \
            --skempi-csv "$DATA_DIR/skempi/SKEMPI2.csv" \
            --output-dir "$DATA_DIR/skempi/structures/" \
            --n-workers "$N_WORKERS" \
            || warn "SKEMPI2 PDB download incomplete"

        # Create train/test split
        python3 scripts/split_skempi.py \
            --input "$DATA_DIR/skempi/SKEMPI2.csv" \
            --output-dir "$DATA_DIR/skempi/" \
            || warn "SKEMPI2 split failed"
    fi
fi


# =============================================================================
# STAGE 1+2 — PDBbind 2020
# =============================================================================

if $DL_PDBBIND; then
    section "PDBbind 2020 — Protein-Ligand Affinity with Structures"

    warn "PDBbind requires registration at http://www.pdbbind.org.cn"
    warn "After registering, download:"
    warn "  - PDBbind v2020 general set (pdbbind_v2020_general.tar.gz)"
    warn "  - PDBbind v2020 core set   (pdbbind_v2020_core.tar.gz)"
    warn "  - PDBbind v2020 index      (INDEX_general_PL_data.2020)"
    warn "Place them in: $DATA_DIR/pdbbind/"
    warn ""
    warn "If files already present, running preprocessing..."

    if [[ -f "$DATA_DIR/pdbbind/pdbbind_v2020_general.tar.gz" ]]; then
        info "Extracting PDBbind general set..."
        tar -xzf "$DATA_DIR/pdbbind/pdbbind_v2020_general.tar.gz" \
            -C "$DATA_DIR/pdbbind/structures/" \
            && info "PDBbind general set extracted."
    fi

    if [[ -f "$DATA_DIR/pdbbind/pdbbind_v2020_core.tar.gz" ]]; then
        info "Extracting PDBbind core set..."
        tar -xzf "$DATA_DIR/pdbbind/pdbbind_v2020_core.tar.gz" \
            -C "$DATA_DIR/pdbbind/structures/" \
            && info "PDBbind core set extracted."
    fi

    # Parse index file → CSV
    python3 scripts/parse_pdbbind_index.py \
        --index-file "$DATA_DIR/pdbbind/INDEX_general_PL_data.2020" \
        --structures-dir "$DATA_DIR/pdbbind/structures/" \
        --output-csv "$DATA_DIR/pdbbind/pdbbind2020.csv" \
        || warn "PDBbind index parsing failed"

    python3 scripts/parse_pdbbind_index.py \
        --index-file "$DATA_DIR/pdbbind/INDEX_core_data.2020" \
        --structures-dir "$DATA_DIR/pdbbind/structures/" \
        --output-csv "$DATA_DIR/pdbbind/pdbbind_core_2020.csv" \
        || warn "PDBbind core index parsing failed"
fi


# =============================================================================
# STAGE 2 — RNAcompete
# =============================================================================

if $DL_RNACOMPETE; then
    section "RNAcompete — RNA-Binding Protein Affinity"

    # RNAcompete data from NCBI GEO (multiple experiments)
    info "Downloading RNAcompete data from NCBI GEO..."

    # GSE117309 — RNAcompete 2.0 (240 RBPs)
    safe_download \
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE117309&format=file" \
        "$DATA_DIR/rnacompete/GSE117309_RAW.tar" \
        "RNAcompete 2.0 (GSE117309)"

    if [[ -f "$DATA_DIR/rnacompete/GSE117309_RAW.tar" ]]; then
        tar -xf "$DATA_DIR/rnacompete/GSE117309_RAW.tar" \
            -C "$DATA_DIR/rnacompete/" \
            && info "RNAcompete extracted." \
            || warn "RNAcompete extraction failed"
    fi

    # RNAcompete protein sequences
    safe_download \
        "http://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/RBP_fasta_sequences.tar.gz" \
        "$DATA_DIR/rnacompete/rbp_sequences.tar.gz" \
        "RNAcompete RBP sequences" \
        || warn "RNAcompete RBP sequences unavailable"
fi


# =============================================================================
# STAGE 2 — CovalentDB
# =============================================================================

if $DL_COVALENTDB; then
    section "CovalentDB — Covalent Inhibitor Kinetics"

    safe_download \
        "https://covalentdb.bioinformatics.ch/downloads/CovalentDB_full.csv" \
        "$DATA_DIR/covalentdb/CovalentDB.csv" \
        "CovalentDB" \
        || {
            warn "CovalentDB unavailable from primary URL"
            warn "Visit https://covalentdb.bioinformatics.ch and download manually"
            warn "Place at: $DATA_DIR/covalentdb/CovalentDB.csv"
        }
fi


# =============================================================================
# BENCHMARK TEST SETS
# =============================================================================

if $DL_BENCHMARKS; then
    section "Benchmark Test Sets"

    # CASF-2016
    safe_download \
        "http://www.pdbbind.org.cn/casf/files/casf/CASF-2016.tar.gz" \
        "$DATA_DIR/benchmarks/CASF-2016.tar.gz" \
        "CASF-2016" \
        || warn "CASF-2016 requires PDBbind registration"

    # DUD-E for virtual screening benchmark
    safe_download \
        "http://dude.docking.org/db/subsets/all/all.tar.gz" \
        "$DATA_DIR/benchmarks/dude_all.tar.gz" \
        "DUD-E virtual screening benchmark"

    if [[ -f "$DATA_DIR/benchmarks/dude_all.tar.gz" ]]; then
        info "Extracting DUD-E..."
        tar -xzf "$DATA_DIR/benchmarks/dude_all.tar.gz" \
            -C "$DATA_DIR/benchmarks/" \
            && python3 scripts/preprocess_dude.py \
                --dude-dir "$DATA_DIR/benchmarks/dude_all/" \
                --output-dir "$DATA_DIR/benchmarks/virtual_screening/" \
            || warn "DUD-E preprocessing failed"
    fi

    # RNA-ligand benchmark
    info "Building RNA-ligand test set from PDB..."
    python3 scripts/build_rna_ligand_benchmark.py \
        --pdb-dir "$DATA_DIR/pdb/complexes/" \
        --output "$DATA_DIR/benchmarks/rna_ligand_test.csv" \
        || warn "RNA-ligand benchmark build failed"

    # Allosteric test set (from ASD database)
    safe_download \
        "http://mdl.shsmu.edu.cn/ASD/downloads/ASD_Release_201807_XF.txt.gz" \
        "$DATA_DIR/benchmarks/asd_allosteric.txt.gz" \
        "Allosteric Sites Database" \
        || warn "ASD unavailable — using manual allosteric set from literature"

    python3 scripts/build_allosteric_benchmark.py \
        --asd "$DATA_DIR/benchmarks/asd_allosteric.txt.gz" \
        --output "$DATA_DIR/benchmarks/allosteric_test.csv" \
        || warn "Allosteric benchmark build failed"
fi


# =============================================================================
# FINAL VALIDATION
# =============================================================================

section "Download Summary"

check_dir() {
    local dir="$1" desc="$2"
    local n
    n=$(find "$dir" -type f 2>/dev/null | wc -l)
    if [[ "$n" -gt 0 ]]; then
        printf "  %-35s %6d files\n" "$desc" "$n"
    else
        printf "  %-35s %6s\n" "$desc" "EMPTY"
    fi
}

echo ""
echo "Data directory contents:"
check_dir "$DATA_DIR/pdb"          "PDB structures"
check_dir "$DATA_DIR/qm9"          "QM9 / PCQM4Mv2"
check_dir "$DATA_DIR/bindingdb"    "BindingDB"
check_dir "$DATA_DIR/chembl"       "ChEMBL"
check_dir "$DATA_DIR/aptabase"     "AptaBase"
check_dir "$DATA_DIR/skempi"       "SKEMPI2"
check_dir "$DATA_DIR/pdbbind"      "PDBbind 2020"
check_dir "$DATA_DIR/rnacompete"   "RNAcompete"
check_dir "$DATA_DIR/covalentdb"   "CovalentDB"
check_dir "$DATA_DIR/benchmarks"   "Benchmark test sets"
echo ""

# Disk usage
du -sh "$DATA_DIR" 2>/dev/null && echo ""
info "Download complete. Next step:"
info "  python3 scripts/build_pdb_index.py --data-dir $DATA_DIR"
info "  python3 scripts/build_affinity_index.py --data-dir $DATA_DIR"
