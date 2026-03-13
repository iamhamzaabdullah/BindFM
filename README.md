<div align="center">

<br/>

```
██████╗ ██╗███╗   ██╗██████╗ ███████╗███╗   ███╗
██╔══██╗██║████╗  ██║██╔══██╗██╔════╝████╗ ████║
██████╔╝██║██╔██╗ ██║██║  ██║█████╗  ██╔████╔██║
██╔══██╗██║██║╚██╗██║██║  ██║██╔══╝  ██║╚██╔╝██║
██████╔╝██║██║ ╚████║██████╔╝██║     ██║ ╚═╝ ██║
╚═════╝ ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝     ╚═╝     ╚═╝
```

**Universal Biomolecular Binding Foundation Model**

*Built atom-by-atom. From scratch. Five modalities. One representation.*

<br/>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/iamhamzaabdullah/BindFM/ci.yml?style=for-the-badge&label=CI)](https://github.com/iamhamzaabdullah/BindFM/actions)

<br/>

*[Terminal Bio](https://terminalbio.io) · Lead Researcher: **Hamza Abdullah***

<br/>

---

</div>

## The Problem with Every Other Approach

Every serious attempt at universal binding prediction hits the same wall: **incompatible representation spaces**.

ESM-C lives in protein sequence space. Small molecule encoders live in SMILES graph space. RNA models live in covariance structure space. When you stack these together and call it "universal," you're not unifying anything — you're building a routing table.

**BindFM takes the only principled approach**: represent every molecular entity — protein, RNA, DNA, small molecule — as the same thing it actually is: **a graph of atoms in 3D space**. One encoder. Shared weights. Universal representation.

A protein carbon and an RNA carbon pass through identical learned transformations. This is not a design constraint. This is the entire point.

---

## What BindFM Does

<table>
<tr>
<th width="50%">Input</th>
<th width="50%">Output</th>
</tr>
<tr>
<td>

Any two molecules in any format:
- Protein sequence (`MKTLL...`)
- RNA/DNA sequence (`GGTTGG...`)
- SMILES string (`CC(=O)Oc1...`)
- PDB file path (`complex.pdb`)
- Modified aptamer (`[LNA-A][2F-C]...`)

</td>
<td>

Three co-trained predictions:
- **Binding affinity** — log Kd, P(bind), kon/koff, t½, uncertainty
- **3D complex structure** — atom coordinates via SE(3)-equivariant flow matching
- **De novo binder design** — generated candidates conditioned on target Kd

</td>
</tr>
</table>

### Binding Modalities Covered

| # | Modality | Examples |
|---|----------|---------|
| 1 | **Protein ↔ Small Molecule** | Drug-target binding, covalent inhibitors, fragments |
| 2 | **Protein ↔ Protein** | PPI inhibitors, antibody-antigen, SKEMPI2 ΔΔG mutations |
| 3 | **Protein ↔ Nucleic Acid** | RNA-binding proteins, CRISPR-Cas9, splicing factors |
| 4 | **Nucleic ↔ Small Molecule** | Aptamer-drug binding, riboswitches, G-quadruplex ligands |
| 5 | **Nucleic ↔ Nucleic** | siRNA-mRNA, aptamer secondary structure, miRNA targeting |

> **Special depth on therapeutic aptamers.** Full support for LNA, 2'F, 2'OMe, phosphorothioate, FANA, and morpholino modifications — not as special cases, but through explicit modification tokens in the universal atom representation.

---

## Architecture

```
                          ╔══════════════════════════════════════╗
                          ║        197-dim Atom Tokenizer        ║
                          ║  element · hybridization · chirality ║
                          ║  ring · H-bond · charge · mod-type   ║
                          ╚═══════════════╦══════════════════════╝
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
        Entity A ──────────────►  ╔═══════════════╗  ◄────────────── Entity B
   (protein/RNA/DNA/              ║ Shared EGNN   ║               (any modality)
    small molecule)               ║   Encoder     ║
                                  ║               ║
                                  ║  8 equivariant║
                                  ║  EGNN layers  ║
                                  ║               ║
                                  ║  Same weights ║
                                  ║  both sides   ║
                                  ╚═══════════════╝
                                  h_a [N_a, 512]      h_b [N_b, 512]
                                  x_a [N_a, 3]        x_b [N_b, 3]
                                          │
                          ╔═══════════════╩═════════════════╗
                          ║         PairFormer Trunk         ║
                          ║                                  ║
                          ║  32 layers of:                   ║
                          ║  · outer product mean            ║
                          ║  · triangle multiplicative update ║
                          ║  · triangle self-attention       ║
                          ║  · single representation FFN     ║
                          ╚══════════╦══════════════════════╝
                                     │
         ╔═══════════════════════════╬══════════════════════════════╗
         ▼                           ▼                              ▼
  ┌─────────────┐           ┌─────────────────┐           ┌─────────────────┐
  │  Affinity   │           │    Structure    │           │   Generative   │
  │    Head     │           │      Head       │           │     Head        │
  │─────────────│           │─────────────────│           │─────────────────│
  │ log_kd [nM] │           │ Euler ODE       │           │ Flow matching   │
  │ P(bind)     │           │ flow matching   │           │ in (feat,coord) │
  │ kon / koff  │           │ 3D complex      │           │ space           │
  │ half-life   │           │ coordinates     │           │                 │
  │ uncertainty │           │ (Å)             │           │ conditioned on  │
  │ (Kendall)   │           │                 │           │ target Kd       │
  └─────────────┘           └─────────────────┘           └─────────────────┘
```

### Core Design Decisions

**Why EGNN (not SE(3)-Transformer or NequIP)?**
Satorras et al. 2021's EGNN achieves the same E(n)-equivariance as more complex architectures through a beautiful simplification: update coordinates as a weighted sum of displacement vectors. No spherical harmonics. No Clebsch-Gordan coefficients. Same theoretical guarantees, dramatically simpler implementation, proven on molecular binding tasks.

**Why shared encoder weights?**
Because that's the only way two molecules end up in the same representation space. When Entity A and Entity B share an encoder, their atom embeddings are directly comparable. The trunk can compute meaningful pairwise interactions between a protein residue's nitrogen and an aptamer's phosphate backbone — not just their respective pre-computed embeddings.

**Why flow matching (not diffusion)?**
OT-CFM (Lipman et al. 2022) uses straight interpolation paths between noise and data. This means fewer NFE at inference, better mode coverage, and a cleaner loss landscape for multi-task co-training.

**Why train from scratch?**
Because there's no pretrained model that can be fine-tuned to this. ESM-C, Evo2, RNA-FM — all of these learned representations shaped by evolutionary signals, not binding physics. Borrowing them would mean inheriting the wrong inductive biases for every modality boundary crossing.

---

## Installation

```bash
git clone https://github.com/iamhamzaabdullah/BindFM.git
cd BindFM
```

**Minimal (CPU / testing):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy biopython
```

**Full (GPU / training):**
```bash
pip install -r requirements.txt
```

**Verify:**
```bash
python3 quickstart.py --device cpu --size small --skip-training
# ✓ Tokenizer        ATOM_FEAT_DIM=197
# ✓ Encoder          DualEntityEncoder — 2 entities, shared weights
# ✓ Trunk            PairFormerTrunk
# ✓ Heads            Affinity + Structure + Generative
# ✓ Full forward pass (aspirin ↔ peptide)
# ✓ Thrombin aptamer ↔ thrombin protein
# ✓ HIV TAR RNA ↔ argininamide
# ✓ ALL TESTS PASSED
```

---

## Usage

### Predict Binding Affinity

```python
from inference.api import BindFMPredictor
from model.bindfm import BindFMConfig, BindFM

# Load trained checkpoint
predictor = BindFMPredictor.from_checkpoint("checkpoints/bindfm_stage3_best.pt")

# --- or initialize with random weights for testing ---
predictor = BindFMPredictor.from_config(BindFMConfig.small())

# Auto-detects input format (SMILES / sequence / PDB path)
result = predictor.predict_affinity(
    binder = "GGTTGGTGTGGTTGG",        # thrombin DNA aptamer
    target = "MKTLLLTLVVVTIVCLDLGYT",  # thrombin (fragment)
)
print(result)
```

```
BindFM Affinity Prediction
  Binding probability:  0.912
  Kd:                   8.4 nM
  kon:                  10^6.3 M⁻¹s⁻¹
  koff:                 10^-1.6 s⁻¹
  Residence time (t½):  18.7 min
  Uncertainty:          ±0.38 log units
```

### Predict 3D Complex Structure

```python
struct = predictor.predict_structure(
    binder      = "CC(=O)Oc1ccccc1C(=O)O",   # aspirin SMILES
    target      = "MKTLLLTLVVVTIVCLDL",
    n_steps     = 100,                         # Euler ODE steps
    output_pdb  = "aspirin_complex.pdb",
)
# StructureResult: 21 binder atoms, 162 target atoms
```

### Generate De Novo Aptamers

```python
candidates = predictor.generate_binders(
    target       = "MKTLLLTLVVVTIVCLDLGYT",
    modality     = "aptamer",        # or: "dna_aptamer", "protein", "small_mol"
    n_candidates = 100,
    target_kd_nM = 10.0,             # condition generation on desired affinity
    n_steps      = 200,
)

for c in candidates[:5]:
    print(c)
# Candidate #1: GCGGTTGGTGTGGTTGGCG...
#   Pred. Kd:  6.2 nM
#   P(bind):   0.953
```

### Screen a Compound Library

```python
library = [
    "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    # ... thousands of compounds
]

hits = predictor.screen_library(
    library    = library,
    target     = "MKTLLLTLVVVTIVCLDLGYT",
    top_k      = 10,
    assay_type = "Kd",
    verbose    = True,
)
# Screened 1000/1000...
# Rank 1  CC(=O)Oc1ccccc1...  Kd=4.1 nM  P(bind)=0.97
```

### Explicit Modality Hints

```python
# For ambiguous sequences or modified aptamers:
result = predictor.predict_affinity(
    binder      = "[LNA-G][LNA-G]TTGGTGTGGTTGG",
    target      = "MKTLLLTLVVVTIVCLDLGYT",
    binder_hint = "aptamer",    # force RNA aptamer interpretation
    target_hint = "protein",
    assay_type  = "SPR_Kd",     # Kd | Ki | IC50 | EC50 | SPR_Kd | ITC | EMSA
)
```

---

## Training

### 1. Data Setup

```bash
# Download all training sources (~500 GB)
bash scripts/download_data.sh --data-dir ./data

# Build PDB structure index (100k structures)
python3 scripts/download_pdb_subset.py \
    --output-dir ./data/pdb \
    --max-structures 100000

# Merge all affinity sources into unified CSV
python3 scripts/build_affinity_index.py --data-dir ./data
```

<details>
<summary><b>Training data sources</b></summary>

| Source | Modality | Scale |
|--------|----------|-------|
| [PDBbind 2020](http://www.pdbbind.org.cn/) | Protein–ligand | ~19k complexes with Kd |
| [BindingDB](https://www.bindingdb.org/) | Protein–SM, Protein–Protein | ~2.8M measurements |
| [ChEMBL 33](https://www.ebi.ac.uk/chembl/) | Protein–SM | ~20M bioactivity points |
| [AptaBase](https://aptabase.fi/) | Aptamer–Protein | SELEX-derived binding data |
| [SKEMPI2](https://life.bsc.es/pid/skempi2) | Protein–Protein | 7,085 ΔΔG on mutation |
| [RNAcompete](https://hugheslab.ccbr.utoronto.ca/) | RNA–Protein | 244 RBP binding profiles |
| [CovalentDB](https://zinc20.docking.org/) | Covalent inhibitors | Warhead + target pairs |
| PDB co-crystals | All modalities | ~150k complex structures |

</details>

### 2. Curriculum Training (4 Stages)

```bash
# Stage 0 — Geometry pretraining: learn what molecules look like in 3D
# Denoising objective on single-molecule coordinates
# Hardware: 4×A100, ~2 weeks
python3 training/train.py \
    --config configs/training_configs.yaml \
    --key full_stage0 \
    --data-dir ./data \
    --checkpoint-dir ./checkpoints

# Stage 1 — Complex structural pretraining: learn binding poses
# Flow matching on PDB co-crystal complexes, all modalities
# Hardware: 8×A100, ~3 weeks
python3 training/train.py --key full_stage1 \
    --prev-checkpoint ./checkpoints/bindfm_stage0_final.pt [...]

# Stage 2 — Affinity regression: introduce Kd/Ki/IC50 signal
# Multi-task: regression + binary classification + uncertainty
# Hardware: 4×A100, ~1 week
python3 training/train.py --key full_stage2 \
    --prev-checkpoint ./checkpoints/bindfm_stage1_final.pt [...]

# Stage 3 — Joint fine-tuning + generation
# All heads active. Generative flow matching co-trained end-to-end.
# Hardware: 8×A100, ~2 weeks
python3 training/train.py --key full_stage3 \
    --prev-checkpoint ./checkpoints/bindfm_stage2_final.pt [...]
```

**Using `make`:**
```bash
make train-small    # Small model, all 4 stages
make train-medium   # Medium model
make train-full     # Full model (A100 cluster required)
```

### Curriculum Rationale

The 4-stage curriculum mirrors how a physicist would think about this problem:

1. **Stage 0** — Learn molecular geometry without any binding signal. No label noise, no task complexity. Just: what do atoms look like in 3D space?
2. **Stage 1** — Learn what bound complexes look like structurally. The model learns interface geometry before it sees any affinity numbers.
3. **Stage 2** — Introduce quantitative affinity. The encoder and trunk are now primed with structural intuition, so affinity signals propagate meaningfully.
4. **Stage 3** — Unlock everything simultaneously. Generation is co-trained with affinity and structure, ensuring generated binders are both chemically valid and predicted to bind.

---

## Benchmarks

```bash
python3 -c "
from benchmarks.evaluate import BindFMEvaluator
from model.bindfm import BindFM

model     = BindFM.load('checkpoints/bindfm_stage3_best.pt')
evaluator = BindFMEvaluator(model, data_dir='./data')
results   = evaluator.run_all()
evaluator.save_results(results, 'benchmark_results.json')
"
```

| Benchmark | Task | Primary Metric | Target |
|-----------|------|----------------|--------|
| **PDBbind Core Set** | Protein–ligand Kd | Pearson R | > 0.80 |
| **CASF-2016** | Scoring / ranking / docking | Scoring R | > 0.78 |
| **BindingDB held-out** | Generalization to new targets | RMSE (log Kd) | < 0.80 |
| **SKEMPI2 ΔΔG** | Mutation effect on PPI | Spearman R | > 0.65 |
| **AptaBase** | Aptamer–protein binding | AUROC | > 0.85 |
| **Virtual Screening** | Enrichment in compound library | EF @ 1% | > 20× |
| **Novel Scaffold** | Chemotype generalization | Pearson R | > 0.65 |
| **Cross-Modality** | Train SM → test aptamer | Pearson R | > 0.60 |
| **Allosteric Sites** | Non-orthosteric binding | AUROC | > 0.75 |

---

## Model Sizes

| Variant | Encoder | Trunk | Parameters | Hardware | Stage 0–3 Time |
|---------|---------|-------|:----------:|----------|:--------------:|
| **Small** | 3L, d=64 | 3L, d_s=128 | ~2.1M | Free T4 (Colab) | 40 min |
| **Medium** | 6L, d=128 | 12L, d_s=256 | ~45M | 4×A100 | 1 week |
| **Full** | 8L, d=256 | 32L, d_s=512 | ~380M | 8×A100 | 8 weeks |

---

## Micro-Run on Free GPU

Run the small model end-to-end on Google Colab or Kaggle in **~40 minutes**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamhamzaabdullah/BindFM/blob/main/notebooks/BindFM_MicroRun.ipynb)

The notebook trains the nano model (~2.1M params) on synthetic data and runs all three heads. It's designed to be a legible, self-contained demonstration of every architectural component.

---

## Repository Structure

```
BindFM/
│
├── model/                          # Core neural network
│   ├── tokenizer.py                # 197-dim universal atom featurizer
│   ├── encoder.py                  # SE(3)-equivariant EGNN encoder (shared weights)
│   ├── trunk.py                    # PairFormer binding trunk (triangle updates)
│   ├── heads.py                    # Affinity + Structure + Generative heads
│   └── bindfm.py                   # Complete BindFM model + BindFMConfig
│
├── data/
│   ├── parsers.py                  # SMILES (RDKit), PDB (biopython), SELEX, SDF
│   └── dataset.py                  # Stage 0–3 curriculum Dataset classes
│
├── training/
│   └── train.py                    # BindFMTrainer: AMP, grad accum, curriculum
│
├── inference/
│   └── api.py                      # BindFMPredictor: clean public API
│
├── benchmarks/
│   └── evaluate.py                 # 9 standardized benchmark evaluations
│
├── configs/
│   ├── training_configs.yaml       # All hyperparameters (3 sizes × 4 stages)
│   └── config_loader.py
│
├── scripts/
│   ├── download_data.sh            # Master data download
│   ├── download_pdb_subset.py      # RCSB PDB API downloader
│   ├── build_affinity_index.py     # Merge all affinity sources → unified CSV
│   └── preprocessing_utils.py     # 9 preprocessing routines
│
├── notebooks/
│   └── BindFM_MicroRun.ipynb       # Colab/Kaggle micro-run (T4, ~40 min)
│
├── quickstart.py                   # End-to-end smoke test (real molecules)
├── Makefile                        # All operations as one-liners
├── requirements.txt
├── CITATION.cff
└── CONTRIBUTING.md
```

---

## Comparison to Related Work

| System | Modalities | Affinity | Structure | Generation | Aptamers | Open |
|--------|-----------|:--------:|:---------:|:----------:|:--------:|:----:|
| **BindFM** | All 5 | ✓ | ✓ | ✓ | ✓ (modified) | ✓ |
| AlphaFold3 | Protein+SM+NA | ✗ | ✓ | ✗ | partial | ✗ |
| Boltz-2 | Protein+SM | ✓ | ✓ | ✗ | ✗ | ✓ |
| DiffDock | Protein+SM | ✗ | ✓ | ✗ | ✗ | ✓ |
| AptaBLE | Aptamer+Protein | partial | ✗ | ✗ | ✓ | ✓ |
| RoseTTAFold-All-Atom | Protein+SM+NA | ✗ | ✓ | ✗ | partial | ✓ |

The closest work is **EPT** (Nature Comms 2025) — an all-atom equivariant pretrained transformer. BindFM differs in: (1) explicit aptamer modification tokens, (2) co-trained generative head, (3) kinetic outputs (kon/koff), and (4) it's genuinely universal rather than protein-centric with nucleic acid support bolted on.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome.

Key areas where contributions are most valuable:
- **Modified nucleotide chemistry**: extending the modification token vocabulary
- **SELEX data processing**: parsing diverse raw SELEX experimental formats
- **Benchmark additions**: allosteric site annotations, aptamer SPR validation sets
- **Inference optimization**: batching strategies, ONNX export

---

## Citation

```bibtex
@software{bindfm2025,
  author       = {Abdullah, Hamza},
  title        = {{BindFM}: Universal Biomolecular Binding Foundation Model},
  year         = {2025},
  institution  = {Terminal Bio},
  url          = {https://github.com/iamhamzaabdullah/BindFM},
  note         = {SE(3)-equivariant · atom-level · five binding modalities · from scratch}
}
```

If you use BindFM in academic work, please also cite the core architectural papers:

<details>
<summary>Core citations (EGNN, AlphaFold2 PairFormer, OT-CFM, Kendall uncertainty)</summary>

```bibtex
@inproceedings{satorras2021egnn,
  title     = {{E(n)} Equivariant Graph Neural Networks},
  author    = {Satorras, Victor Garcia and Hoogeboom, Emiel and Welling, Max},
  booktitle = {ICML},
  year      = {2021}
}

@article{jumper2021alphafold,
  title   = {Highly accurate protein structure prediction with {AlphaFold}},
  author  = {Jumper, John and others},
  journal = {Nature},
  volume  = {596},
  pages   = {583--589},
  year    = {2021}
}

@article{lipman2022flow,
  title   = {Flow Matching for Generative Modeling},
  author  = {Lipman, Yaron and Chen, Ricky T. Q. and others},
  year    = {2022},
  journal = {arXiv:2210.02747}
}

@inproceedings{kendall2018uncertainty,
  title     = {Multi-Task Learning Using Uncertainty to Weigh Losses},
  author    = {Kendall, Alex and Gal, Yarin},
  booktitle = {CVPR},
  year      = {2018}
}
```

</details>

---

<div align="center">

**Terminal Bio · MIT License · 2025**

*The goal is a single model that knows what binds to what.*
*Every molecule. Every modality. One representation.*

</div>
