#!/usr/bin/env bash
# ================================================================
# BindFM — Fresh GitHub Push Script
# ================================================================
# Usage:
#   bash scripts/github_push.sh
#
# Requirements:
#   - Git installed
#   - GitHub Personal Access Token set as GH_TOKEN environment variable
#     OR be ready to enter credentials when prompted
#
# What this does:
#   1. Initializes git (if not already a repo)
#   2. Configures remote to https://github.com/iamhamzaabdullah/BindFM
#   3. Creates a fresh orphan main branch (clean history)
#   4. Commits everything with a clean initial commit
#   5. Force-pushes to main
# ================================================================

set -e

REPO_URL="https://github.com/iamhamzaabdullah/BindFM.git"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  BindFM → GitHub Fresh Push"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Git init ──────────────────────────────────────────
if [ ! -d .git ]; then
    echo "→ Initializing git repository..."
    git init
else
    echo "→ Git repository already initialized."
fi

# ── .gitignore ────────────────────────────────────────
cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Data (too large for git — use DVC or download scripts)
data/
checkpoints/
*.pt
*.pkl
*.h5
*.hdf5

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Env
.env
.venv
venv/
env/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Results / outputs
benchmark_results.json
*.log
wandb/
runs/
GITIGNORE

echo "→ .gitignore written."

# ── User identity ─────────────────────────────────────
git config user.name  "Hamza Abdullah"
git config user.email "hamza@terminalbio.io"

# ── Orphan branch ─────────────────────────────────────
echo "→ Creating fresh orphan branch 'main'..."
git checkout --orphan main 2>/dev/null || true

# Stage everything
git add -A

# ── Initial commit ────────────────────────────────────
git commit -m "BindFM: Universal Biomolecular Binding Foundation Model

Architecture:
- SE(3)-equivariant EGNN atom encoder (shared weights, both entities)
- PairFormer binding trunk (32 layers, triangle multiplicative updates)
- Three co-trained heads: Affinity + Structure + Generative
- 197-dim universal atom featurizer across all molecular types

Capabilities:
- All 5 binding modalities: protein-SM, protein-protein, protein-NA, NA-SM, NA-NA
- Therapeutic aptamers with full modification support (LNA, 2'F, 2'OMe, PS)
- Binding affinity: log Kd, P(bind), kon/koff, half-life, uncertainty
- 3D complex structure prediction via SE(3)-equivariant flow matching
- De novo binder generation conditioned on target and desired Kd

Training:
- 4-stage curriculum: geometry → structure → affinity → joint
- Sources: PDBbind, BindingDB, ChEMBL, AptaBase, SKEMPI2, RNAcompete, CovalentDB
- Mixed precision (bf16), gradient accumulation, cosine LR schedule

Models: Small (~2.1M), Medium (~45M), Full (~380M params)
Micro-run: Free T4 GPU on Colab/Kaggle (~40 min)

Built from scratch at Terminal Bio.
Lead Researcher: Hamza Abdullah"

# ── Remote ────────────────────────────────────────────
echo "→ Configuring remote..."
git remote remove origin 2>/dev/null || true

# Use token if available
if [ -n "$GH_TOKEN" ]; then
    REMOTE_URL="https://${GH_TOKEN}@github.com/iamhamzaabdullah/BindFM.git"
    echo "→ Using GH_TOKEN for authentication."
else
    REMOTE_URL="$REPO_URL"
    echo "→ No GH_TOKEN found. You will be prompted for credentials."
    echo "  TIP: Set GH_TOKEN=<your_personal_access_token> to skip the prompt."
fi

git remote add origin "$REMOTE_URL"
git branch -M main

# ── Push ──────────────────────────────────────────────
echo "→ Pushing to GitHub (force)..."
git push -u origin main --force

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ Successfully pushed to:"
echo "    https://github.com/iamhamzaabdullah/BindFM"
echo "═══════════════════════════════════════════════════"
echo ""
