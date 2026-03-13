#!/usr/bin/env bash
# BindFM — GitHub Push Instructions
# Run these commands from the bindfm/ directory

echo "=========================================="
echo "  BindFM GitHub Setup"
echo "  Repo: https://github.com/iamhamzaabdullah/BindFM"
echo "=========================================="

# Step 1: Initialize git (if not already done)
git init
git branch -M main

# Step 2: Configure identity
git config user.name "Hamza Abdullah"
git config user.email "your@email.com"     # replace with your email

# Step 3: Add remote
git remote add origin https://github.com/iamhamzaabdullah/BindFM.git

# Step 4: Stage everything
git add .

# Step 5: First commit
git commit -m "Initial release: BindFM universal binding foundation model

BindFM: A from-scratch SE(3)-equivariant foundation model for
universal biomolecular binding prediction across all 5 modalities.

Architecture:
- Universal atom tokenizer (197-dim, all elements + modifications)
- SE(3)-equivariant encoder (shared across both entities)
- PairFormer binding trunk (triangle attention)
- Three co-trained heads: affinity, structure, generative

Capabilities unique to BindFM:
- Aptamer-protein Kd prediction (DNA/RNA, with modifications)
- G-quadruplex topology conditioning
- Allosteric binding prediction
- De novo binder generation via flow matching
- kon/koff kinetics
- 5/5 binding modalities in one model

Codebase:
- 8,700+ lines, 29 Python files, 0 syntax errors
- Full data pipeline: PDB, BindingDB, ChEMBL, AptaBase, SKEMPI2
- 4-stage curriculum training (Stage 0-3)
- 9 benchmark evaluations
- Colab/Kaggle micro-run notebook
- CI via GitHub Actions

Lead: Hamza Abdullah @ Terminal Bio
License: MIT"

# Step 6: Push
git push -u origin main

echo ""
echo "Done. Repo live at: https://github.com/iamhamzaabdullah/BindFM"
echo ""
echo "Next: Enable GitHub Actions in the repo settings,"
echo "then the CI badge will go green after first push."
