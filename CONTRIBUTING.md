# Contributing to BindFM

Thank you for your interest in contributing to BindFM.
This project is led by **Hamza Abdullah** at [Terminal Bio](https://terminalbio.io).

---

## Ways to Contribute

### High-value contributions

**1. New data parsers**
- RNAcompete format improvements
- SELEX FASTQ pipeline hardening
- New aptamer databases (Aptagen, SomaLogic)

**2. Architecture improvements**
- Proper atom-feature → sequence decoder for the generative head
- Conformational ensemble support in the affinity head
- Covalent bond warhead geometry encoding

**3. Benchmark additions**
- CASP16 RNA structure track integration
- Aptamer SPR validation pipeline
- Cooperative/multivalent binding cases

**4. Compute contributions**
- Running curriculum stages and sharing checkpoints
- Ablation experiments with results
- Benchmark numbers on held-out sets

---

## Development Setup

```bash
git clone https://github.com/iamhamzaabdullah/BindFM
cd BindFM
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, black, mypy
python3 quickstart.py                  # must pass before any PR
```

---

## Code Standards

- **Black** formatting: `black .` before committing
- **Type hints** on all public functions
- **Docstrings** on all classes and public methods
- All new code must pass `python3 quickstart.py`

---

## PR Process

1. Open an issue first describing what you plan to change
2. Fork → branch named `feature/your-description` or `fix/issue-number`
3. Run `python3 quickstart.py --full` — all tests must pass
4. PR description must include: what changed, why, and any benchmark numbers

---

## Areas Where We Especially Need Help

| Area | Status | Notes |
|------|--------|-------|
| RDKit integration tests | Open | Need CI with RDKit installed |
| RNAcompete parser | Partial | Several GEO formats differ |
| G-quadruplex training data | Missing | Need curated G4-ligand set |
| Kinetics (kon/koff) data | Sparse | CovalentDB + manual curation |
| Allosteric test set | Minimal | ASD curation needed |
| Generation quality metrics | None | Need SPR-validated sequences |

---

## Contact

- GitHub Issues for bugs and feature requests
- Lead: **Hamza Abdullah** — Terminal Bio
- GitHub: [@iamhamzaabdullah](https://github.com/iamhamzaabdullah)
