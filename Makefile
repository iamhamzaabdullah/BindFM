# BindFM Makefile
# ================
# Common operations as one-liners.

.PHONY: quickstart test syntax benchmark train-small train-full \
        download-data format clean zip help

PYTHON    = python3
DATA_DIR  = ./data
CKPT_DIR  = ./checkpoints

# ── Verification ──────────────────────────────────────────────────────────────

quickstart:
	@echo "Running BindFM quickstart smoke test..."
	$(PYTHON) quickstart.py --device cpu --size small

quickstart-gpu:
	$(PYTHON) quickstart.py --device cuda --size small

test: quickstart

syntax:
	@$(PYTHON) -c "
import ast
from pathlib import Path
files = [f for f in Path('.').rglob('*.py') if '__pycache__' not in str(f)]
errors = sum(1 for f in files if not (lambda: (ast.parse(f.read_text()), 0) or (print(f'✗ {f}'), 1))()[1])
print(f'Syntax check: {len(files)} files, {errors} errors')
"

# ── Benchmarks ────────────────────────────────────────────────────────────────

benchmark:
	$(PYTHON) -c "
from benchmarks.evaluate import BindFMEvaluator
from model.bindfm import BindFM
model = BindFM.load('$(CKPT_DIR)/bindfm_stage3_best.pt')
evaluator = BindFMEvaluator(model, data_dir='$(DATA_DIR)')
results = evaluator.run_all()
evaluator.save_results(results, 'benchmark_results.json')
"

# ── Data Download ─────────────────────────────────────────────────────────────

download-data:
	bash scripts/download_data.sh --data-dir $(DATA_DIR)

build-index:
	$(PYTHON) scripts/download_pdb_subset.py \
		--output-dir $(DATA_DIR)/pdb \
		--max-structures 100000
	$(PYTHON) scripts/build_affinity_index.py --data-dir $(DATA_DIR)

# ── Training ──────────────────────────────────────────────────────────────────

train-small: $(DATA_DIR)
	for stage in 0 1 2 3; do \
		$(PYTHON) training/train.py \
			--config configs/training_configs.yaml \
			--key small_stage$$stage \
			--data-dir $(DATA_DIR) \
			--checkpoint-dir $(CKPT_DIR); \
	done

train-medium: $(DATA_DIR)
	for stage in 0 1 2 3; do \
		$(PYTHON) training/train.py \
			--config configs/training_configs.yaml \
			--key medium_stage$$stage \
			--data-dir $(DATA_DIR) \
			--checkpoint-dir $(CKPT_DIR); \
	done

train-full: $(DATA_DIR)
	for stage in 0 1 2 3; do \
		$(PYTHON) training/train.py \
			--config configs/training_configs.yaml \
			--key full_stage$$stage \
			--data-dir $(DATA_DIR) \
			--checkpoint-dir $(CKPT_DIR); \
	done

# ── Utilities ─────────────────────────────────────────────────────────────────

format:
	@which black >/dev/null 2>&1 && black . --line-length 100 || echo "Install black: pip install black"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name "*.pyo" -delete 2>/dev/null; true

zip:
	cd .. && zip -r BindFM.zip BindFM/ \
		--exclude "*/\.*" \
		--exclude "*/__pycache__/*" \
		--exclude "*.pyc" \
		--exclude "*/data/*" \
		--exclude "*/checkpoints/*"
	@echo "Created BindFM.zip"

help:
	@echo "BindFM Makefile targets:"
	@echo "  make quickstart       Run smoke test (CPU, small model)"
	@echo "  make quickstart-gpu   Run smoke test on GPU"
	@echo "  make syntax           Syntax check all Python files"
	@echo "  make benchmark        Run all benchmarks (requires trained checkpoint)"
	@echo "  make download-data    Download all training data"
	@echo "  make build-index      Build PDB and affinity indices"
	@echo "  make train-small      Train small model (4 stages)"
	@echo "  make train-medium     Train medium model"
	@echo "  make train-full       Train full model (8×A100)"
	@echo "  make format           Format code with black"
	@echo "  make clean            Remove __pycache__ and .pyc files"
	@echo "  make zip              Package repo as BindFM.zip"
