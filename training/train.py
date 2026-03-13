"""
BindFM Training Pipeline
-------------------------
4-stage curriculum training from scratch.

Stage 0: Molecular geometry pretraining
  Learn atom-level geometry and chemistry from single molecules.
  No binding signal. Just what molecules look like in 3D.

Stage 1: Complex structural pretraining
  Learn what bound complexes look like.
  Flow matching on all 5 binding modalities.

Stage 2: Binding affinity training
  Introduce Kd/Ki/IC50 signal across all assay types.
  Multi-task regression + binary classification.

Stage 3: End-to-end joint fine-tuning + generation
  Unfreeze all parameters. Add generative head loss.
  Uncertainty-weighted multi-task objective.

Usage:
    python3 training/train.py \\
        --config configs/training_configs.yaml \\
        --key small_stage0 \\
        --data-dir ./data \\
        --checkpoint-dir ./checkpoints
"""

from __future__ import annotations
import os
import sys
import math
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.bindfm import BindFM, BindFMConfig
from model.tokenizer import BindingPair, EntityType
from data.dataset import build_dataloader, collate_binding_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bindfm.train")


# ── Training Config ───────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """All training hyperparameters for one curriculum stage."""

    stage:              int   = 0

    # Optimization
    lr:                 float = 1e-4
    min_lr:             float = 1e-6
    weight_decay:       float = 0.01
    grad_clip:          float = 1.0
    warmup_steps:       int   = 2000
    total_steps:        int   = 200_000

    # Batch
    batch_size:         int   = 8
    accumulate_grad:    int   = 4     # effective batch = batch_size * accumulate_grad

    # Stage-specific encoder LR scaling
    # In stages 2–3, encoder LR is reduced to prevent catastrophic forgetting
    encoder_lr_scale:   float = 1.0

    # Checkpointing
    save_every:         int   = 5_000
    eval_every:         int   = 1_000
    checkpoint_dir:     str   = "./checkpoints"

    # Mixed precision (disable on CPU)
    use_amp:            bool  = True

    # Stage 2+: negative example augmentation ratio
    decoy_ratio:        float = 0.2

    # Freeze trunk in stage 0 (only train encoder)
    freeze_trunk:       bool  = False

    # Freeze encoder in stage 2 (fine-tune trunk + heads only)
    freeze_encoder:     bool  = False


# ── Warmup + Cosine LR Schedule ───────────────────────────────────────────────

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay."""
    warmup = LinearLR(
        optimizer,
        start_factor = 1e-6,
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max   = total_steps - warmup_steps,
        eta_min = 0.0,
    )
    return SequentialLR(optimizer, [warmup, cosine],
                        milestones=[warmup_steps])


# ── Flow Matching Noise ───────────────────────────────────────────────────────

def add_flow_noise(coords: torch.Tensor, t: float) -> torch.Tensor:
    """
    OT-CFM noise for structure flow matching.
    Interpolate between Gaussian noise (t=0) and clean coords (t=1).
    Returns noisy coordinates at time t.
    """
    noise = torch.randn_like(coords) * 10.0  # initial Gaussian noise scale
    return (1 - t) * noise + t * coords


def flow_velocity_target(
    noisy: torch.Tensor, clean: torch.Tensor, t: float
) -> torch.Tensor:
    """OT-CFM velocity target: v = (clean - noisy) / (1 - t + eps)."""
    return (clean - noisy) / max(1 - t, 1e-4)


# ── Per-Step Training Functions ───────────────────────────────────────────────

def step_stage0(
    model: BindFM,
    pair:  BindingPair,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Stage 0 training step: coordinate denoising.
    Uses structure head as the denoising network.
    """
    if pair.entity_a.coords is None or pair.complex_coords is None:
        return None

    t = torch.rand(1).item()

    # Noisy coords at time t
    noisy_a = add_flow_noise(pair.entity_a.coords.to(device), t)
    noisy_b = add_flow_noise(pair.entity_b.coords.to(device), t)

    # Ground truth velocity
    vel_true_a = flow_velocity_target(noisy_a,
                                      pair.complex_coords[:pair.entity_a.n_atoms].to(device), t)
    vel_true_b = flow_velocity_target(noisy_b,
                                      pair.complex_coords[pair.entity_a.n_atoms:].to(device), t)

    t_tensor = torch.tensor(t, device=device)
    out = model.forward(
        pair.entity_a.atom_feats.to(device),
        pair.entity_a.edge_index.to(device),
        pair.entity_a.edge_feats.to(device),
        pair.entity_a.coords.to(device),
        pair.entity_b.atom_feats.to(device),
        pair.entity_b.edge_index.to(device),
        pair.entity_b.edge_feats.to(device),
        pair.entity_b.coords.to(device),
        run_affinity  = False,
        run_structure = True,
        noisy_coords_a = noisy_a,
        noisy_coords_b = noisy_b,
        flow_t         = t_tensor,
    )

    loss_dict = model.loss_fn(
        vel_pred_a = out["structure"]["vel_a"],
        vel_pred_b = out["structure"]["vel_b"],
        vel_true_a = vel_true_a,
        vel_true_b = vel_true_b,
    )
    return loss_dict["total"]


def step_stage1(
    model: BindFM,
    pair:  BindingPair,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Stage 1 training step: complex structure prediction.
    Same as stage 0 but with real binding complex coordinates.
    """
    return step_stage0(model, pair, device)


def step_stage2(
    model: BindFM,
    pair:  BindingPair,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Stage 2 training step: binding affinity regression + classification.
    """
    d = device
    has_kd       = pair.log_kd is not None
    has_is_binder= pair.is_binder is not None

    if not has_kd and not has_is_binder:
        return None

    out = model.forward(
        pair.entity_a.atom_feats.to(d),
        pair.entity_a.edge_index.to(d),
        pair.entity_a.edge_feats.to(d),
        (pair.entity_a.coords.to(d) if pair.entity_a.coords is not None else None),
        pair.entity_b.atom_feats.to(d),
        pair.entity_b.edge_index.to(d),
        pair.entity_b.edge_feats.to(d),
        (pair.entity_b.coords.to(d) if pair.entity_b.coords is not None else None),
        run_affinity  = True,
        run_structure = False,
    )

    log_kd = (torch.tensor(pair.log_kd, device=d, dtype=torch.float32)
              if has_kd else None)
    is_bdr = (torch.tensor(float(pair.is_binder), device=d, dtype=torch.float32)
              if has_is_binder else None)

    loss_dict = model.loss_fn(
        affinity_pred = out["affinity"],
        log_kd        = log_kd,
        is_binder     = is_bdr,
    )
    return loss_dict["total"]


def step_stage3(
    model: BindFM,
    pair:  BindingPair,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Stage 3: joint step combining affinity + structure + generation losses.
    Each component is run stochastically to keep GPU memory manageable.
    """
    losses = []

    # Always run affinity
    aff_loss = step_stage2(model, pair, device)
    if aff_loss is not None:
        losses.append(aff_loss)

    # 50% chance: also run structure
    if pair.complex_coords is not None and torch.rand(1).item() > 0.5:
        struct_loss = step_stage0(model, pair, device)
        if struct_loss is not None:
            losses.append(struct_loss)

    # 30% chance: run generation (condition on entity_b, generate entity_a)
    if torch.rand(1).item() > 0.7 and pair.entity_b.n_atoms < 500:
        gen_loss = _step_generation(model, pair, device)
        if gen_loss is not None:
            losses.append(gen_loss)

    if not losses:
        return None

    return sum(losses)


def _step_generation(
    model: BindFM,
    pair:  BindingPair,
    device: str,
) -> Optional[torch.Tensor]:
    """Generation sub-step: flow matching in (atom_feat, coord) space."""
    d    = device
    mol_a = pair.entity_a

    if mol_a.n_atoms < 2:
        return None

    t     = torch.rand(1).item()
    noise_feats  = torch.randn(mol_a.n_atoms, mol_a.atom_feats.shape[1], device=d)
    noise_coords = torch.randn(mol_a.n_atoms, 3, device=d) * 5.0

    noisy_feats  = (1 - t) * noise_feats  + t * mol_a.atom_feats.to(d)
    noisy_coords = (1 - t) * noise_coords + t * (mol_a.coords.to(d)
                                                   if mol_a.coords is not None
                                                   else noise_coords)

    vel_feats_true  = mol_a.atom_feats.to(d) - noise_feats
    vel_coords_true = (mol_a.coords.to(d) if mol_a.coords is not None
                       else torch.zeros_like(noisy_coords)) - noise_coords

    modality = torch.tensor(int(mol_a.entity_type), device=d)
    t_tensor = torch.tensor(t, device=d)
    log_kd_t = (torch.tensor(pair.log_kd, device=d, dtype=torch.float32)
                if pair.log_kd is not None else None)

    out = model.forward(
        pair.entity_b.atom_feats.to(d),
        pair.entity_b.edge_index.to(d),
        pair.entity_b.edge_feats.to(d),
        (pair.entity_b.coords.to(d) if pair.entity_b.coords is not None else None),
        pair.entity_b.atom_feats.to(d),
        pair.entity_b.edge_index.to(d),
        pair.entity_b.edge_feats.to(d),
        (pair.entity_b.coords.to(d) if pair.entity_b.coords is not None else None),
        run_affinity  = False,
        run_gen       = True,
        gen_modality       = modality,
        gen_noisy_feats    = noisy_feats,
        gen_noisy_coords   = noisy_coords,
        flow_t             = t_tensor,
        gen_log_kd_target  = log_kd_t,
    )

    loss_dict = model.loss_fn(
        gen_vel_feats_pred  = out["generation"]["vel_feats"],
        gen_vel_feats_true  = vel_feats_true,
        gen_vel_coords_pred = out["generation"]["vel_coords"],
        gen_vel_coords_true = vel_coords_true,
    )
    return loss_dict["total"]


STEP_FN = {0: step_stage0, 1: step_stage1, 2: step_stage2, 3: step_stage3}


# ── Main Training Loop ────────────────────────────────────────────────────────

class BindFMTrainer:
    """
    Full curriculum trainer for BindFM.

    Handles:
      - Mixed precision (AMP)
      - Gradient accumulation
      - Checkpoint save / load / resume
      - Per-stage learning rate control
      - Logging
    """

    def __init__(
        self,
        model:       BindFM,
        train_cfg:   TrainingConfig,
        model_cfg:   BindFMConfig,
        data_dir:    str,
        device:      str = "cuda",
    ):
        self.model     = model.to(device)
        self.cfg       = train_cfg
        self.model_cfg = model_cfg
        self.device    = device
        self.data_dir  = data_dir
        self.step      = 0
        self.best_val  = float("inf")

        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

        # Parameter groups with separate LR for encoder
        enc_params  = list(model.encoder.parameters())
        rest_params = [p for p in model.parameters()
                       if not any(p is ep for ep in enc_params)]

        self.optimizer = AdamW([
            {"params": enc_params,  "lr": train_cfg.lr * train_cfg.encoder_lr_scale},
            {"params": rest_params, "lr": train_cfg.lr},
        ], weight_decay=train_cfg.weight_decay)

        self.scheduler = build_scheduler(
            self.optimizer,
            train_cfg.warmup_steps,
            train_cfg.total_steps,
        )

        # Mixed precision
        self.use_amp = train_cfg.use_amp and device != "cpu" and torch.cuda.is_available()
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Apply stage-specific freezing
        if train_cfg.freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            logger.info("Encoder frozen for this stage.")

        if train_cfg.freeze_trunk:
            for p in model.trunk.parameters():
                p.requires_grad_(False)
            logger.info("Trunk frozen for this stage.")

    def _get_dataloader(self, split: str):
        return build_dataloader(
            stage       = self.cfg.stage,
            data_dir    = self.data_dir,
            split       = split,
            batch_size  = self.cfg.batch_size,
            num_workers = min(4, os.cpu_count() or 1),
        )

    def save(self, tag: str = "latest"):
        path = os.path.join(
            self.cfg.checkpoint_dir,
            f"bindfm_stage{self.cfg.stage}_{tag}.pt"
        )
        torch.save({
            "step":           self.step,
            "model_config":   self.model_cfg,
            "train_config":   self.cfg,
            "model_state":    self.model.state_dict(),
            "optimizer_state":self.optimizer.state_dict(),
            "scheduler_state":self.scheduler.state_dict(),
            "best_val":       self.best_val,
        }, path)
        logger.info(f"Saved checkpoint → {path}")
        return path

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.step     = ckpt["step"]
        self.best_val = ckpt.get("best_val", float("inf"))
        logger.info(f"Resumed from step {self.step} ({path})")

    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from:
            self.load(resume_from)

        step_fn     = STEP_FN[self.cfg.stage]
        train_loader= self._get_dataloader("train")

        logger.info(
            f"Training Stage {self.cfg.stage} | "
            f"device={self.device} | "
            f"total_steps={self.cfg.total_steps} | "
            f"AMP={self.use_amp}"
        )

        accum_loss  = 0.0
        accum_count = 0
        t0          = time.time()

        self.model.train()
        self.optimizer.zero_grad()

        while self.step < self.cfg.total_steps:
            for batch in train_loader:
                if self.step >= self.cfg.total_steps:
                    break

                for pair in batch:
                    try:
                        if self.use_amp:
                            with torch.cuda.amp.autocast():
                                loss = step_fn(self.model, pair, self.device)
                        else:
                            loss = step_fn(self.model, pair, self.device)

                        if loss is None or not torch.isfinite(loss):
                            continue

                        scaled = loss / self.cfg.accumulate_grad
                        if self.use_amp:
                            self.scaler.scale(scaled).backward()
                        else:
                            scaled.backward()

                        accum_loss  += loss.item()
                        accum_count += 1

                    except Exception as e:
                        logger.debug(f"Step error (skipped): {e}")
                        continue

                    if accum_count % self.cfg.accumulate_grad == 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)

                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.grad_clip
                        )

                        if self.use_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.step += 1

                        # Logging
                        if self.step % 50 == 0:
                            elapsed = time.time() - t0
                            lr      = self.optimizer.param_groups[0]["lr"]
                            avg_loss= accum_loss / max(accum_count, 1)
                            logger.info(
                                f"Step {self.step:6d}/{self.cfg.total_steps} | "
                                f"Loss={avg_loss:.4f} | LR={lr:.2e} | "
                                f"{elapsed:.0f}s"
                            )
                            accum_loss  = 0.0
                            accum_count = 0

                        # Validation
                        if self.step % self.cfg.eval_every == 0:
                            val_loss = self._validate()
                            if val_loss < self.best_val:
                                self.best_val = val_loss
                                self.save("best")

                        # Checkpoint
                        if self.step % self.cfg.save_every == 0:
                            self.save(f"step{self.step:07d}")

        self.save("final")
        logger.info(f"Stage {self.cfg.stage} training complete.")

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation for up to 100 steps."""
        self.model.eval()
        val_loader  = self._get_dataloader("val")
        step_fn     = STEP_FN[self.cfg.stage]
        total_loss  = 0.0
        count       = 0
        max_val_steps = 100

        for batch in val_loader:
            for pair in batch:
                try:
                    loss = step_fn(self.model, pair, self.device)
                    if loss is not None and torch.isfinite(loss):
                        total_loss += loss.item()
                        count      += 1
                except Exception:
                    pass
            if count >= max_val_steps:
                break

        val_loss = total_loss / max(count, 1)
        logger.info(f"Validation loss @ step {self.step}: {val_loss:.4f}")
        self.model.train()
        return val_loss


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BindFM Training")
    p.add_argument("--config",          required=True,
                   help="Path to training_configs.yaml")
    p.add_argument("--key",             required=True,
                   help="Config key, e.g. small_stage0")
    p.add_argument("--data-dir",        default="./data",
                   help="Root data directory")
    p.add_argument("--checkpoint-dir",  default="./checkpoints")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available()
                                                       else "cpu")
    p.add_argument("--resume",          default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--prev-checkpoint", default=None,
                   help="Path to checkpoint from previous stage (for warm-start)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load configs
    from configs.config_loader import load_config
    model_cfg, train_cfg = load_config(args.config, args.key)

    # Override paths from CLI
    train_cfg.checkpoint_dir = args.checkpoint_dir

    # Build model
    model = BindFM(model_cfg)

    # Warm-start from previous stage
    if args.prev_checkpoint:
        ckpt = torch.load(args.prev_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state", ckpt.get("state_dict", {})),
                               strict=False)
        logger.info(f"Warm-started from {args.prev_checkpoint}")

    # Build trainer
    trainer = BindFMTrainer(
        model      = model,
        train_cfg  = train_cfg,
        model_cfg  = model_cfg,
        data_dir   = args.data_dir,
        device     = args.device,
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
