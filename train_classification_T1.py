"""Training entrypoint for DA6401 Assignment 2.

Usage:
    # Fresh start:
    python train.py --task classification \
        --data_root /path/to/Oxford-IIIT-Pet

    # Resume (same command, auto-detects resume file):
    python train.py --task classification \
        --data_root /path/to/Oxford-IIIT-Pet

    # Start fresh even if a resume file exists:
    python train.py --task classification \
        --data_root /path/to/Oxford-IIIT-Pet --no_resume
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models.classification import VGG11Classifier
from data.pets_dataset import build_classification_loaders

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
#  Device
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# =============================================================================
#  Checkpointing
# =============================================================================

def save_best_checkpoint(model: nn.Module, path: str):
    """Model weights only -- what the autograder loads."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  [ckpt] Best model saved -> {path}")


def save_resume_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                            epoch: int, best_val_acc: float,
                            phase: int, path: str):
    """Full training state for pause/resume. Saved every epoch."""
    resume_path = path.replace('.pth', '.resume.pth')
    Path(resume_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':        epoch,
        'phase':        phase,          # 1=frozen, 2=unfrozen
        'model_state':  model.state_dict(),
        'optim_state':  optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }, resume_path)
    print(f"  [ckpt] Resume state saved -> {resume_path}  (epoch {epoch})")


def load_resume_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                            path: str, device: torch.device):
    """Returns (start_epoch, best_val_acc, phase). (1, 0.0, 1) if no file."""
    resume_path = path.replace('.pth', '.resume.pth')
    if not Path(resume_path).exists():
        print(f"  No resume checkpoint at {resume_path} -- starting fresh.")
        return 1, 0.0, 1
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    start_epoch  = ckpt['epoch'] + 1
    best_val_acc = ckpt['best_val_acc']
    phase        = ckpt.get('phase', 1)
    print(f"  Resumed from  : {resume_path}")
    print(f"  Last epoch    : {ckpt['epoch']}  |  phase: {phase}")
    print(f"  Best val acc  : {best_val_acc:.4f}")
    print(f"  Resuming at   : epoch {start_epoch}")
    return start_epoch, best_val_acc, phase


# =============================================================================
#  LR schedule -- fixed two-phase cosine with independent epoch counters
#
#  Root cause of the plateau: the old code passed the GLOBAL epoch index
#  into warmup_cosine after unfreeze.  At epoch 11 that means progress=11/80
#  already 14% decayed; by epoch 40 the LR was 6e-6 instead of ~2e-4.
#  Fix: each phase has its own epoch counter that resets to 0 at phase start.
# =============================================================================

def cosine_lr(phase_epoch: int, warmup: int, total: int,
              peak: float, min_lr: float = 1e-6) -> float:
    """Cosine decay with linear warmup.
    phase_epoch: 0-indexed epoch *within* this phase."""
    if phase_epoch < warmup:
        return peak * (phase_epoch + 1) / warmup
    progress = (phase_epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
#  Backbone freeze / unfreeze
# =============================================================================

def freeze_backbone(model: nn.Module, up_to: int):
    for i, layer in enumerate(model.encoder.features):
        for p in layer.parameters():
            p.requires_grad_(i >= up_to)


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(True)


# =============================================================================
#  Classification training
# =============================================================================

def train_classification(args):
    device = get_device()
    print(f"Device: {device}")
    pin = torch.cuda.is_available()

    train_loader, val_loader = build_classification_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = VGG11Classifier(
        num_classes=args.num_classes,
        in_channels=3,
        dropout_p=args.dropout_p,
    ).to(device)

    # Optimizer created before loading checkpoint so state can be restored
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)

    # Resume or fresh start
    if args.no_resume:
        start_epoch, best_val_acc, phase = 1, 0.0, 1
        print("  --no_resume set: starting fresh.")
    else:
        start_epoch, best_val_acc, phase = load_resume_checkpoint(
            model, optimizer, args.classifier_ckpt, device)

    # Apply correct freeze state
    if phase == 1:
        freeze_backbone(model, up_to=args.freeze_blocks)
        print(f"  Phase 1: first {args.freeze_blocks} encoder blocks frozen "
              f"(epochs 1-{args.unfreeze_epoch}).")
    else:
        unfreeze_all(model)
        print("  Phase 2: all layers trainable.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args),
                   name="vgg11_classification", resume="allow")

    # Phase lengths
    phase1_total = args.unfreeze_epoch               # epochs 1 .. unfreeze_epoch
    phase2_total = args.epochs - args.unfreeze_epoch  # epochs unfreeze+1 .. end

    for epoch in range(start_epoch, args.epochs + 1):

        # ── Phase transition ────────────────────────────────────────────────
        if phase == 1 and epoch > args.unfreeze_epoch:
            phase = 2
            unfreeze_all(model)
            # Rebuild optimizer so Adam moments from frozen phase don't pollute
            optimizer = optim.AdamW(model.parameters(),
                                    lr=args.phase2_lr,
                                    weight_decay=args.weight_decay)
            print(f"  -> Phase 2 at epoch {epoch}: all layers unfrozen, "
                  f"peak LR = {args.phase2_lr:.1e}")

        # ── LR schedule (independent per phase) ────────────────────────────
        if phase == 1:
            phase_epoch = epoch - 1                   # 0-indexed within phase 1
            lr = cosine_lr(phase_epoch,
                           warmup=args.warmup_epochs,
                           total=phase1_total,
                           peak=args.lr)
        else:
            phase_epoch = epoch - args.unfreeze_epoch - 1  # 0-indexed within phase 2
            lr = cosine_lr(phase_epoch,
                           warmup=2,
                           total=phase2_total,
                           peak=args.phase2_lr)

        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        train_loss = train_acc = 0.0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()
            train_acc  += accuracy(logits, labels)

        train_loss /= len(train_loader)
        train_acc  /= len(train_loader)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = val_acc = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item()
                val_acc  += accuracy(logits, labels)

        val_loss /= len(val_loader)
        val_acc  /= len(val_loader)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] ph={phase} "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={lr:.2e}  time={elapsed:.1f}s"
        )

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "phase": phase,
                       "train/loss": train_loss, "train/acc": train_acc,
                       "val/loss": val_loss, "val/acc": val_acc, "lr": lr})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_checkpoint(model, args.classifier_ckpt)

        save_resume_checkpoint(model, optimizer, epoch, best_val_acc,
                                phase, args.classifier_ckpt)

    print(f"\nBest val accuracy : {best_val_acc:.4f}")
    print(f"Model checkpoint  : {args.classifier_ckpt}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


# =============================================================================
#  Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="DA6401 A2 -- Training")
    p.add_argument("--task", default="classification",
                   choices=["classification", "localization", "segmentation"])
    p.add_argument("--data_root", required=True)
    # Model
    p.add_argument("--num_classes",   type=int,   default=37)
    p.add_argument("--dropout_p",     type=float, default=0.6)
    # Training
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=32)
    # Phase 1 (frozen backbone)
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="Peak LR for phase 1 (frozen backbone)")
    p.add_argument("--warmup_epochs", type=int,   default=2)
    p.add_argument("--freeze_blocks", type=int,   default=4)
    p.add_argument("--unfreeze_epoch",type=int,   default=10)
    # Phase 2 (full network) -- independent schedule, resets at unfreeze
    p.add_argument("--phase2_lr",     type=float, default=3e-4,
                   help="Peak LR for phase 2 (all layers unfrozen). "
                        "Higher than phase1 because the full network benefits "
                        "from a fresh cosine cycle starting from a higher peak.")
    p.add_argument("--weight_decay",  type=float, default=5e-4)
    p.add_argument("--num_workers",   type=int,   default=4)
    # Resume control
    p.add_argument("--no_resume",     action="store_true",
                   help="Ignore any existing resume checkpoint and start fresh")
    # Checkpoints
    p.add_argument("--classifier_ckpt", default="checkpoints/classifier.pth")
    p.add_argument("--localizer_ckpt",  default="checkpoints/localizer.pth")
    p.add_argument("--unet_ckpt",       default="checkpoints/unet.pth")
    # W&B
    p.add_argument("--use_wandb",     action="store_true")
    p.add_argument("--wandb_project", default="da6401_a2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "classification":
        train_classification(args)
    else:
        raise NotImplementedError(f"Task '{args.task}' not yet implemented.")