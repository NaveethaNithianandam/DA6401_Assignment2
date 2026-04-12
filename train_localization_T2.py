"""Training entrypoint for DA6401 Assignment 2.

Usage:
    # Task 1 -- classification (fresh start):
    python train.py --task classification --data_root /path/to/pets

    # Task 2 -- localization (uses saved classifier encoder as init):
    python train.py --task localization --data_root /path/to/pets

    # Resume any task (same command, auto-detects .resume.pth):
    python train.py --task <task> --data_root /path/to/pets

    # Force fresh start ignoring existing resume file:
    python train.py --task <task> --data_root /path/to/pets --no_resume

Checkpoints:
    checkpoints/classifier.pth / classifier.resume.pth
    checkpoints/localizer.pth  / localizer.resume.pth
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss
from data.pets_dataset import build_classification_loaders, build_localization_loaders, build_segmentation_loaders

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
    return (logits.argmax(dim=1) == labels).float().mean().item()


# =============================================================================
#  Checkpointing
# =============================================================================

def save_best_checkpoint(model: nn.Module, path: str):
    """Model weights only -- what the autograder loads."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  [ckpt] Best model saved -> {path}")


def save_resume_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                            epoch: int, best_metric: float,
                            phase: int, path: str):
    """Full training state for pause/resume. Called every epoch."""
    resume_path = path.replace('.pth', '.resume.pth')
    Path(resume_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':       epoch,
        'phase':       phase,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'best_metric': best_metric,
    }, resume_path)
    print(f"  [ckpt] Resume state saved -> {resume_path}  (epoch {epoch})")


def load_resume_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                            path: str, device: torch.device):
    """Returns (start_epoch, best_metric, phase). (1, init_metric, 1) if no file."""
    resume_path = path.replace('.pth', '.resume.pth')
    if not Path(resume_path).exists():
        print(f"  No resume checkpoint at {resume_path} -- starting fresh.")
        return 1, None, 1
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    start_epoch  = ckpt['epoch'] + 1
    best_metric  = ckpt['best_metric']
    phase        = ckpt.get('phase', 1)
    print(f"  Resumed from  : {resume_path}")
    print(f"  Last epoch    : {ckpt['epoch']}  |  phase: {phase}")
    print(f"  Best metric   : {best_metric:.4f}")
    print(f"  Resuming at   : epoch {start_epoch}")
    return start_epoch, best_metric, phase


# =============================================================================
#  LR schedule (two independent cosine phases, counters reset at transition)
# =============================================================================

def cosine_lr(phase_epoch: int, warmup: int, total: int,
              peak: float, min_lr: float = 1e-6) -> float:
    if phase_epoch < warmup:
        return peak * (phase_epoch + 1) / warmup
    progress = (phase_epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


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
#  Task 1 -- Classification
# =============================================================================

def train_classification(args):
    device = get_device()
    print(f"Device: {device}")
    pin = torch.cuda.is_available()

    train_loader, val_loader = build_classification_loaders(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=pin,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = VGG11Classifier(num_classes=args.num_classes,
                             in_channels=3, dropout_p=args.dropout_p).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)

    if args.no_resume:
        start_epoch, best_metric, phase = 1, 0.0, 1
        print("  --no_resume: starting fresh.")
    else:
        start_epoch, best_metric, phase = load_resume_checkpoint(
            model, optimizer, args.classifier_ckpt, device)
        if best_metric is None:
            best_metric = 0.0

    if phase == 1:
        freeze_backbone(model, up_to=args.freeze_blocks)
        print(f"  Phase 1: encoder blocks 0-{args.freeze_blocks-1} frozen.")
    else:
        unfreeze_all(model)
        print("  Phase 2: all layers trainable.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    phase1_total = args.unfreeze_epoch
    phase2_total = args.epochs - args.unfreeze_epoch

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args),
                   name="vgg11_classification", resume="allow")

    for epoch in range(start_epoch, args.epochs + 1):
        if phase == 1 and epoch > args.unfreeze_epoch:
            phase = 2
            unfreeze_all(model)
            optimizer = optim.AdamW(model.parameters(), lr=args.phase2_lr,
                                     weight_decay=args.weight_decay)
            print(f"  -> Phase 2 at epoch {epoch}: peak LR = {args.phase2_lr:.1e}")

        lr = cosine_lr(
            epoch - 1 if phase == 1 else epoch - args.unfreeze_epoch - 1,
            warmup=args.warmup_epochs if phase == 1 else 2,
            total=phase1_total if phase == 1 else phase2_total,
            peak=args.lr if phase == 1 else args.phase2_lr,
        )
        set_lr(optimizer, lr)

        model.train()
        tr_loss = tr_acc = 0.0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            tr_loss += loss.item()
            tr_acc  += accuracy(logits, labels)
        tr_loss /= len(train_loader)
        tr_acc  /= len(train_loader)

        model.eval()
        va_loss = va_acc = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                va_loss += criterion(logits, labels).item()
                va_acc  += accuracy(logits, labels)
        va_loss /= len(val_loader)
        va_acc  /= len(val_loader)

        print(f"Epoch [{epoch:03d}/{args.epochs}] ph={phase} "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  "
              f"lr={lr:.2e}  time={time.time()-t0:.1f}s")

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "phase": phase,
                       "train/loss": tr_loss, "train/acc": tr_acc,
                       "val/loss": va_loss, "val/acc": va_acc, "lr": lr})

        if va_acc > best_metric:
            best_metric = va_acc
            save_best_checkpoint(model, args.classifier_ckpt)
        save_resume_checkpoint(model, optimizer, epoch, best_metric,
                                phase, args.classifier_ckpt)

    print(f"\nBest val accuracy : {best_metric:.4f}")
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


# =============================================================================
#  Task 2 -- Localization
# =============================================================================

def mean_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute mean IoU between predicted and target boxes (cxcywh, pixel space)."""
    def to_xyxy(b):
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)

    p = to_xyxy(pred)
    t = to_xyxy(target)
    ix1 = torch.max(p[:, 0], t[:, 0])
    iy1 = torch.max(p[:, 1], t[:, 1])
    ix2 = torch.min(p[:, 2], t[:, 2])
    iy2 = torch.min(p[:, 3], t[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    ap = (pred[:, 2] * pred[:, 3]).clamp(0)
    at = (target[:, 2] * target[:, 3]).clamp(0)
    union = ap + at - inter + eps
    return (inter / union).mean().item()


def train_localization(args):
    device = get_device()
    print(f"Device: {device}")
    pin = torch.cuda.is_available()

    train_loader, val_loader = build_localization_loaders(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=pin,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = VGG11Localizer(in_channels=3, dropout_p=args.dropout_p).to(device)

    # Load classification encoder weights before anything else.
    # The encoder features from 37-class pet classification transfer directly
    # to localization; we never update them (frozen throughout).
    model.load_encoder_from_classifier(args.classifier_ckpt, device)
    model.freeze_encoder()

    # Only the regression head (12.8M params) is trained.
    # The previous 4096x4096 head (119M params) memorised training bboxes
    # perfectly (train_mIoU=0.73, val_mIoU=0.21 -- 3.5x gap).
    # Tiny head + frozen encoder is the correct strategy for this dataset size.
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase2_lr, weight_decay=args.weight_decay,
    )

    if args.no_resume:
        start_epoch, best_metric = 1, 0.0
        print("  --no_resume: starting fresh. Encoder frozen, head only.")
    else:
        start_epoch, best_metric, _ = load_resume_checkpoint(
            model, optimizer, args.localizer_ckpt, device)
        if best_metric is None:
            best_metric = 0.0
        model.freeze_encoder()  # re-freeze after state restore
        print("  Encoder re-frozen after resume.")

    # Combined loss: SmoothL1 + IoU (MSE + IoU as required by assignment)
    # SmoothL1(beta=10)/224 gives ~0.05-0.20 for typical 20-50px errors,
    # matching IoU loss scale in [0,1]. IoU alone has zero gradient when
    # boxes don't overlap; SmoothL1 always provides a pull signal.
    iou_loss_fn = IoULoss(reduction="mean")
    mse_loss_fn = nn.SmoothL1Loss(beta=10.0)
    COORD_SCALE = 224.0

    mse_weight = args.mse_weight
    iou_weight = args.iou_weight

    # Single cosine schedule over all epochs (no freeze/unfreeze phases)
    total_epochs = args.epochs

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args),
                   name="vgg11_localization", resume="allow")

    for epoch in range(start_epoch, args.epochs + 1):
        lr = cosine_lr(epoch - 1, warmup=args.warmup_epochs,
                       total=total_epochs, peak=args.phase2_lr)
        set_lr(optimizer, lr)

        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        model.encoder.eval()   # keep BN stats frozen in encoder
        tr_loss = tr_mse = tr_iou_l = tr_miou = 0.0
        t0 = time.time()

        for imgs, boxes in train_loader:
            imgs, boxes = imgs.to(device), boxes.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            mse  = mse_loss_fn(pred / COORD_SCALE, boxes / COORD_SCALE)
            iou_l = iou_loss_fn(pred, boxes)
            loss = mse_weight * mse + iou_weight * iou_l
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            tr_loss  += loss.item()
            tr_mse   += mse.item()
            tr_iou_l += iou_l.item()
            with torch.no_grad():
                tr_miou += mean_iou(pred.detach(), boxes)

        n = len(train_loader)
        tr_loss /= n; tr_mse /= n; tr_iou_l /= n; tr_miou /= n

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        va_loss = va_mse = va_iou_l = va_miou = 0.0

        with torch.no_grad():
            for imgs, boxes in val_loader:
                imgs, boxes = imgs.to(device), boxes.to(device)
                pred = model(imgs)
                mse   = mse_loss_fn(pred / COORD_SCALE, boxes / COORD_SCALE)
                iou_l = iou_loss_fn(pred, boxes)
                va_loss  += (mse_weight * mse + iou_weight * iou_l).item()
                va_mse   += mse.item()
                va_iou_l += iou_l.item()
                va_miou  += mean_iou(pred, boxes)

        n = len(val_loader)
        va_loss /= n; va_mse /= n; va_iou_l /= n; va_miou /= n

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f}(sl1={tr_mse:.4f},iou={tr_iou_l:.3f})  "
            f"train_mIoU={tr_miou:.4f}  "
            f"val_loss={va_loss:.4f}  val_mIoU={va_miou:.4f}  "
            f"lr={lr:.2e}  time={time.time()-t0:.1f}s"
        )

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({"epoch": epoch,
                       "train/loss": tr_loss, "train/sl1": tr_mse,
                       "train/iou_loss": tr_iou_l, "train/mIoU": tr_miou,
                       "val/loss": va_loss, "val/sl1": va_mse,
                       "val/iou_loss": va_iou_l, "val/mIoU": va_miou,
                       "lr": lr})

        if va_miou > best_metric:
            best_metric = va_miou
            save_best_checkpoint(model, args.localizer_ckpt)
        save_resume_checkpoint(model, optimizer, epoch, best_metric,
                                2, args.localizer_ckpt)

    print(f"\nBest val mIoU : {best_metric:.4f}")
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

def train_segmentation(args):
    device = get_device()
    print(f"Device: {device}")
    pin = torch.cuda.is_available()

    train_loader, val_loader = build_segmentation_loaders(
        root=args.data_root, batch_size=args.batch_size // 2,  # seg uses more memory
        num_workers=args.num_workers, pin_memory=pin,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = VGG11UNet(num_classes=3, in_channels=3).to(device)

    # Warm-start encoder from classifier checkpoint
    if Path(args.classifier_ckpt).exists():
        cls_state = torch.load(args.classifier_ckpt, map_location=device)
        enc_state = {k.replace("encoder.", "", 1): v
                     for k, v in cls_state.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_state, strict=False)
        print(f"  Warm-started encoder from {args.classifier_ckpt}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)

    if args.no_resume:
        start_epoch, best_metric = 1, float("inf")
        print("  --no_resume: starting fresh.")
    else:
        start_epoch, best_metric, _ = load_resume_checkpoint(
            model, optimizer, args.unet_ckpt, device)
        if best_metric is None:
            best_metric = float("inf")

    # CrossEntropyLoss: per-pixel multi-class classification loss.
    # Oxford-IIIT Pet trimaps have 3 classes (fg/bg/boundary).
    # CE applies softmax + NLL per pixel; standard and well-justified choice.
    criterion = nn.CrossEntropyLoss()

    def pixel_accuracy(logits, masks):
        return (logits.argmax(dim=1) == masks).float().mean().item()

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args),
                   name="vgg11_segmentation", resume="allow")

    for epoch in range(start_epoch, args.epochs + 1):
        lr = cosine_lr(epoch - 1, warmup=args.warmup_epochs,
                       total=args.epochs, peak=args.lr)
        set_lr(optimizer, lr)

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_acc = 0.0
        t0 = time.time()

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)                    # [B, 3, H, W]
            loss = criterion(logits, masks)         # masks: [B, H, W] long
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            tr_loss += loss.item()
            tr_acc  += pixel_accuracy(logits.detach(), masks)

        tr_loss /= len(train_loader)
        tr_acc  /= len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        va_loss = va_acc = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                va_loss += criterion(logits, masks).item()
                va_acc  += pixel_accuracy(logits, masks)

        va_loss /= len(val_loader)
        va_acc  /= len(val_loader)

        print(f"Epoch [{epoch:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  "
              f"lr={lr:.2e}  time={time.time()-t0:.1f}s")

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({"epoch": epoch,
                       "train/loss": tr_loss, "train/acc": tr_acc,
                       "val/loss": va_loss, "val/acc": va_acc, "lr": lr})

        if va_loss < best_metric:
            best_metric = va_loss
            save_best_checkpoint(model, args.unet_ckpt)
        save_resume_checkpoint(model, optimizer, epoch, best_metric,
                                1, args.unet_ckpt)

    print(f"\nBest val loss : {best_metric:.4f}")
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

def parse_args():
    p = argparse.ArgumentParser(description="DA6401 A2 -- Training")
    p.add_argument("--task", default="classification",
                   choices=["classification", "localization", "segmentation"])
    p.add_argument("--data_root", required=True)
    # Model
    p.add_argument("--num_classes",   type=int,   default=37)
    p.add_argument("--dropout_p",     type=float, default=0.6)
    # Training schedule
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="Phase 1 peak LR (frozen backbone)")
    p.add_argument("--phase2_lr",     type=float, default=3e-4,
                   help="Phase 2 peak LR (unfrozen, independent cosine)")
    p.add_argument("--weight_decay",  type=float, default=5e-4)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--warmup_epochs", type=int,   default=2)
    p.add_argument("--freeze_blocks", type=int,   default=4)
    p.add_argument("--unfreeze_epoch",type=int,   default=10)
    # Localization loss weights
    p.add_argument("--mse_weight",    type=float, default=0.5,
                   help="Weight for MSE term in localization loss")
    p.add_argument("--iou_weight",    type=float, default=0.5,
                   help="Weight for IoU loss term in localization loss")
    # Resume control
    p.add_argument("--no_resume",     action="store_true")
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
    elif args.task == "localization":
        train_localization(args)
    elif args.task == "segmentation":
        train_segmentation(args)