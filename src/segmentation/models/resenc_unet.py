"""MONAI SegResNetDS segmentation — train + test-time-augmented inference.

This is one arm of the segmentation ensemble (the HECKTOR 2022 winner used
SegResNet via Auto3DSeg). The other arm is nnU-Net (seg_nnunet.py); ensemble
their softmax in seg_ensemble.py.

Recipe (winning-style): 1 mm isotropic, CT HU-window + per-channel PET z-score,
192^3 foreground-biased patches, Dice+CE with deep supervision, AdamW 2e-4 +
cosine, ~300 epochs, sliding-window + 8-flip TTA.

Data flow:
  scripts/01_preprocess.py calls preprocess.preprocess_to_pt() which saves each
  case as a .pt file (2-channel image + label, fixed 200×200×310 crop).
  train_fold() loads those .pt files — only fast augmentation transforms run
  at training time, so data loading is ~0.2 s/sample instead of ~8 s/sample.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROI = (192, 192, 192)
NUM_CLASSES = 3


def build_model(device="cuda", init_filters=32, blocks_down=(1, 2, 2, 4), dsdepth=4):
    from monai.networks.nets import SegResNetDS
    m = SegResNetDS(spatial_dims=3, init_filters=init_filters, in_channels=2,
                    out_channels=NUM_CLASSES, blocks_down=blocks_down,
                    norm="instance", dsdepth=dsdepth)
    return m.to(device)


class _LoadPt:
    """Load a preprocessed .pt file saved by preprocess.preprocess_to_pt()."""
    def __call__(self, item: dict) -> dict:
        data = torch.load(item["pt_path"], weights_only=False)
        label = data["label"].float()                  # (1, Z, Y, X)
        # Precompute flat linear indices for foreground/background so
        # RandCropByPosNegLabeld always finds foreground (MONAI expects 1-D)
        flat = label[0].reshape(-1)
        fg = torch.where(flat > 0)[0]
        bg = torch.where(flat == 0)[0]
        return {
            "image": data["image"].float(),            # (2, Z, Y, X)
            "label": label,
            "label_fg_indices": fg,
            "label_bg_indices": bg,
        }


def _transforms(train: bool):
    from monai.transforms import (
        Compose, EnsureTyped,
        RandCropByPosNegLabeld, RandFlipd, RandAffined,
        RandScaleIntensityd, RandShiftIntensityd, DivisiblePadd,
    )
    base = [
        _LoadPt(),
        EnsureTyped(keys=["image", "label"]),
    ]
    if not train:
        return Compose(base)
    return Compose(base + [
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=ROI, pos=2, neg=1, num_samples=2,
            allow_smaller=True,
            fg_indices_key="label_fg_indices",
            bg_indices_key="label_bg_indices",
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"], prob=0.2, mode=("bilinear", "nearest"),
            rotate_range=(0.26, 0.26, 0.26), scale_range=(0.1, 0.1, 0.1),
            padding_mode="border",
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        # ROI z (110) from the H&N crop isn't divisible by the SegResNetDS
        # downsampling factor (8). Pad each patch up to a multiple of 16 so the
        # encoder/decoder spatial dims line up. (Inference uses sliding-window,
        # which pads windows internally, so this is train-only.)
        DivisiblePadd(keys=["image", "label"], k=16),
    ])


def _ds_loss(outputs, label, loss_fn):
    if not isinstance(outputs, (list, tuple)):
        return loss_fn(outputs, label)
    total = wsum = 0.0
    for i, out in enumerate(outputs):
        w = 1.0 / (2 ** i)
        lbl = F.interpolate(label.float(), size=out.shape[2:], mode="nearest")
        total = total + w * loss_fn(out, lbl)
        wsum += w
    return total / wsum


def train_fold(datalist_items, val_fold, out_ckpt, max_epochs=300,
               early_stop_patience=50, device="cuda",
               lr=2e-4, weight_decay=1e-5, init_filters=32):
    """Train SegResNetDS on pre-processed .pt files.

    datalist_items: list of {"pt_path": str, "fold": int}
    Produce by calling scripts/01_preprocess.py which runs preprocess_to_pt().
    early_stop_patience: stop if val Dice doesn't improve for this many epochs.
    lr, weight_decay, init_filters: hyperparameters for HPO.
    """
    from monai.data import Dataset, DataLoader, decollate_batch
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference
    from monai.transforms import AsDiscrete

    train_items = [d for d in datalist_items if d["fold"] != val_fold]
    val_items   = [d for d in datalist_items if d["fold"] == val_fold]
    print(f"  train={len(train_items)}  val={len(val_items)}", flush=True)
    if not train_items or not val_items:
        raise ValueError(
            f"Empty SegResNet split (train={len(train_items)}, val={len(val_items)}). "
            f"val_fold={val_fold} but datalist folds are "
            f"{sorted(set(d['fold'] for d in datalist_items))}. "
            f"Need cases in BOTH fold=={val_fold} (val) and fold!={val_fold} (train).")

    tr = Dataset(train_items, _transforms(True))
    va = Dataset(val_items,   _transforms(False))
    tl = DataLoader(tr, batch_size=1, shuffle=True, num_workers=2,
                    pin_memory=False, drop_last=True,
                    multiprocessing_context="spawn")
    vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=2,
                    pin_memory=False,
                    multiprocessing_context="spawn")

    model    = build_model(device, init_filters=init_filters)
    loss_fn  = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    scaler   = torch.amp.GradScaler("cuda")
    pp = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    pl = AsDiscrete(to_onehot=NUM_CLASSES)
    dice = DiceMetric(include_background=False, reduction="mean")

    best = -1.0
    no_improve = 0
    for ep in range(max_epochs):
        model.train()
        for b in tl:
            x = b["image"].to(device)
            y = b["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = _ds_loss(model(x), y, loss_fn)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sch.step()

        model.eval()
        with torch.no_grad():
            for b in vl:
                x = b["image"].to(device)
                y = b["label"].to(device)
                with torch.amp.autocast("cuda"):
                    logits = sliding_window_inference(
                        x, ROI, 2, model, overlap=0.5, mode="gaussian"
                    )
                dice(
                    y_pred=[pp(p) for p in decollate_batch(logits)],
                    y=[pl(p) for p in decollate_batch(y)],
                )
        md = dice.aggregate().item()
        dice.reset()

        if md > best:
            best = md
            no_improve = 0
            torch.save(model.state_dict(), out_ckpt)
        else:
            no_improve += 1
        print(f"[segresnet] fold{val_fold} epoch {ep+1}/{max_epochs} "
              f"dice {md:.4f} (best {best:.4f})  no_improve={no_improve}", flush=True)
        if no_improve >= early_stop_patience:
            print(f"[segresnet] Early stopping at epoch {ep+1} "
                  f"(no improvement for {early_stop_patience} epochs)", flush=True)
            break

    return best


_FLIPS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


@torch.no_grad()
def predict_prob(model, image, device="cuda", roi=ROI, tta=True):
    """image: tensor [1,2,H,W,D] (CT, PET z-scored).
    Returns softmax probabilities [3,H,W,D] (numpy)."""
    from monai.inferers import sliding_window_inference
    image = image.to(device)
    acc = None
    flips = _FLIPS if tta else [()]
    for dims in flips:
        x = torch.flip(image, dims=dims) if dims else image
        with torch.amp.autocast("cuda"):
            logits = sliding_window_inference(
                x, roi, 2, model, overlap=0.5, mode="gaussian"
            )
        p = torch.softmax(logits.float(), 1)
        if dims:
            p = torch.flip(p, dims=dims)
        acc = p if acc is None else acc + p
    return (acc / len(flips))[0].cpu().numpy()


def load(weights, device="cuda"):
    m = build_model(device)
    m.load_state_dict(torch.load(weights, map_location=device))
    m.eval()
    return m
