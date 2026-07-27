#!/usr/bin/env python3
"""Training script for HECKTOR segmentation models."""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for absolute imports (src.segmentation.*)
_root = Path(__file__).resolve()
while _root.parent != _root and not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from tqdm import tqdm
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from src.segmentation.config import UNet3DConfig, SegResNetConfig, UNETRConfig, SwinUNETRConfig, MedNeXtConfig
from src.segmentation.models import UNet3DModel, SegResNetModel, UNETRModel, SwinUNETRModel, MedNeXtModel
from src.segmentation.data import get_dataloaders
from src.segmentation.losses.dice_ce import get_loss_function
from src.segmentation.utils.logging import setup_logging


def evaluate_epoch(model, loader, criterion, dice_metric, device, config, use_sliding_window=False):
    """Run evaluation for one epoch, calculating loss and Dice metric."""
    model.eval()
    total_loss = 0.0
    
    roi_size = config.spatial_size
    sw_batch_size = 4
    
    post_label = AsDiscrete(to_onehot=config.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=config.num_classes)
    
    with torch.no_grad():
        for batch in loader:
            # FIX 1: batch["image"] -> concatenate ct and pet
            images = torch.cat([batch["ct"], batch["pet"]], dim=1).to(device)
            labels = batch["label"].to(device)
            
            if use_sliding_window:
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                    sigma_scale=0.125,
                    padding_mode="constant",
                    cval=0.0,
                    sw_device=device,
                    device=device,
                )
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            labels_list = decollate_batch(labels)
            labels_convert = [post_label(label_tensor) for label_tensor in labels_list]
            outputs_list = decollate_batch(outputs)
            outputs_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]
            
            dice_metric(y_pred=outputs_convert, y=labels_convert)

    avg_loss = total_loss / len(loader)
    avg_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    return avg_loss, avg_dice


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc='Training', leave=False):
        # FIX 2: Same fix as above
        images = torch.cat([batch["ct"], batch["pet"]], dim=1).to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)


def parse_args():
    parser = argparse.ArgumentParser(description="Train HECKTOR segmentation model")
    parser.add_argument("--config", type=str, default="unet3d", choices=["unet3d", "segresnet", "unetr", "swinunetr", "mednext"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup configuration
    if args.config == "unet3d":
        config = UNet3DConfig(fold=args.fold)
    elif args.config == "segresnet":
        config = SegResNetConfig(fold=args.fold)
    elif args.config == "unetr":
        config = UNETRConfig(fold=args.fold)
    elif args.config == "swinunetr":
        config = SwinUNETRConfig(fold=args.fold)
    elif args.config == "mednext":
        config = MedNeXtConfig(fold=args.fold)
    else:
        raise ValueError(f"Unknown config: {args.config}")
    
    if args.device:
        config.device = args.device
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    # CPU-specific settings
    if config.device == "cpu":
        config.num_workers = 0
        config.cache_rate = 0.0
    
    logger = setup_logging(config.log_dir)
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")
    
    # Setup device
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.cuda.set_device(device)
        logger.info(f"Using {device}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        if config.device == "cuda":
            logger.warning("CUDA not available, falling back to CPU.")
        logger.info(f"Using device: {device}")

    # Create model
    if args.config == "unet3d":
        model = UNet3DModel(config).to(device)
    elif args.config == "segresnet":
        model = SegResNetModel(config).to(device)
    elif args.config == "unetr":
        model = UNETRModel(config).to(device)
    elif args.config == "swinunetr":
        model = SwinUNETRModel(config).to(device)
    elif args.config == "mednext":  
        model = MedNeXtModel(config).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.config}")
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup data
    train_loader, val_loader = get_dataloaders(config, fold=args.fold)
    logger.info(f"Data loaded for fold {args.fold}: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Setup training components
    criterion = get_loss_function("dice_ce")
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # FIX 3: Removed verbose parameter (deprecated in newer PyTorch)
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=config.num_epochs,
        power=config.poly_lr_power,
    )
    writer = SummaryWriter(config.log_dir) if config.use_tensorboard else None
    
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_dice = 0.0
    if args.resume:
        checkpoint = model.load_checkpoint(args.resume, device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_dice = checkpoint.get("best_dice", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, previous best Dice: {best_val_dice:.4f}")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train - Loss: {train_loss:.4f}")
        
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            
        # Validation every 5 epochs or at the end
        val_loss, val_dice = 0.0, 0.0
        should_validate = (epoch + 1) % 5 == 0 or (epoch + 1) == config.num_epochs

        if should_validate:
            logger.info("Running validation with sliding window inference...")
            val_loss, val_dice = evaluate_epoch(
                model, val_loader, criterion, dice_metric, device, config, use_sliding_window=True
            )
            logger.info(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
            
            if writer:
                writer.add_scalar("Loss/validation", val_loss, epoch)
                writer.add_scalar("Dice/validation", val_dice, epoch)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        if writer:
            writer.add_scalar("Learning_Rate", current_lr, epoch)

        if should_validate and val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
            model.save_checkpoint(best_path, epoch, optimizer.state_dict(), best_dice=best_val_dice)
            logger.info(f"New best model saved with Dice: {best_val_dice:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.save_checkpoint_every == 0 or (epoch + 1) == config.num_epochs:
            last_model_path = os.path.join(config.checkpoint_dir, "last_model.pth")
            model.save_checkpoint(last_model_path, epoch, optimizer.state_dict(), best_dice=best_val_dice)
            logger.info(f"Saved last model checkpoint at epoch {epoch+1}")

    logger.info("Training completed!")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()