import math
import argparse
import math
import os
import random

import numpy as np
import torch
from torch import optim, nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    StructuralSimilarityIndexMeasure as SSIM,
)
from tqdm import tqdm

from data import DebandDataset
from loss import Loss
from model import SGN_with_Guide, weights_init, GGDBN


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one_epoch(epoch: int, model, criterion, optimizer, train_loader, device):
    psnr_cal = PSNR(data_range=1).to(device=device)
    ssim_cal = SSIM(data_range=1).to(device=device)
    model.train()
    mean_loss, mean_psnr, mean_ssim = 0, 0, 0
    for gt, degrade in tqdm(train_loader, desc=f"Training epoch {epoch}"):
        gt, degrade = gt.to(device), degrade.to(device)
        optimizer.zero_grad()

        output, grad = model(degrade)
        loss = criterion(output, gt, grad)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mean_loss += loss.item()
            mean_psnr += psnr_cal(output, gt).item()
            mean_ssim += ssim_cal(output, gt).item()
    mean_loss /= len(train_loader)
    mean_psnr /= len(train_loader)
    mean_ssim /= len(train_loader)
    return {
        "loss": mean_loss,
        "psnr": mean_psnr,
        "ssim": mean_ssim,
    }


@torch.no_grad()
def validate(epoch: int, model, criterion, val_loader, device):
    model.eval()
    psnr_cal = PSNR(data_range=1).to(device=device)
    ssim_cal = SSIM(data_range=1).to(device=device)
    mean_loss, mean_psnr, mean_ssim = 0, 0, 0
    for gt, degrade in tqdm(val_loader, desc=f"Validating epoch {epoch}"):
        gt, degrade = gt.to(device), degrade.to(device)
        output, grad = model(degrade)
        loss = criterion(output, gt, grad)
        mean_loss += loss.item()
        mean_psnr += psnr_cal(output, gt).item()
        mean_ssim += ssim_cal(output, gt).item()

    mean_loss /= len(val_loader)
    mean_psnr /= len(val_loader)
    mean_ssim /= len(val_loader)
    return {
        "loss": mean_loss,
        "psnr": mean_psnr,
        "ssim": mean_ssim,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_dir", type=str,
                        help="path to dataset directory that contains train and val subdirectories")
    # training
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for training. Default is 4")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="batch size for validation. Default is 2")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading. Default is 4")
    parser.add_argument("--epochs", type=int, default=70, help="number of epochs for training. Default is 70")
    parser.add_argument("--device", type=str, default="cuda", help="device to use for training. Default is cuda")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for training. Default is 1e-4")
    parser.add_argument("--T_max", type=int, default=50,
                        help="number of epochs for cosine annealing scheduler. Default is 50")
    parser.add_argument("--eta_min", type=float, default=1e-6,
                        help="minimum learning rate for cosine annealing scheduler. Default is 1e-6")

    parser.add_argument("--seed", type=int, default=3407, help="random seed for reproducibility. Default is 3407")

    parser.add_argument("--ckpt_dir", default="./checkpoints", type=str,
                        help="path to directory to save checkpoints")
    parser.add_argument("--log_dir", default="./logs", type=str, help="path to directory to save tensorboard logs")
    parser.add_argument("--save_freq", type=int, default=10, help="frequency of saving checkpoints. Default is 10")

    parser.add_argument("--resume_from", type=str, help="path to checkpoint to resume training")

    args = parser.parse_args()

    print("Training arguments:")
    print(args)

    # set random seed
    seed_everything(args.seed)

    # create ckpt and log directories
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # create tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # define model and optimizer
    raw = SGN_with_Guide(3, 3, 32, act_op=nn.PReLU).to(args.device)
    model = GGDBN(raw, device=args.device).to(args.device)
    # model = SGN_with_Guide(3, 3, 32, device=args.device).to(device=args.device)
    weights_init(model, init_type='xavier')

    criterion = Loss(device=args.device).to(device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    # load dataset
    path = args.data_dir
    train_path = os.path.join(path, "train")
    valid_path = os.path.join(path, "valid")

    train_set = DebandDataset(train_path, data_aug=True, phase="train")
    valid_set = DebandDataset(valid_path, data_aug=False, phase="valid")

    print(f"Loading dataset from {path}:")
    print(f"Training set: {len(train_set)} images")
    print(f"Validation set: {len(valid_set)} images")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    start_epoch = 1
    best_psnr = 0

    # resume training from checkpoint
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint["best_psnr"]

    # train and validate
    for epoch in range(start_epoch, args.epochs + 1):
        writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epoch)
        train_metrics = train_one_epoch(epoch, model, criterion, optimizer, train_loader, args.device)
        print(
            f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.5f}, Train PSNR: {train_metrics['psnr']:.5f}, Train "
            f"SSIM: {train_metrics['ssim']:.5f}")
        valid_metrics = validate(epoch, model, criterion, valid_loader, args.device)
        print(
            f"Epoch {epoch}: Valid Loss: {valid_metrics['loss']:.5f}, Valid PSNR: {valid_metrics['psnr']:.5f}, Valid "
            f"SSIM: {valid_metrics['ssim']:.5f}")

        if math.isnan(train_metrics["loss"]) or math.isnan(valid_metrics["loss"]):
            print("Training terminated due to NaN loss")
            exit(1)

        # scheduler step, if it has running 50 epochs, keep the learning rate unchanged
        if epoch <= 50:
            scheduler.step()

        # update tensorboard
        for tag, value in train_metrics.items():
            writer.add_scalar(f"{tag}/train", value, epoch)
        for tag, value in valid_metrics.items():
            writer.add_scalar(f"{tag}/valid", value, epoch)
        writer.flush()

        # save checkpoint

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr,
        }

        if valid_metrics["psnr"] > best_psnr:
            best_psnr = valid_metrics["psnr"]
            ckpt["best_psnr"] = best_psnr
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pth"))

        if epoch % args.save_freq == 0:
            torch.save(ckpt, os.path.join(args.ckpt_dir, f"epoch_{epoch}.pth"))

    # save final model

    torch.save(ckpt, os.path.join(args.ckpt_dir, "final.pth"))

    writer.close()
