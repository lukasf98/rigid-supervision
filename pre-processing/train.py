import os
import sys
import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader
from monai.metrics import DiceMetric
from monai.losses import DiceLoss

import itk

import datasets
import losses
import utils
from models.TransMorph import CONFIGS as CONFIGS_TM, SpatialTransformer
import models.TransMorph as TransMorph
import models.TransMorphSegs as TransMorphSegs

from accelerate import Accelerator, DistributedDataParallelKwargs

# Accelerator configuration
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])


class Logger:
    """
    Simple logger that writes both to terminal and a logfile.
    """
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_lr(optimizer):
    """Return the current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def adjust_learning_rate(optimizer, epoch, max_epochs, init_lr, power=0.9):
    """Adjust the learning rate according to a polynomial decay schedule."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = round(init_lr * np.power(1 - epoch / max_epochs, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224), device="cuda"):
    """Create a grid image tensor."""
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    return torch.from_numpy(grid_img).to(device)


def comput_fig(img):
    """Compute a figure from a tensor image slice for visualization."""
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis("off")
        plt.imshow(img[i, :, :], cmap="gray")
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def save_checkpoint(state, save_dir="models/", filename="checkpoint.pth.tar", max_model_num=8):
    """Save checkpoint and remove old ones if exceeding max_model_num."""
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + "*"))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + "*"))


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(device="cpu"):
    # Experiment parameters and save directories
    batch_size = 1
    weights = [1, 1000, 1, 1]  # [ncc, rigid, reg, dsc]
    w_segs = False
    save_dir = "1_TransMorph_ncc_{}_new_rigid_{}_diffusion_{}_dice_{}_wsegs_{}/".format(
        weights[0], weights[1], weights[2], weights[3]], w_segs
    )
    print("Save directory: " + save_dir)
    
    os.makedirs("/home/jovyan/artifacts/experiments/" + save_dir, exist_ok=True)
    os.makedirs("/home/jovyan/artifacts/checkpoints/" + save_dir, exist_ok=True)
    os.makedirs("/home/jovyan/artifacts/logs/" + save_dir, exist_ok=True)
    sys.stdout = Logger("/home/jovyan/artifacts/logs/" + save_dir)

    lr = 0.0001
    epoch_start = 0
    max_epoch = 501
    cont_training = False
    print(f"weights: {weights}, w_segs: {w_segs}, lr: {lr}, max_epoch: {max_epoch}, cont_training: {cont_training}")

    # Initialize model configuration and network
    H, W, D = 224, 192, 224
    config = CONFIGS_TM["TransMorph-Large"]
    config.img_size = (H, W, D)
    if w_segs:
        config.in_chans = 60
        model = TransMorphSegs.TransMorph(config)
    else:
        model = TransMorph.TransMorph(config)
    model.to(device)

    spatial_trans = SpatialTransformer((H, W, D)).to(accelerator.device)
    reg_model = utils.register_model(config.img_size, "nearest").to(device)
    reg_model_bilin = utils.register_model(config.img_size, "bilinear").to(device)

    # Continue training configuration
    if cont_training:
        epoch_start = 330
        model_dir = "/home/jovyan/artifacts/checkpoints/" + save_dir + f"epoch_{epoch_start}"
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        print("LR updated: {}!".format(updated_lr))
    else:
        updated_lr = lr

    # Initialize data loaders
    print("Loading data")
    train_ds = datasets.NLSTDataset(
        "/home/jovyan/working_dir/data/NLST",
        "/home/jovyan/data/NLST_REG_3",
        "/home/jovyan/data/NLST_SEG/NLST/NLST_TS/lung_vessels",
        "/home/jovyan/data/NLST_REG_LV_2",
        stage="train",
        use_cache=True,
    )
    val_ds = datasets.NLSTDataset(
        "/home/jovyan/working_dir/data/NLST",
        "/home/jovyan/data/NLST_REG_3",
        "/home/jovyan/data/NLST_SEG/NLST/NLST_TS/lung_vessels",
        "/home/jovyan/data/NLST_REG_LV_2",
        stage="val",
        use_cache=True,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_ncc = losses.NCC_vxm()
    criterion_dsc = DiceLoss()
    criterion_rigid_ddf = torch.nn.MSELoss(reduction="sum")
    criterion_reg = losses.Grad3d(penalty="l2")
    best_dsc = 0

    writer = SummaryWriter(log_dir="logs/" + save_dir)

    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    if cont_training:
        accelerator.load_state(model_dir)
        print("Model loaded from: {}".format(model_dir))

    # Training loop
    for epoch in range(epoch_start, max_epoch):
        print("Training Starts")
        loss_all = utils.AverageMeter()
        idx = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for data in pbar:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            
            # Retrieve tensors from data and move to device
            x = data["moving_image"].to(device)
            y = data["fixed_image"].to(device)
            rigid_00000_to_0001_ddf = data["rigid_00000_to_0001_ddf"].to(device)
            rigid_00001_to_0000_ddf = data["rigid_00001_to_0000_ddf"].to(device)
            rigid_00001_to_0000_label = data["rigid_00001_to_0000_label"].to(device)
            rigid_00000_to_0001_label = data["rigid_00000_to_0001_label"].to(device)

            # If segmentation losses are used, load segmentation labels
            if w_segs or weights[3] != 0 or weights[4] != 0:
                x_seg_lv = data["moving_lv_label"].to(device)
                y_seg_lv = data["fixed_lv_label"].to(device)
                x_seg_lung = data["moving_label"].to(device)
                y_seg_lung = data["fixed_label"].to(device)

            # One-hot encode segmentation labels if required
            if w_segs or weights[3] != 0:
                # Process rigid segmentation
                x_seg_rigid = data["moving_rigid_label"].to(device)
                x_seg_rigid = torch.where(x_seg_rigid > 26, torch.tensor(0, device=x_seg_rigid.device), x_seg_rigid)
                x_seg_rigid_oh = torch.nn.functional.one_hot(x_seg_rigid.long(), num_classes=27)
                x_seg_rigid_oh = x_seg_rigid_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                # Process lung vessels segmentation
                x_seg_lv_oh = torch.nn.functional.one_hot(x_seg_lv.long(), num_classes=3)
                x_seg_lv_oh = x_seg_lv_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                # Process lung segmentation
                x_seg_lung_oh = torch.nn.functional.one_hot(x_seg_lung.long(), num_classes=2)
                x_seg_lung_oh = x_seg_lung_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                x_seg = torch.cat((x_seg_rigid_oh, x_seg_lv_oh, x_seg_lung_oh), dim=1)

                y_seg_rigid = data["fixed_rigid_label"].to(device)
                y_seg_rigid = torch.where(y_seg_rigid > 26, torch.tensor(0, device=y_seg_rigid.device), y_seg_rigid)
                y_seg_rigid_oh = torch.nn.functional.one_hot(y_seg_rigid.long(), num_classes=27)
                y_seg_rigid_oh = y_seg_rigid_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                y_seg_lv_oh = torch.nn.functional.one_hot(y_seg_lv.long(), num_classes=3)
                y_seg_lv_oh = y_seg_lv_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                y_seg_lung_oh = torch.nn.functional.one_hot(y_seg_lung.long(), num_classes=2)
                y_seg_lung_oh = y_seg_lung_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                y_seg = torch.cat((y_seg_rigid_oh, y_seg_lv_oh, y_seg_lung_oh), dim=1)

            # Forward pass: compute registration from moving to fixed image
            if w_segs:
                x_in = torch.cat((x, x_seg, y, y_seg), dim=1)
                output, flow, flow_grid_sample = model(x_in, x)
            else:
                x_in = torch.cat((x, y), dim=1)
                output, flow, flow_grid_sample = model(x_in)

            # Compute segmentation loss if needed
            loss_dsc = 0
            if weights[3] != 0:
                def_segs = []
                for i in range(x_seg.shape[1]):
                    def_seg = spatial_trans(x_seg[:, i : i + 1].float(), flow.float())
                    def_segs.append(def_seg[0])
                def_seg = torch.cat(def_segs, dim=1)
                loss_dsc = criterion_dsc(def_seg, y_seg) * weights[3]

            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_rigid = (
                criterion_rigid_ddf(flow_grid_sample * rigid_00001_to_0000_label, rigid_00001_to_0000_ddf)
                / torch.sum(rigid_00001_to_0000_label > 0)
                * weights[1]
            )
            loss_reg = criterion_reg(flow) * weights[2]

            loss = loss_ncc + loss_reg + loss_rigid + loss_dsc
            loss_all.update(loss.item(), y.numel())

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            # Second pass: reverse direction (fixed to moving)
            if w_segs:
                y_in = torch.cat((y, y_seg, x, x_seg), dim=1)
                output, flow, flow_grid_sample = model(y_in, y)
            else:
                y_in = torch.cat((y, x), dim=1)
                output, flow, flow_grid_sample = model(y_in)

            loss_dsc = 0
            if weights[3] != 0:
                def_segs = []
                for i in range(x_seg.shape[1]):
                    def_seg = spatial_trans(y_seg[:, i : i + 1].float(), flow.float())
                    def_segs.append(def_seg[0])
                def_seg = torch.cat(def_segs, dim=1)
                loss_dsc = criterion_dsc(def_seg, x_seg) * weights[3]

            loss_ncc = criterion_ncc(output, x) * weights[0]
            loss_rigid = (
                criterion_rigid_ddf(flow_grid_sample * rigid_00000_to_0001_label, rigid_00000_to_0001_ddf)
                / torch.sum(rigid_00000_to_0001_label > 0)
                * weights[1]
            )

            loss_reg = criterion_reg(flow) * weights[2]
            loss = loss_ncc + loss_reg + loss_rigid + loss_dsc
            loss_all.update(loss.item(), x.numel())

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            pbar.set_description(
                "Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, DSC: {:.6f}, Reg: {:.6f}, Rigid: {:.6f}, lr: {:.6f}".format(
                    idx,
                    len(train_loader),
                    loss.item(),
                    loss_ncc.item(),
                    loss_dsc.item() if weights[3] != 0 else 0,
                    loss_reg.item(),
                    loss_rigid.item(),
                    get_lr(optimizer),
                )
            )

        writer.add_scalar("Loss/train", loss_all.avg, epoch)
        print("Epoch {} loss {:.4f}".format(epoch, loss_all.avg))

        # Validation every 10 epochs
        if epoch % 10 == 0:
            dice_metric_before = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
            dice_metric_after = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
            eval_dsc = utils.AverageMeter()

            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc="Validation")
                for data in pbar_val:
                    model.eval()
                    x = data["moving_image"].to(device)
                    y = data["fixed_image"].to(device)

                    if w_segs:
                        x_seg_rigid = data["moving_rigid_label"].to(device)
                        x_seg_rigid = torch.where(x_seg_rigid > 26, torch.tensor(0, device=x_seg_rigid.device), x_seg_rigid)
                        x_seg_rigid_oh = torch.nn.functional.one_hot(x_seg_rigid.long(), num_classes=27)
                        x_seg_rigid_oh = x_seg_rigid_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                        x_seg_lv = data["moving_lv_label"].to(device)
                        x_seg_lv_oh = torch.nn.functional.one_hot(x_seg_lv.long(), num_classes=3)
                        x_seg_lv_oh = x_seg_lv_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                        x_seg_lung = data["moving_label"].to(device)
                        x_seg_lung_oh = torch.nn.functional.one_hot(x_seg_lung.long(), num_classes=2)
                        x_seg_lung_oh = x_seg_lung_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                        x_seg = torch.cat((x_seg_rigid_oh, x_seg_lv_oh, x_seg_lung_oh), dim=1)

                        y_seg_rigid = data["fixed_rigid_label"].to(device)
                        y_seg_rigid = torch.where(y_seg_rigid > 26, torch.tensor(0, device=y_seg_rigid.device), y_seg_rigid)
                        y_seg_rigid_oh = torch.nn.functional.one_hot(y_seg_rigid.long(), num_classes=27)
                        y_seg_rigid_oh = y_seg_rigid_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                        y_seg_lv = data["fixed_lv_label"].to(device)
                        y_seg_lv_oh = torch.nn.functional.one_hot(y_seg_lv.long(), num_classes=3)
                        y_seg_lv_oh = y_seg_lv_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                        y_seg_lung = data["fixed_label"].to(device)
                        y_seg_lung_oh = torch.nn.functional.one_hot(y_seg_lung.long(), num_classes=2)
                        y_seg_lung_oh = y_seg_lung_oh.squeeze(1).permute(0, 4, 1, 2, 3)[:, 1:]
                        y_seg = torch.cat((y_seg_rigid_oh, y_seg_lv_oh, y_seg_lung_oh), dim=1)
                        x_in = torch.cat((x, x_seg, y, y_seg), dim=1)
                        output = model(x_in, x)
                    else:
                        x_in = torch.cat((x, y), dim=1)
                        output = model(x_in)

                    grid_img = mk_grid_img(8, 1, config.img_size, device)
                    x_seg = data["moving_label"].to(device)
                    y_seg = data["fixed_label"].to(device)

                    def_out = reg_model([x_seg.float(), output[1].float()])
                    def_grid = reg_model_bilin([grid_img.float(), output[1].float()])

                    dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                    dsc_before = dice_metric_before(y_pred=x_seg, y=y_seg)
                    dsc_after = dice_metric_after(y_pred=def_out, y=y_seg)
                    eval_dsc.update(dsc.item(), x.size(0))
                    pbar_val.set_description(f"DSC: {eval_dsc.avg}")

                best_dsc = max(eval_dsc.avg, best_dsc)
                dice_before = dice_metric_before.aggregate().item()
                dice_metric_before.reset()
                dice_after = dice_metric_after.aggregate().item()
                dice_metric_after.reset()
                print(f"dice_before = {dice_before:.3f}, dice_after = {dice_after:.3f}")

                accelerator.save_state(output_dir="/home/jovyan/artifacts/checkpoints/" + save_dir + f"epoch_{epoch}")
                writer.add_scalar("DSC/validate", eval_dsc.avg, epoch)

                pred_fig = comput_fig(def_out)
                grid_fig = comput_fig(def_grid)
                x_fig = comput_fig(x_seg)
                tar_fig = comput_fig(y_seg)
                writer.add_figure("Grid", grid_fig, epoch)
                plt.close(grid_fig)
                writer.add_figure("input", x_fig, epoch)
                plt.close(x_fig)
                writer.add_figure("ground truth", tar_fig, epoch)
                plt.close(tar_fig)
                writer.add_figure("prediction", pred_fig, epoch)
                plt.close(pred_fig)

            loss_all.reset()

        writer.close()


if __name__ == "__main__":
    print("Number of GPUs:", torch.cuda.device_count())
    set_seed(42)
    main(device=accelerator.device)
