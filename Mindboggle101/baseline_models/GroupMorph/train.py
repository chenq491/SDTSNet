import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import torch
from data import datasets, trans
from metrics.dice import dice_val_VOI
from model.function import AverageMeter, SpatialTransformer
from model.losses import (
    compute_per_channel_dice,
    mask_to_one_hot,
    ncc_loss,
)
from model.model import GruopMorph
from natsort import natsorted
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# 设置参数
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
parser.add_argument("--epoch_start", type=int, default=0, help="开始轮数")
parser.add_argument("--max_epoch", type=int, default=30, help="最大轮数")
parser.add_argument(
    "--cont_training", type=bool, default=False, help="是否从上次继续训练"
)
parser.add_argument("--learning_rate", type=float, default=0.0001, help="学习率")
parser.add_argument("--img_size", type=tuple, default=(160, 192, 160), help="数据尺寸")
parser.add_argument("--model", type=str, default="GroupMorph", help="使用的模型名称")
parser.add_argument(
    "--train_dir",
    type=str,
    default="/root/autodl-tmp/Mindboggle101/Train/",
    help="训练集数据目录",
)
parser.add_argument(
    "--val_dir",
    type=str,
    default="/root/autodl-tmp/Mindboggle101/Val/",
    help="验证集数据目录",
)
parser.add_argument(
    "--smooth",
    type=float,
    dest="smooth",
    default=0.5,
    help="Gradient smooth loss: suggested range 0.1 to 10",
)
parser.add_argument(
    "--dice",
    type=float,
    dest="dice",
    default=1,
    help="Dice loss: suggested range 0.1 to 10",
)
parser.add_argument(
    "--classes", type=int, dest="classes", default=63, help="number classes"
)
opt = parser.parse_args()


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


same_seeds(42)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        # self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    # load hyperparameter
    batch_size = opt.batch_size
    train_dir = opt.train_dir
    val_dir = opt.val_dir
    lr = opt.learning_rate

    epoch_start = opt.epoch_start
    max_epoch = opt.max_epoch
    img_size = opt.img_size

    model_name = opt.model

    classes = opt.classes
    dice = opt.dice
    smooth = opt.smooth

    save_dir = "result/{}_dice_{}_smooth_{}_lr_{}/".format(model_name, dice, smooth, lr)
    if not os.path.exists(save_dir + "experiments/"):
        os.makedirs(save_dir + "experiments/")
    if not os.path.exists(save_dir + "logs/"):
        os.makedirs(save_dir + "logs/")
    sys.stdout = Logger(save_dir + "logs/")
    """
    Initialize model
    """
    groups = (4, 2, 2)
    model = GruopMorph(1, 8, img_size, groups)
    model.cuda()
    """
    Initialize training
    """
    train_composed = transforms.Compose(
        [trans.Seg_norm(), trans.NumpyType((np.float32, np.float32))]
    )

    val_composed = transforms.Compose(
        [trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))]
    )
    train_set = datasets.MindBoggle101BrainDatasetS2S(
        glob.glob(train_dir + "*.pkl"), transforms=train_composed
    )
    val_set = datasets.Mindboggle101BrainInferDatasetS2S(
        glob.glob(val_dir + "*.pkl"), transforms=val_composed
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    """
    optimizer and loss function    
    """
    loss_similarity = ncc_loss
    transfor = SpatialTransformer().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[0], gamma=0.1)

    # log train meta information
    train_log_path = save_dir + "logs/" + "trainlog.log"
    write_epoch_log(
        train_log_path,
        f"model:{model_name}\nloss weight: NCC_1\tGrad3D_{smooth}\tDice_{dice}\nepoch: {epoch_start} -> {max_epoch - 1}\nlearning rate: {lr}\ndataset: Mindboggle101\timage size: {img_size}\ttrain number:{len(train_loader)}\tval number:{len(val_loader)}",
    )

    best_dsc = 0
    writer = SummaryWriter(log_dir=save_dir + "logs/")
    step = 0
    for epoch in range(epoch_start, max_epoch):
        print("Training Starts")
        """
        Training
        """
        loss_all = AverageMeter()

        start_time = time.perf_counter()

        loop = tqdm(train_loader, total=len(train_loader))
        for data in loop:
            model.train()

            data = [t.cuda() for t in data]
            x, y, x_seg, y_seg = data

            flows, warps, smo = model(x, y)

            # dice loss
            Y_label_onehot = mask_to_one_hot(y_seg, n_classes=classes)
            X_label_onehot = mask_to_one_hot(x_seg, n_classes=classes)
            warps_label_onehot = transfor(X_label_onehot, flows)
            diceloss = compute_per_channel_dice(
                warps_label_onehot, Y_label_onehot, classes=classes
            )
            # dice loss

            sim = loss_similarity(warps, y)
            smo_loss = smo
            loss = sim + dice * diceloss + smooth * smo_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_all.update(loss.item(), y.numel())
            loop.set_description(f"Train Epoch [{epoch}/{max_epoch}]")
            loop.set_postfix(
                loss=loss.item(),
                ImgSim=sim.item(),
                Reg=smo_loss.item(),
                Dice=diceloss.item(),
            )
            print(
                "step {} -> loss:{:.4f} - sim_loss:{:.4f} - dice_loss:{:.4f} - smo_loss:{:.4f} - lr:{:.6f}".format(
                    step,
                    loss.item(),
                    sim.item(),
                    diceloss.item(),
                    smo_loss.item(),
                    optimizer.param_groups[0]["lr"],
                )
            )
            step += 1

        end_time = time.perf_counter()
        minutes, seconds = divmod(end_time - start_time, 60)
        train_time_cost = f"{int(minutes):02d}:{int(seconds):02d}"

        # write epoch train loss to tensorboard
        writer.add_scalar("Loss/train", loss_all.avg, epoch)
        print("{} Epoch {} loss {:.4f}".format(save_dir, epoch, loss_all.avg))

        """
        Validation
        """
        start_time = time.perf_counter()
        eval_dsc = AverageMeter()
        with torch.no_grad():
            loop = tqdm(val_loader, total=len(val_loader))
            for data in loop:
                model.eval()
                data = [t.cuda() for t in data]
                x, y, x_seg, y_seg = data

                flows, _, _ = model(x, y)
                def_out = transfor(x_seg.float().cuda(), flows.cuda(), mode="nearest")

                dsc = dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))

                loop.set_description(f"Val Epoch [{epoch}/{max_epoch}]")
                loop.set_postfix(eval_dsc=eval_dsc.avg)
                print(epoch, ":", eval_dsc)

        end_time = time.perf_counter()
        minutes, seconds = divmod(end_time - start_time, 60)
        val_time_cost = f"{int(minutes):02d}:{int(seconds):02d}"

        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_dsc": best_dsc,
                "optimizer": optimizer.state_dict(),
            },
            save_dir=save_dir + "experiments/",
            filename="dsc{:.4f}.pth.tar".format(eval_dsc.avg),
        )

        # Write validate dsc to tensorboard
        writer.add_scalar("DSC/validate", eval_dsc.avg, epoch)

        minutes, seconds = divmod(end_time - start_time, 60)
        write_epoch_log(
            train_log_path,
            f"Epoch {epoch} train loss: {loss_all.avg:.4f} | validate DSC: {eval_dsc.avg:.4f} | train time cost:{train_time_cost} | val time cost:{val_time_cost}",
        )

        # reset loss
        loss_all.reset()


def write_epoch_log(filepath, content):
    with open(filepath, "a") as f:
        f.write(content + "\n")


def save_checkpoint(
    state, save_dir="models", filename="checkpoint.pth.tar", max_model_num=8
):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + "*"))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + "*"))


if __name__ == "__main__":
    """
    GPU configuration
    """
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print("Number of GPU: " + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print("     GPU #" + str(GPU_idx) + ": " + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print("Currently using: " + torch.cuda.get_device_name(GPU_iden))
    print("If the GPU is available? " + str(GPU_avai))
    main()
