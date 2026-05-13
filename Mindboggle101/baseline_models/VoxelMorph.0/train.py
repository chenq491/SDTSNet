import argparse
import glob
import os
import random
import sys
import time

import losses
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
from data import datasets, trans
from model import VxmDense_1
from natsort import natsorted
from torch import optim
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
parser.add_argument("--model", type=str, default="VoxelMorph", help="使用的模型名称")
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
    cont_training = opt.cont_training

    model_name = opt.model
    weights = [1, 1]  # loss weights
    save_dir = "result/{}_ncc_{}_reg_{}_lr_{}/".format(model_name, *weights, lr)
    if not os.path.exists(save_dir + "experiments/"):
        os.makedirs(save_dir + "experiments/")
    if not os.path.exists(save_dir + "logs/"):
        os.makedirs(save_dir + "logs/")
    sys.stdout = Logger(save_dir + "logs/")
    """
    Initialize model
    """
    model = VxmDense_1(img_size)
    model.cuda()
    """
    Initialize spatial transformation function
    """
    reg_model = utils.register_model(img_size, "nearest")
    reg_model.cuda()
    """
    If continue from previous training
    """
    if cont_training:
        model_dir = "experiments/" + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])[
            "state_dict"
        ]
        model.load_state_dict(best_model)
        print(model_dir + natsorted(os.listdir(model_dir))[-1])
    else:
        updated_lr = lr
    """
    Initialize training
    """
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])

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
    optimizer = optim.Adam(
        model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True
    )
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty="l2")]

    # log train meta information
    train_log_path = save_dir + "logs/" + "trainlog.log"
    write_epoch_log(
        train_log_path,
        f"model:{model_name}\nloss weight: NCC_{weights[0]}\tGrad3D_{weights[1]}\nepoch: {epoch_start} -> {max_epoch - 1}\nlearning rate: {lr}\ndataset: LPBA\timage size: {img_size}\ttrain number:{len(train_loader)}\tval number:{len(val_loader)}",
    )

    best_dsc = 0
    writer = SummaryWriter(log_dir=save_dir + "logs/")
    for epoch in range(epoch_start, max_epoch):
        print("Training Starts")
        """
        Training
        """
        loss_all = utils.AverageMeter()
        idx = 0

        start_time = time.perf_counter()

        loop = tqdm(train_loader, total=len(train_loader))
        for data in loop:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]

            output = model(x, y)

            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Train Epoch [{epoch}/{max_epoch}]")
            loop.set_postfix(
                loss=loss.item(), ImgSim=loss_vals[0].item(), Reg=loss_vals[1].item()
            )
            print(
                "Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}".format(
                    idx,
                    len(train_loader),
                    loss.item(),
                    loss_vals[0].item(),
                    loss_vals[1].item(),
                )
            )

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
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            loop = tqdm(val_loader, total=len(val_loader))
            for data in loop:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                output = model(x, y)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))

                loop.set_description(f"Val Epoch [{epoch}/{max_epoch}]")
                loop.set_postfix(eval_dsc=eval_dsc.avg)
                print(epoch, ":", dsc)

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


def compute_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis("off")
        plt.imshow(img[i, :, :], cmap="gray")
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group["lr"] = round(
            INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8
        )


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
