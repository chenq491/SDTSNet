import argparse
import glob
import os
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import numpy as np
import torch
from data import datasets, trans
from metrics.dice import dice_val, dice_val_VOI
from metrics.jacb import jacobian_determinant_vxm
from metrics.surface_distance.metrics import (
    compute_average_surface_distance,
    compute_robust_hausdorff,
    compute_surface_distances,
)
from model import function
from model.model import PRNetplusplus
from natsort import natsorted
from thop import profile
from torch.utils.data import DataLoader
from torchvision import transforms

# 设置参数
parser = argparse.ArgumentParser()


parser.add_argument("--img_size", type=tuple, default=(160, 192, 160), help="数据尺寸")
parser.add_argument("--model", type=str, default="Dual-PRNet++", help="使用的模型名称")
parser.add_argument(
    "--test_dir",
    type=str,
    default="/root/autodl-tmp/RDP_LPBA_data/Test/",
    help="测试集数据目录",
)
parser.add_argument("--learning_rate", type=float, default=0.0001, help="学习率")
parser.add_argument("--model_index", type=int, default=-1, help="使用的模型索引")
parser.add_argument("--num_labels", type=int, default=55, help="计算DSC的标签数")
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
    # torch.backends.cudnn.deterministic = True


same_seeds(42)


def main():

    test_dir = opt.test_dir

    model_idx = opt.model_index
    lr = opt.learning_rate
    model_name = opt.model
    weights = [1, 1]  # loss weights
    model_dir = "result/{}_ncc_{}_reg_{}_lr_{}/experiments/".format(
        model_name, *weights, lr
    )

    img_size = opt.img_size

    """
    加载模型
    """
    model = PRNetplusplus(img_size)

    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])
    best_epoch = best_model["epoch"]
    best_model = best_model["state_dict"]

    print(
        "Best model: {}\tEpoch: {}".format(
            natsorted(os.listdir(model_dir))[model_idx], best_epoch
        )
    )

    model.load_state_dict(best_model)
    model.cuda()
    reg_model = function.register_model(img_size, "nearest")
    reg_model.cuda()

    reg_model = function.register_model(img_size, "nearest")
    reg_model.cuda()

    """
    数据集
    """
    test_composed = transforms.Compose(
        [
            trans.Seg_norm(),
            trans.NumpyType((np.float32, np.int16)),
        ]
    )
    test_set = datasets.LPBABrainInferDatasetS2S(
        glob.glob(test_dir + "*.pkl"), transforms=test_composed
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    """
    计算指标
    """
    eval_dsc_raw = function.AverageMeter()  # 原始DSC
    eval_dsc_def = function.AverageMeter()  # 配准后DSC
    eval_jacb = function.AverageMeter()  # 负雅可比行列式百分比
    eval_HD95 = function.AverageMeter()  # 95%豪斯多夫距离 HD95
    eval_ASSD = function.AverageMeter()  # 平均对称表明距离 ASSD
    eval_timecost = function.AverageMeter()  # 配准时间
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            start_time = time.time()
            x_def, flow = model(x, y)
            eval_timecost.update(time.time() - start_time, x.size(0))

            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]

            jac_det = jacobian_determinant_vxm(
                flow.detach().cpu().numpy()[0, :, :, :, :]
            )
            eval_jacb.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            # DSC
            dsc_trans = dice_val_VOI(def_out.long(), y_seg.long())
            dsc_raw = dice_val_VOI(x_seg.long(), y_seg.long())
            dsc_trans_opt = dice_val(def_out.long(), y_seg.long(), opt.num_labels)
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))

            # HD95 和 ASSD
            y_seg = y_seg.detach().cpu().numpy()[0, 0, ...]
            def_out = def_out.detach().cpu().numpy()[0, 0, ...]
            hd95 = []
            assd = []
            for i in range(1, 55):
                if ((y_seg == i).sum() == 0) or ((def_out == i).sum() == 0):
                    continue
                surface_distance = compute_surface_distances(
                    (y_seg == i), (def_out == i), np.ones(3)
                )
                # HD95
                hd95.append(
                    compute_robust_hausdorff(
                        surface_distance,
                        95.0,
                    )
                )
                # ASSD
                gt2pred, pred2gt = compute_average_surface_distance(surface_distance)
                assd.append((gt2pred + pred2gt) / 2)
            eval_HD95.update(np.mean(np.array(hd95)), x.size(0))
            eval_ASSD.update(np.mean(np.array(assd)), x.size(0))

            # log
            print("-----------------------------------------------")
            print(f"Raw Dice: {eval_dsc_raw.val:.4f}")
            print(f"Deformed Dice: {eval_dsc_def.val:.4f}")
            print(f"Deformed Dice opt: {dsc_trans_opt.item():.4f}")
            print(f"|js|<=0: {eval_jacb.val:.4f}")
            print(f"HD95: {eval_HD95.val:.4f}")
            print(f"ASSD: {eval_ASSD.val:.4f}")

        # Flpos and Paramters
        xx = torch.rand(1, 1, *img_size).float().cuda()
        flop, para = profile(
            model,
            inputs=(
                xx,
                xx,
            ),
        )

        # print result
        print(
            "Summary --->  \n"
            "Affine Dice:    {:10.6f} +- {:10.6f}  \n"
            "Deformed Dice:  {:10.6f} +- {:10.6f}  \n"
            "|js|<=0:        {:10.6f} +- {:10.6f}  \n"
            "HD95:           {:10.6f} +- {:10.6f}  \n"
            "ASSD:           {:10.6f} +- {:10.6f}  \n"
            "Infer Time:     {:10.6f} s            \n"
            "Flops:          {:10.6f} M            \n"
            "Params:         {:10.6f} Mb           \n".format(
                eval_dsc_raw.avg,
                eval_dsc_raw.std,
                eval_dsc_def.avg,
                eval_dsc_def.std,
                eval_jacb.avg,
                eval_jacb.std,
                eval_HD95.avg,
                eval_HD95.std,
                eval_ASSD.avg,
                eval_ASSD.std,
                eval_timecost.avg,
                flop / 1e6,
                para / 1e6,
            )
        )


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
