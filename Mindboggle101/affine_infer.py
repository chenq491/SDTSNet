import argparse
import glob
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import numpy as np
import torch
from SDTSNet.data import datasets, trans
from SDTSNet.metrics.dice import dice_val_VOI
from SDTSNet.metrics.surface_distance.metrics import (
    compute_average_surface_distance,
    compute_robust_hausdorff,
    compute_surface_distances,
)
from SDTSNet.model import function
from torch.utils.data import DataLoader
from torchvision import transforms

# 设置参数
parser = argparse.ArgumentParser()

parser.add_argument(
    "--test_dir",
    type=str,
    default="/root/autodl-tmp/Mindboggle101/Test/",
    help="测试集数据目录",
)

parser.add_argument("--num_labels", type=int, default=63, help="计算DSC的标签数")
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

    """
    加载数据集
    """
    test_composed = transforms.Compose(
        [
            trans.Seg_norm(),
            trans.NumpyType((np.float32, np.int16)),
        ]
    )
    test_set = datasets.Mindboggle101BrainInferDatasetS2S(
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
    eval_dsc_def = function.AverageMeter()  # 配准后DSC
    eval_HD95 = function.AverageMeter()  # 95%豪斯多夫距离 HD95
    eval_ASSD = function.AverageMeter()  # 平均对称表明距离 ASSD
    with torch.no_grad():
        for data in test_loader:
            x, y, x_seg, y_seg = [t.cuda() for t in data]

            def_out = x_seg

            # DSC
            dsc_trans = dice_val_VOI(def_out.long(), y_seg.long())
            eval_dsc_def.update(dsc_trans.item(), x.size(0))

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
            print(f"Deformed Dice: {eval_dsc_def.val:.4f}")
            print(f"HD95: {eval_HD95.val:.4f}")
            print(f"ASSD: {eval_ASSD.val:.4f}")

        # print result
        print(
            "Summary --->  \n"
            "Deformed Dice:  {:10.6f} +- {:10.6f}  \n"
            "HD95:           {:10.6f} +- {:10.6f}  \n"
            "ASSD:           {:10.6f} +- {:10.6f}  \n".format(
                eval_dsc_def.avg,
                eval_dsc_def.std,
                eval_HD95.avg,
                eval_HD95.std,
                eval_ASSD.avg,
                eval_ASSD.std,
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
