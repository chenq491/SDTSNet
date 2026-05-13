import argparse
import glob
import os
import random

import numpy as np
import torch
import utils
from data import datasets, trans
from model import VxmDense_1
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import AverageMeter

# 设置参数
parser = argparse.ArgumentParser()


parser.add_argument("--img_size", type=tuple, default=(160, 192, 160), help="数据尺寸")
parser.add_argument("--model", type=str, default="VoxelMorph", help="使用的模型名称")
parser.add_argument(
    "--test_dir",
    type=str,
    default="/root/autodl-tmp/Mindboggle101/Test/",
    help="测试集数据目录",
)
parser.add_argument(
    "--model_folder",
    type=str,
    default="VoxelMorph_ncc_1_reg_1_lr_0.0001",
    help="模型文件夹名称",
)
parser.add_argument("--model_index", type=int, default=-1, help="使用的模型索引")
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
    model_folder = opt.model_folder
    model_dir = "result/" + model_folder + "experiments/"

    img_size = opt.img_size
    model = VxmDense_1(img_size)

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
    reg_model = utils.register_model(img_size, "nearest")
    reg_model.cuda()
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
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_def, flow = model(x, y)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])  # warped segment
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]

            # jacobian determinant of a displacement field.
            jac_det = utils.jacobian_determinant_vxm(
                flow.detach().cpu().numpy()[0, :, :, :, :]
            )
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            # DSC
            dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())

            print(
                "Trans dsc: {:.4f}, Raw dsc: {:.4f}".format(
                    dsc_trans.item(), dsc_raw.item()
                )
            )

            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))

        print(
            "Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}".format(
                eval_dsc_def.avg, eval_dsc_def.std, eval_dsc_raw.avg, eval_dsc_raw.std
            )
        )
        print("deformed det: {}, std: {}".format(eval_det.avg, eval_det.std))


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
