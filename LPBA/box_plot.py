import glob
import json
import os
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import torch

# from SDTSNet.model.model import SDTSNet
# from baseline_models.VoxelMorph.model.model import VxmDense_2
# from baseline_models.TransMorph.model.model import CONFIGS, TransMorph
# from baseline_models.RCN.model.model import RCN_test
# from baseline_models.Dual_PRNet_PlusPlus.model.model import PRNetplusplus
# from baseline_models.GroupMorph.model.model import GruopMorph
from baseline_models.RDP.model.model import RDP
from natsort import natsorted
from SDTSNet.data import datasets, trans
from SDTSNet.model import function
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

base_dir = Path(__file__).resolve().parent
substruct_ids = {
    "Frontal": [21, 22, 23, 24, 25, 26, 91, 92],
    "Parietal": [31, 32, 33, 34, 29, 30],
    "Temporal": [65, 66, 67, 68, 81, 82, 89, 90],
    "Occipital": [43, 44, 45, 46, 47, 48, 49, 50, 61, 62, 63, 64],
    "Limbic": [41, 42, 83, 84, 85, 86, 87, 88],
    "Subcortical": [101, 102, 121, 122],
    "Cerebellum": [161, 162, 163, 164],
}
methods = [
    "SDTSNet",
    "VoxelMorph",
    "TransMorph",
    "RCN",
    "Dual-PRNet++",
    "GroupMorph",
    "RDP",
]


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


def dice_val_VOI(y_pred, y_true, VOI_lbls):
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.0 * intersection) / (union + 1e-5)
        DSCs[idx] = dsc
        idx += 1
    return np.mean(DSCs)


def get_data():
    if os.path.exists(base_dir / "dice_substruct.json"):
        with open(base_dir / "dice_substruct.json", "r") as f:
            data = json.load(f)
    else:
        data = {}
        for substruct_name, _ in substruct_ids.items():
            data[substruct_name] = {}
            for model_name in methods:
                data[substruct_name][model_name] = []
        with open(base_dir / "dice_substruct.json", "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return data


def compute_dice():
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
    """
    加载模型
    """
    img_size = (160, 192, 160)
    model_idx = -1
    # model_dir = "SDTSNet/result/SDTSNet_ncc_1_reg_1_lr_0.0001/experiments/"
    # model_dir = "baseline_models/VoxelMorph/result/VxmDense_2_ncc_1_reg_1_lr_0.0001/experiments/"
    # model_dir = "baseline_models/TransMorph/result/TransMorph_ncc_1_reg_1_lr_0.0001/experiments/"
    # model_dir = "baseline_models/RCN/result/RCN_lr_0.0001/experiments/"
    # model_dir = "baseline_models/Dual_PRNet_PlusPlus/result/Dual-PRNet++_ncc_1_reg_1_lr_0.0001/experiments/"
    # model_dir = "baseline_models/GroupMorph/result/GroupMorph_dice_1_smooth_0.5_lr_0.0001/experiments/"
    model_dir = "baseline_models/RDP/result/RDP_ncc_1_reg_1_lr_0.0001/experiments/"
    model_name = "RDP"

    # model = SDTSNet(img_size, channels=8)
    # model = VxmDense_2(img_size)
    # model = TransMorph(CONFIGS["TransMorph"])
    # model = RCN_test(img_size, flow_multiplier=2, n_cascade=10)
    # model = PRNetplusplus(img_size)
    # model = GruopMorph(1, 8, img_size, (4, 2, 2))
    model = RDP(img_size, channels=16)

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

    """
    加载数据集
    """
    test_dir = "/root/autodl-tmp/RDP_LPBA_data/Test/"
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
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

    dice = get_data()
    """
    计算指标
    """
    with torch.no_grad():
        loop = tqdm(test_loader, total=len(test_loader))
        for data in loop:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_def, flow = model(x, y)
            # flow,_,_ = model(x,y)

            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])  # warped segment

            for substruct_name, VOI_lbls in substruct_ids.items():
                dsc = dice_val_VOI(def_out.long(), y_seg.long(), VOI_lbls)
                dice[substruct_name][model_name].append(dsc)

    with open(base_dir / "dice_substruct.json", "w") as f:
        json.dump(dice, f, ensure_ascii=False, indent=4)


def box_plot():
    plt.style.use("default")

    methods = [
        "VoxelMorph",
        "TransMorph",
        "RCN",
        "Dual-PRNet++",
        "GroupMorph",
        "RDP",
        "SDTSNet",
    ]

    regions = list(substruct_ids.keys())
    data_structure = get_data()

    # 2. 定义颜色 (对应图例顺序)
    # 注意：为了匹配图片，颜色顺序必须与 methods 列表一致
    colors = [
        "#4C72B0",
        "#55A868",
        "#C44E52",
        "#8172B2",
        "#4C4C4C",
        "#CCB974",
        "#8C564B",
    ]
    color_map = dict(zip(methods, colors))

    # 3. 开始绘图
    fig, ax = plt.subplots(figsize=(14, 8))

    # 设置箱型图的样式参数
    box_props = dict(linewidth=1)
    whisker_props = dict(linewidth=1)
    capprops = dict(linewidth=1)
    median_props = dict(color="orange", linewidth=1.5)  # 中位线为橙色
    flier_props = dict(
        marker="o", markerfacecolor="black", markersize=3, alpha=0.5
    )  # 离群点

    n_methods = len(methods)
    n_regions = len(regions)
    group_width = 0.9  # 每个大组（脑区）的总宽度
    bar_width = group_width / n_methods  # 单个箱型图的宽度

    # 循环绘制每个脑区的数据
    for i, region in enumerate(regions):
        # 计算当前组箱型图的中心位置
        center = i * 1.2  # 组间距设为 1.5

        # 计算该组内每个方法的具体位置
        positions = [
            center - group_width / 2 + bar_width / 2 + j * bar_width
            for j in range(n_methods)
        ]

        # 收集该组的数据
        data_to_plot = [data_structure[region][m] for m in methods]

        # 绘制箱型图
        bp = ax.boxplot(
            data_to_plot,
            positions=positions,
            widths=bar_width * 0.8,  # 箱体宽度稍微窄一点，留点缝隙
            patch_artist=True,  # 允许填充颜色
            boxprops=box_props,
            whiskerprops=whisker_props,
            capprops=capprops,
            medianprops=median_props,
            flierprops=flier_props,
        )

        # 为每个箱体上色
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(color_map[method])

    # 设置 X 轴
    ax.set_xticks([j * 1.2 for j in range(n_regions)])
    ax.set_xticklabels(regions, fontsize=12)
    ax.set_xlim(-0.5, (n_regions - 1) * 1.2 + 0.5)
    ax.set_ylabel("DSC", fontsize=12)
    ax.set_ylim(0.1, 0.9)

    # 6. 创建图例 (手动创建代理艺术家)
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=color_map[m], label=m) for m in methods]
    # 分两列显示图例，放在左下角
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        ncol=2,
        bbox_to_anchor=(0.01, 0.01),
        fontsize=10,
    )

    # 添加网格线 (仅 Y 轴)
    ax.yaxis.grid(True, linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(base_dir / "box_plot.png", bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.show()


if __name__ == "__main__":
    # compute_dice()
    box_plot()
