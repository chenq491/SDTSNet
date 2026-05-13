import os
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from baseline_models.Dual_PRNet_PlusPlus.model.model import PRNetplusplus
from baseline_models.GroupMorph.model.model import GruopMorph
from baseline_models.RCN.model.model import RCN_test
from baseline_models.RDP.model.model import RDP
from baseline_models.TransMorph.model.model import CONFIGS, TransMorph
from baseline_models.VoxelMorph.model.model import VxmDense_2
from natsort import natsorted
from SDTSNet.model.function import register_model
from SDTSNet.model.model import SDTSNet

base_dir = Path(__file__).resolve().parent
img_size = (160, 192, 160)
axis = 0
slice_idx = 80

"""
加载模型
"""
SDTSNet_model_path = "SDTSNet/result/SDTSNet_ncc_1_reg_1_lr_0.0001/experiments/"
VoxelMorph_model_path = (
    "baseline_models/VoxelMorph/result/VxmDense_2_ncc_1_reg_1_lr_0.0001/experiments/"
)
TransMorph_model_path = (
    "baseline_models/TransMorph/result/TransMorph_ncc_1_reg_1_lr_0.0001/experiments/"
)
RCN_model_path = "baseline_models/RCN/result/RCN_lr_0.0001/experiments/"
DualPRNetpp_model_path = "baseline_models/Dual_PRNet_PlusPlus/result/Dual-PRNet++_ncc_1_reg_1_lr_0.0001/experiments/"
GroupMorph_model_path = "baseline_models/GroupMorph/result/GroupMorph_dice_1_smooth_0.5_lr_0.0001/experiments/"
RDP_model_path = "baseline_models/RDP/result/RDP_ncc_1_reg_1_lr_0.0001/experiments/"

sdtsnet = SDTSNet(img_size, channels=8)
voxelmorph = VxmDense_2(img_size)
transmorph = TransMorph(CONFIGS["TransMorph"])
rcn = RCN_test(img_size, flow_multiplier=2, n_cascade=10)
groupmorph = GruopMorph(1, 8, img_size, (4, 2, 2))
dualprnetpp = PRNetplusplus(img_size)
rdp = RDP(img_size, channels=16)

model_idx = -1

sdtsnet.load_state_dict(
    torch.load(
        SDTSNet_model_path + natsorted(os.listdir(SDTSNet_model_path))[model_idx]
    )["state_dict"]
)
sdtsnet.cuda()
voxelmorph.load_state_dict(
    torch.load(
        VoxelMorph_model_path + natsorted(os.listdir(VoxelMorph_model_path))[model_idx]
    )["state_dict"]
)
voxelmorph.cuda()
transmorph.load_state_dict(
    torch.load(
        TransMorph_model_path + natsorted(os.listdir(TransMorph_model_path))[model_idx]
    )["state_dict"]
)
transmorph.cuda()
rcn.load_state_dict(
    torch.load(RCN_model_path + natsorted(os.listdir(RCN_model_path))[model_idx])[
        "state_dict"
    ]
)
rcn.cuda()
dualprnetpp.load_state_dict(
    torch.load(
        DualPRNetpp_model_path
        + natsorted(os.listdir(DualPRNetpp_model_path))[model_idx]
    )["state_dict"]
)
dualprnetpp.cuda()
groupmorph.load_state_dict(
    torch.load(
        GroupMorph_model_path + natsorted(os.listdir(GroupMorph_model_path))[model_idx]
    )["state_dict"]
)
groupmorph.cuda()
rdp.load_state_dict(
    torch.load(RDP_model_path + natsorted(os.listdir(RDP_model_path))[model_idx])[
        "state_dict"
    ]
)
rdp.cuda()

reg_model = register_model(img_size, "bilinear").cuda()
"""
加载示例数据并形变
"""


def save_img_slice(img, name):
    if axis == 0:
        img_slice = img[slice_idx, :, :]
    elif axis == 1:
        img_slice = img[:, slice_idx, :]
    elif axis == 2:
        img_slice = img[:, :, slice_idx]

    # 整图
    fig = plt.figure(frameon=False, figsize=(6, 6))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img_slice, cmap="gray", interpolation="bilinear")
    x, y, h, w = 10, 25, 40, 40
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=5,  # 边框线宽
        edgecolor="green",  # 边框颜色
        facecolor="none",  # 内部不填充颜色（画轮廓）
        linestyle="-",  # 线型：'--'为虚线，'-'为实线
    )
    ax.add_patch(rect)
    plt.savefig(
        base_dir / f"{name}_all.png", bbox_inches="tight", pad_inches=0, dpi=300
    )
    # 局部图
    fig = plt.figure(frameon=False, figsize=(6, 6))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()  # 彻底关闭坐标轴显示
    fig.add_axes(ax)
    ax.imshow(img_slice[y : y + h, x : x + w], cmap="gray", interpolation="bilinear")
    outer_border = patches.Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        linewidth=20,
        edgecolor="green",
        facecolor="none",
        clip_on=False,
    )
    ax.add_patch(outer_border)
    plt.savefig(
        base_dir / f"{name}_local.png", bbox_inches="tight", pad_inches=0, dpi=300
    )


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 160)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_flow_slice(flow, name):
    if axis == 0:
        flow_slice = flow.cpu().numpy()[0, :, slice_idx, :, :]
    elif axis == 1:
        flow_slice = flow.cpu().numpy()[0, :, :, slice_idx, :]
    elif axis == 2:
        flow_slice = flow.cpu().numpy()[0, :, :, :, slice_idx]

    # 伪彩色图
    fig = plt.figure(frameon=False, figsize=(6, 6))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    norm = np.linalg.norm(flow_slice, axis=0)
    normalized_flow = flow_slice / (norm + 1e-10)
    dx, dy, dz = normalized_flow
    r = (dx + 1) / 2
    g = (dy + 1) / 2
    b = (dz + 1) / 2
    rgb_flow = np.stack((r, g, b), axis=2)
    rgb_flow = np.clip(rgb_flow, 0, 1)
    ax.imshow(rgb_flow, interpolation="bilinear")
    plt.savefig(
        base_dir / f"{name}_flow_rgb.png", bbox_inches="tight", pad_inches=0, dpi=300
    )
    plt.close()

    # 网格图
    fig = plt.figure(frameon=False, figsize=(6, 6))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    grid = mk_grid_img(grid_step=8, line_thickness=1, grid_sz=(160, 192, 160))
    print(grid.shape, flow.shape)
    grid = reg_model([grid.float().cuda(), flow.cuda()])

    grid = grid.cpu().numpy()[0, 0, ...]
    if axis == 0:
        grid_slice = grid[slice_idx, :, :]
    elif axis == 1:
        grid_slice = grid[:, slice_idx, :]
    elif axis == 2:
        grid_slice = grid[:, :, slice_idx]

    ax.imshow(grid_slice, cmap="gray", interpolation="bilinear")
    plt.savefig(
        base_dir / f"{name}_flowgrid.png", bbox_inches="tight", pad_inches=0, dpi=300
    )
    plt.close()


fixed_path = "/root/autodl-tmp/Mindboggle101/Test/subject_02.pkl"
moving_path = "/root/autodl-tmp/Mindboggle101/Test/subject_03.pkl"
with open(fixed_path, "rb") as f:
    fixed_img, fixed_label = pickle.load(f)
with open(moving_path, "rb") as f:
    moving_img, moving_label = pickle.load(f)

save_img_slice(fixed_img, "fixed")
save_img_slice(moving_img, "moving")

with torch.no_grad():
    fixed_img, fixed_label, moving_img, moving_label = (
        torch.from_numpy(fixed_img).float().cuda(),
        torch.from_numpy(fixed_label).float().cuda(),
        torch.from_numpy(moving_img).float().cuda(),
        torch.from_numpy(moving_label).float().cuda(),
    )
    fixed_img, moving_img = fixed_img[None, None, ...], moving_img[None, None, ...]
    # SDTSNet
    sdtsnet.eval()
    wapred_img, flow = sdtsnet(moving_img, fixed_img)
    save_img_slice(wapred_img.cpu().numpy()[0, 0, ...], "SDTSNet_wraped")
    save_flow_slice(flow, "SDTSNet")

    # VoxelMorph
    voxelmorph.eval()
    wapred_img, flow = voxelmorph(moving_img, fixed_img)
    save_img_slice(wapred_img.cpu().numpy()[0, 0, ...], "VoxelMorph_wraped")
    save_flow_slice(flow, "VoxelMorph")
    # TransMorph
    transmorph.eval()
    wapred_img, flow = transmorph(moving_img, fixed_img)
    save_img_slice(wapred_img.cpu().numpy()[0, 0, ...], "TransMorph_wraped")
    save_flow_slice(flow, "TransMorph")
    # RCN
    rcn.eval()
    wapred_img, flow = rcn(moving_img, fixed_img)
    save_img_slice(wapred_img.cpu().numpy()[0, 0, ...], "RCN_wraped")
    save_flow_slice(flow, "RCN")
    # Dual-PRNet++
    dualprnetpp.eval()
    wapred_img, flow = dualprnetpp(moving_img, fixed_img)
    save_img_slice(wapred_img.cpu().numpy()[0, 0, ...], "Dual_PRNetpp_wraped")
    save_flow_slice(flow, "Dual_PRNetpp")
    # GroupMorph
    groupmorph.eval()
    flow, warped_img, _ = groupmorph(moving_img, fixed_img)
    save_img_slice(wapred_img.cpu().numpy()[0, 0, ...], "GroupMorph_wraped")
    save_flow_slice(flow, "GroupMorph")
    # RDP
    rdp.eval()
    wapred_img, flow = rdp(moving_img, fixed_img)
    save_img_slice(wapred_img.cpu().numpy()[0, 0, ...], "RDP_wraped")
    save_flow_slice(flow, "RDP")
