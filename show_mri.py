"""
查看MRI图像数据
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

data_path = (
    "D:\\work\\brain_mri_image_registration\\Mindboggle101\\Test\\subject_01.pkl"
)

axis = 2  # 0:冠状，1:矢状，2:横断
current_slice = None


def show_image(data_path):
    global current_slice
    with open(data_path, "rb") as f:
        data1 = pickle.load(f)

    img, seg = data1
    print(np.min(img), np.max(img), np.mean(img))

    target_label = None

    # 当前切片索引
    current_slice = img.shape[2] // 2

    fig, ax = plt.subplots(figsize=[8, 8])
    plt.title(f"按左右箭头切换切片（当前：{current_slice}）")

    def update_plot():
        ax.clear()

        # 获取切片
        if axis == 0:
            mri_s = img[current_slice, :, :]
            lab_s = seg[current_slice, :, :]
        elif axis == 1:
            mri_s = img[:, current_slice, :]
            lab_s = seg[:, current_slice, :]
        else:
            mri_s = img[:, :, current_slice]
            lab_s = seg[:, :, current_slice]

        # 设置掩码
        if target_label is not None:
            display_mask = (lab_s == target_label).astype(float) * target_label
        else:
            display_mask = lab_s

        plot_data = display_mask.copy()
        # plot_data[plot_data == 0] = np.nan

        # 绘图
        ax.imshow(mri_s, cmap="gray", origin="lower")
        ax.imshow(
            plot_data,
            cmap="jet",
            alpha=0.4,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(f"Axis:{axis}, Slice:{current_slice}/{img.shape[axis] - 1}\n")
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        global axis
        global current_slice

        max_slice = img.shape[axis] - 1

        if event.key == "right":
            current_slice = min(current_slice + 1, max_slice)
        elif event.key == "left":
            current_slice = max(current_slice - 1, 0)
        elif event.key == "up":
            # 可选：切换轴向
            axis = (axis + 1) % 3
            current_slice = img.shape[axis] // 2
        elif event.key == "down":
            axis = (axis - 1) % 3
            current_slice = img.shape[axis] // 2

        update_plot()

    # 绑定键盘事件
    fig.canvas.mpl_connect("key_press_event", on_key)

    # 初始绘制
    update_plot()
    plt.show()


if __name__ == "__main__":
    # for i in range(22, 64):
    #     print(i)
    #     data_path = f"D:\\work\\brain_mri_image_registration\\Mindboggle101\\Train\\subject_{i}.pkl"
    #     show_image(data_path)
    # data_path = "D:\\work\\brain_mri_image_registration\\Mindboggle101_volumes\\MMRR-21\\subject_02.pkl"
    data_path = "D:\\work\\brain_mri_image_registration\\IXI_data\\Test\\subject_1.pkl"
    show_image(data_path)
