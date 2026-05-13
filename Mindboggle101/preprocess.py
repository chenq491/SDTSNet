import glob
import os
import pickle

import numpy as np
import SimpleITK as sitk
from natsort import natsorted


def nii2arr(nii_path):

    arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
    return arr


def center(arr):
    c = np.sort(np.nonzero(arr))[:, [0, -1]]
    return np.mean(c, axis=-1).astype("int16")


def cropByCenter(image, center, final_shape=(160, 192, 160)):
    c = center
    crop = np.array([s // 2 for s in final_shape])
    # 0 axis
    cropmin, cropmax = c[0] - crop[0], c[0] + crop[0]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[0]
    if cropmax > image.shape[0]:
        cropmax = image.shape[0]
        cropmin = image.shape[0] - final_shape[0]
    image = image[cropmin:cropmax, :, :]
    # 1 axis
    cropmin, cropmax = c[1] - crop[1], c[1] + crop[1]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[1]
    if cropmax > image.shape[1]:
        cropmax = image.shape[1]
        cropmin = image.shape[1] - final_shape[1]
    image = image[:, cropmin:cropmax, :]

    # 2 axis
    cropmin, cropmax = c[2] - crop[2], c[2] + crop[2]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[2]
    if cropmax > image.shape[2]:
        cropmax = image.shape[2]
        cropmin = image.shape[2] - final_shape[2]
    image = image[:, :, cropmin:cropmax]
    return image


def minmax(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def pksave(img, label, save_path):
    with open(save_path, "wb") as f:
        pickle.dump((img, label), f)


path_to_origin = (
    "D:\\work\\brain_mri_image_registration\\Mindboggle101_volumes\\MMRR-21_volumes"
)
img_niis = natsorted(glob.glob(path_to_origin + "\\*\\t1weighted_brain.MNI152.nii.gz"))
label_niis = natsorted(
    glob.glob(path_to_origin + "\\*\\labels.DKT31.manual.MNI152.nii.gz")
)
save_path = "D:\\work\\brain_mri_image_registration\\Mindboggle101_volumes\\MMRR-21\\"
if not os.path.exists(save_path):
    os.makedirs(save_path)

index = 0
for img_nii, label_nii in zip(img_niis, label_niis):
    # 读取图像并转为numpy数组
    img, label = nii2arr(img_nii), nii2arr(label_nii)
    print(img.shape, label.shape)

    # 中心裁剪
    c = center(img)  # 非零区域中心点
    img = cropByCenter(img, c)  # 中心裁剪至指定形状
    label = cropByCenter(label, c)

    # 归一化
    img = minmax(img).astype("float32")
    label = label.astype("uint16")
    print(
        img.shape,
        np.unique(img),
        label.dtype,
        label.shape,
        np.unique(label),
        len(np.unique(label)),
    )
    print(save_path + "subject_%02d.pkl" % (index + 1))
    pksave(img, label, save_path=save_path + "subject_%02d.pkl" % (index + 1))
    index += 1
