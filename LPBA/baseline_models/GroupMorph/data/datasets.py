import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def pkload(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


class LPBABrainDatasetS2S(Dataset):
    """Subject to subject registration train datasets"""

    def __init__(self, data_path, transforms, half_pair=False):
        self.paths = data_path  # data file path list
        self.transforms = transforms
        self.is_half = half_pair

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def half_pair(self, pair):
        return pair[0][::2, ::2, ::2], pair[1][::2, ::2, ::2]

    def __getitem__(self, index):
        # chose two different data from train datasets
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s

        path_x = self.paths[x_index]
        path_y = self.paths[y_index]

        if self.is_half:
            x, x_seg = self.half_pair(pkload(path_x))
            y, y_seg = self.half_pair(pkload(path_y))
        else:
            x, x_seg = pkload(path_x)
            y, y_seg = pkload(path_y)

        x, y = x[None, ...], y[None, ...]  # add batch size dimension
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        # Convert to memory continuous
        x = np.ascontiguousarray(x)  # [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class LPBABrainInferDatasetS2S(Dataset):
    """Subject to subject registration infer datasets"""

    def __init__(self, data_path, transforms, half_pair=False):
        self.paths = data_path
        self.transforms = transforms
        self.is_half = half_pair

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def half_pair(self, pair):
        return pair[0][::2, ::2, ::2], pair[1][::2, ::2, ::2]

    def __getitem__(self, index):
        # chose two different data from train datasets
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s

        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        # print(os.path.basename(path_x), os.path.basename(path_y))
        if self.is_half:
            x, x_seg = self.half_pair(pkload(path_x))
            y, y_seg = self.half_pair(pkload(path_y))
        else:
            x, x_seg = pkload(path_x)
            y, y_seg = pkload(path_y)

        # add batch size dimension
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        # convert to memory continuous
        x = np.ascontiguousarray(x)  # [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channels,Height,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(x_seg),
            torch.from_numpy(y_seg),
        )
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)
