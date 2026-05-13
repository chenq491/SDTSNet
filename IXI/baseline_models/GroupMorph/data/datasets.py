import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def pkload(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


class IXIBrainDatasetS2Atlas(Dataset):
    """Altas to subject registration train datasets"""

    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_paths = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index % (len(self.paths))
        y_index = index // (len(self.paths))

        path_x = self.paths[x_index]
        path_y = self.paths[y_index]

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
        return len(self.paths) * len(self.atlas_paths)


class IXIBrainInferDatasetS2Atlas(Dataset):
    """Altas to subject registration infer datasets"""

    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_paths = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index % (len(self.paths))
        y_index = index // (len(self.paths))

        path_x = self.paths[x_index]
        path_y = self.atlas_paths[y_index]

        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

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
        return len(self.paths) * len(self.atlas_paths)
