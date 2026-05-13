import pickle

import numpy as np

path = "D:\\work\\brain_mri_image_registration\\IXI_data\\atlas.pkl"


with open(path, "rb") as f:
    data = pickle.load(f)

x, y = data
print(type(x), type(y))
print(x.shape, y.shape)
print(np.mean(x), np.std(x), np.max(x), np.min(x))
print(np.mean(y), np.std(y), np.max(y), np.min(y))
print(np.unique(y), len(np.unique(y)))
