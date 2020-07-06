import glob
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from config import DataMode


class FoveaLoader(Dataset):

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._fovea_gt = {}
        self._images = []
        self._load_annotations(config.path)
        self._load_images(config.path)

    def _load_annotations(self, path):
        xlss = glob.glob(osp.join(path, "fovea", "*.xlsx"))
        for xls in xlss:
            data = pd.read_excel(xls)
            if "ImgName" in self._fovea_gt:
                self._fovea_gt["ImgName"] = pd.concat([data["ImgName"], self._fovea_gt["ImgName"]], ignore_index=True)
                self._fovea_gt["Fovea_X"] = pd.concat([data["Fovea_X"], self._fovea_gt["Fovea_X"]], ignore_index=True)
                self._fovea_gt["Fovea_Y"] = pd.concat([data["Fovea_Y"], self._fovea_gt["Fovea_Y"]], ignore_index=True)
            else:
                self._fovea_gt["ImgName"] = data["ImgName"]
                self._fovea_gt["Fovea_X"] = data["Fovea_X"]
                self._fovea_gt["Fovea_Y"] = data["Fovea_Y"]

    def _load_images(self, path):
        self._images = glob.glob(osp.join(path, "images", "*.jpg"))

    def __len__(self):
        assert len(self._images) == self._fovea_gt["ImgName"].shape[0]
        return len(self._images)

    def __getitem__(self, item):
        img_name, fovea_x, fovea_y = self._fovea_gt["ImgName"].iloc[item], \
                                     int(self._fovea_gt["Fovea_X"].iloc[item]), \
                                     int(self._fovea_gt["Fovea_Y"].iloc[item])
        img = cv2.cvtColor(cv2.imread(list(filter(
            lambda x: img_name in x, self._images))[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.
        scale_x, scale_y = img.shape[1] / self._config.shape[1], img.shape[0] / self._config.shape[0]
        fovea_x = fovea_x / scale_x
        fovea_y = fovea_y / scale_y
        img = cv2.resize(img, (self._config.shape[0], self._config.shape[1]))
        fovea_gt = self.make_mask(np.array(img.shape[:2]) // self._config.output_stride, 11,
                                  (int(fovea_y // self._config.output_stride),
                                   int(fovea_x // self._config.output_stride)))
        return torch.from_numpy(img).permute([2, 0, 1]).float(), torch.from_numpy(fovea_gt).unsqueeze(0)

    @staticmethod
    def make_mask(mask_size, kernel_size, point):
        mask = np.zeros((mask_size[0] + kernel_size, mask_size[1] + kernel_size))
        gaussian_filter = cv2.getGaussianKernel(kernel_size, sigma=np.sqrt(kernel_size))
        gaussian_kernel = gaussian_filter @ gaussian_filter.T
        gaussian_kernel /= np.amax(gaussian_kernel)
        mask[point[0]: point[0] + kernel_size, point[1]: point[1] + kernel_size] = gaussian_kernel
        mask = mask[(kernel_size - 1) // 2: -1 - (kernel_size - 1) // 2,
                    (kernel_size - 1) // 2: -1 - (kernel_size - 1) // 2]
        return mask.astype(np.float32)


def get_data_loader(config):
    dataset = FoveaLoader(config)
    # TODO: finish
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    return {DataMode.train: data_loader, DataMode.eval: data_loader}


if __name__ == "__main__":
    from collections import namedtuple
    from matplotlib import pyplot as plt
    X = namedtuple("X", ["path", "shape"])
    dataset = FoveaLoader(X(path="./data/", shape=(512, 512)))
    for img, data in dataset:
        plt.imshow(data.numpy())
        plt.show()