import glob
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class FoveaLoader(Dataset):

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._fovea_gt = pd.DataFrame()
        self._images = []

    def _load_annotations(self, path):
        xlss = glob.glob(osp.join(path, "*.xls"))
        for xls in xlss:
            data = pd.read_excel(xls)
            self._fovea_gt["ImgName"] = data["ImgName"]
            self._fovea_gt["Fovea_X"] = data["Fovea_X"]
            self._fovea_gt["Fovea_Y"] = data["Fovea_Y"]

    def _load_images(self, path):
        self._images = glob.glob(osp.join(path, "images", "*.jpg"))

    def __len__(self):
        assert len(self._images) == self._fovea_gt["ImgName"].shape[0]
        return len(self._images)

    def __getitem__(self, item):
        img_name, fovea_x, fovea_y = self._fovea_gt.iloc[item, :]
        img = cv2.cvtColor(cv2.imread(list(filter(
            lambda x: img_name in x, self._images))[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.
        fovea_gt = self._prepare_mask(fovea_x, fovea_y, img.shape[:2])

    def _prepare_mask(self, fovea_x, fovea_y, shape):
        return np.zeros_like(shape)


def get_data_loader(config):
    dataset = FoveaLoader(config)
    # TODO: finish
    data_loader = DataLoader(dataset)
    return data_loader
