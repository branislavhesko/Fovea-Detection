import glob
import os
import os.path as osp

import cv2
import pandas as pd
from tqdm import tqdm

from config import Config
from predict import FoveaPredictor


class DatasetGenerator:
    CROP_SIZE = 256
    OUTPUT_FOLDER = "/home/brani/STORAGE/DATA/refugee/fovea_centered_dataset/"

    def __init__(self, config: Config):
        self._predictor = FoveaPredictor(config)
        self._images = glob.glob(osp.join(config.path, "images", "*.jpg"))
        self._fovea_gt = {}
        self._load_annotations(config.path)

    def _load_annotations(self, path):
        xlss = glob.glob(osp.join(path, "fovea_" + "eval", "*.xlsx")) + \
               glob.glob(osp.join(path, "fovea_" + "train", "*.xlsx"))
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

    def execute(self):
        frame = pd.DataFrame(columns=["ImgName", "Fovea_X", "Fovea_Y"])
        for idx in tqdm(range(self._fovea_gt["ImgName"].shape[0])):
            img_name, fovea_x, fovea_y = self._fovea_gt["ImgName"].iloc[idx], \
                                         int(self._fovea_gt["Fovea_X"].iloc[idx]), \
                                         int(self._fovea_gt["Fovea_Y"].iloc[idx])
            image = cv2.imread(list(filter(
                lambda x: img_name in x, self._images))[0], cv2.IMREAD_COLOR)
            prediction, fy, fx = self._predictor.predict(image)
            if -128 < (fovea_x - fx) < self.CROP_SIZE * 2 \
                    and -128 < (fovea_y - fy) < self.CROP_SIZE * 2:
                cropped = image[int(fy) - self.CROP_SIZE: int(fy) + self.CROP_SIZE,
                          int(fx) - self.CROP_SIZE: int(fx) + self.CROP_SIZE, :]
                cv2.imwrite(os.path.join(self.OUTPUT_FOLDER, img_name), cropped)
                frame = frame.append({"ImgName": img_name, "Fovea_X":  self.CROP_SIZE + fovea_x - fx ,
                                      "Fovea_Y": self.CROP_SIZE + fovea_y - fy}, ignore_index=True)
            else:
                print("GT: {}:{}, PREDICT: {}:{}".format(fovea_x, fovea_y, fx, fy))
        frame.to_excel(os.path.join(self.OUTPUT_FOLDER, "locations.xlsx"))


if __name__ == "__main__":
    DatasetGenerator(Config()).execute()
