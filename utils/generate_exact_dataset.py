import os

import cv2
import pandas as pd

from config import Config
from predict import FoveaPredictor


class DatasetGenerator:
    CROP_SIZE = 256
    OUTPUT_FOLDER = ""

    def __init__(self, config: Config):
        self._predictor = FoveaPredictor(config)
        self._annotations: pd.DataFrame = pd.read_excel(os.path.join("path_to_annotatios"))

    def execute(self):

        for row in self._annotations.iterrows():
            image = cv2.imread(row[0])
            prediction, fx, fy = self._predictor.predict(image)
            cv2.imwrite(os.path.join(self.OUTPUT_FOLDER, row[0]),)
