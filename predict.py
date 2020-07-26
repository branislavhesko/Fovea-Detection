import cv2
import numpy as np
import os
import torch

from config import Config
from models.fovea_net import FoveaNet


class FoveaPredictor:

    def __init__(self, config: Config):
        self._config = config
        self._model = FoveaNet(num_classes=self._config.num_classes)
        self._load()

    def _load(self):
        state_dict = torch.load(os.path.join(self._config.path_to_checkpoints, self._config.checkpoint_name))
        self._model.load_state_dict(state_dict["model"])
        self._model = self._model.to(self._config.device)
        self._model.eval()
        self._postprocess_fn = self._config.post_processing_fn.__func__

    @torch.no_grad()
    def predict(self, image):
        image, scale = self._preprocess_image(image)
        output = self._model(image)
        fovea_location = self._postprocess_fn(output[0, 0, :, :].cpu().numpy(), self._config)
        return output[0, 0, :, :].cpu().numpy(), fovea_location[0] * scale[0] * self._config.output_stride, \
               fovea_location[1] * scale[0] * self._config.output_stride

    def _preprocess_image(self, image):
        scale_x, scale_y = image.shape[1] / self._config.shape[1], image.shape[0] / self._config.shape[0]
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                            (self._config.shape[0], self._config.shape[1])) / 255.
        image = torch.from_numpy(image).permute(2, 0, 1).to(self._config.device)
        return image.float().unsqueeze(0), (scale_x, scale_y)


if __name__ == "__main__":
    import glob
    from matplotlib import pyplot as plt
    images = glob.glob(os.path.join("/home/brani/STORAGE/DATA/refugee/test/*.jpg"))
    predictor = FoveaPredictor(Config())
    import pandas as pd
    from tqdm import tqdm
    frame = pd.DataFrame(columns=["ImageName", "Fovea_X", "Fovea_Y"])
    for img_f in tqdm(images):
        img = cv2.imread(img_f)
        output, fy, fx = predictor.predict(img)
        basename = os.path.basename(img_f)
        frame = frame.append({"ImageName": basename, "Fovea_X": fx, "Fovea_Y": fy}, ignore_index=True)
        print(fx)
        print(fy)
        plt.subplot(1, 2, 1)
        plt.imshow(img[:, :, ::-1])
        plt.plot(fx, fy, "xb", markersize=11)
        plt.subplot(1, 2, 2)
        plt.imshow(output)
        plt.show()
    frame.to_csv("fovea_location_results.csv", sep=",", index=False)
