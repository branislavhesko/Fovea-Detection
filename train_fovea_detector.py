import logging
logging.basicConfig(format='%(levelname)s: [%(asctime)s] [%(name)s:%(lineno)d-%(funcName)20s()] %(message)s',
                    level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S')
from time import sleep

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config, DataMode
from data_loader import FoveaLoader, get_data_loader
from focal_loss import FocalLoss, TotalLoss
from models.fovea_net import FoveaNet


class Trainer:

    def __init__(self, config: Config):
        self._config = config
        self._model = FoveaNet(num_classes=config.num_classes).cuda()
        self._optimizer = torch.optim.Adam(params=self._model.parameters(), lr=1e-4)
        self._logger = logging.getLogger(__name__)
        self._visualizer = SummaryWriter()
        self._data_loader = get_data_loader(config)
        self._loss = TotalLoss(config)

    def train(self):

        for epoch in range(self._config.num_epochs):
            self._train_single_epoch(epoch)
            if epoch % self._config.validation_frequency == 0:
                self.validate(epoch)

    def validate(self, epoch):
        pass

    def _train_single_epoch(self, epoch):
        ln = None
        for idx, data in enumerate(self._data_loader[DataMode.train]):
            self._optimizer.zero_grad()
            img, labels = [d.cuda() for d in data]
            outputs = self._model(img)
            loss = self._loss(labels, outputs)
            self._logger.info("Actual loss: {}.".format(loss.item()))
            loss.backward()
            self._optimizer.step()
            plt.clf()
            if ln is not None:
                for l in ln:
                    l.remove()
            out = outputs[0, 0, :, :].detach().cpu().numpy()
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.resize(img.detach().permute([0, 2, 3, 1]).cpu().numpy()[0, :, :, :], out.shape))
            plt.subplot(1, 3, 2)
            plt.imshow(out, vmin=0, vmax=1)
            ln = plt.plot(*np.unravel_index(np.argmax(out, axis=None), shape=out.shape)[::-1], "+r", markersize=15)
            self._logger.info(f"OUT_LOC: {np.unravel_index(np.argmax(out, axis=None), shape=out.shape)[::-1]}")
            plt.subplot(1, 3, 3)
            lab = labels[0, 0, :, :].detach().cpu().numpy()
            plt.imshow(lab)
            plt.plot(np.amax(np.argmax(lab, axis=1)), np.amax(np.argmax(lab, axis=0)), "+r", markersize=15)
            self._logger.info(f"GT_LOC: {(np.amax(np.argmax(lab, axis=1)), np.amax(np.argmax(lab, axis=0)))}")

            plt.draw()
            plt.pause(0.1)


if __name__ == "__main__":
    Trainer(Config()).train()