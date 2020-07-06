import logging
logging.basicConfig(format='%(levelname)s: [%(asctime)s] [%(name)s:%(lineno)d-%(funcName)20s()] %(message)s',
                    level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S')
from time import sleep

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
        for idx, data in enumerate(self._data_loader[DataMode.train]):
            img, labels = [d.cuda() for d in data]
            outputs = self._model(img)
            loss = self._loss(labels, outputs)
            self._logger.info("Actual loss: {}.".format(loss.item()))
            loss.backward()
            self._optimizer.step()
            plt.subplot(1, 2, 1)
            plt.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), vmin=0, vmax=1)
            plt.subplot(1, 2, 2)
            plt.imshow(labels[0, 0, :, :].detach().cpu().numpy())
            plt.draw()
            plt.pause(0.001)




if __name__ == "__main__":
    Trainer(Config()).train()