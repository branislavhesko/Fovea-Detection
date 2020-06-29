import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config
from models.fovea_net import FoveaNet


class Trainer:

    def __init__(self, config: Config):
        self._config = config
        self._model = FoveaNet(num_classes=config.num_classes)
        self._optimizer = torch.optim.Adam(params=self._model.parameters(), lr=1e-4)
        self._logger = logging.getLogger(__name__)
        self._visualizer = SummaryWriter()

    def train(self):

        for epoch in range(self._config.num_epochs):
            self._train_single_epoch(epoch)
            if epoch % self._config.validation_frequency == 0:
                self.validate(epoch)

    def validate(self, epoch):
        pass

    def _train_single_epoch(self, epoch):
        pass
