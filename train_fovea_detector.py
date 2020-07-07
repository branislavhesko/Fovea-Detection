import logging
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, DataMode
from data_loader import get_data_loader
from focal_loss import TotalLoss
from models.fovea_net import FoveaNet
from utils.precision_meter import PrecisionMeter


class Trainer:

    def __init__(self, config: Config):
        logging.basicConfig(format='%(levelname)s: [%(asctime)s] [%(name)s:%(lineno)d-%(funcName)20s()] %(message)s',
                            level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S')
        self._config = config
        self._model = FoveaNet(num_classes=config.num_classes).to(self._config.device)
        self._optimizer = torch.optim.Adam(params=self._model.parameters(), lr=1e-4)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._visualizer = SummaryWriter()
        self._data_loader = get_data_loader(config)
        self._loss = TotalLoss(config)
        self._precision_meter = {
            DataMode.train: PrecisionMeter(postprocess_fn=self._config.post_processing_fn),
            DataMode.eval: PrecisionMeter(postprocess_fn=self._config.post_processing_fn)
        }

    def train(self):

        for epoch in range(self._config.num_epochs):
            self._train_single_epoch(epoch)
            if epoch % self._config.validation_frequency == 0:
                self.validate(epoch)

    @torch.no_grad()
    def validate(self, epoch):
        self._precision_meter[DataMode.eval].reset()
        t = tqdm(self._data_loader[DataMode.eval])

        for idx, data in enumerate(t):
            img, labels = [d.to(self._config.device) for d in data]
            outputs = self._model(img)
            loss = self._loss(labels, outputs)
            t.set_description(f"Actual validation loss: {loss.item():.4f}.")
            self._precision_meter[DataMode.eval].update(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            self._visualizer.add_scalar("eval/TotalLoss", loss.item(),
                                        epoch * len(self._data_loader[DataMode.train]) + idx)
            if idx % self._config.visualization_frequency[DataMode.eval] == 0:
                for batch_idx in range(img.shape[0]):
                    fig = self._get_progress_plot(img, labels, outputs, batch_idx=batch_idx)
                    self._visualizer.add_figure(f"VISUALIZATION_EVAL/{idx}/{batch_idx}", fig, epoch)
        self._logger.info(f"Validation precision: {self._precision_meter[DataMode.eval].precision}")

    def _train_single_epoch(self, epoch):
        self._precision_meter[DataMode.train].reset()
        t = tqdm(self._data_loader[DataMode.train])
        for idx, data in enumerate(t):
            self._optimizer.zero_grad()
            img, labels = [d.to(self._config.device) for d in data]
            outputs = self._model(img)
            loss = self._loss(labels, outputs)
            t.set_description(f"Actual training loss: {loss.item():.4f}.")
            loss.backward()
            self._visualizer.add_scalar("train/TotalLoss", loss.item(),
                                        epoch * len(self._data_loader[DataMode.train]) + idx)
            self._optimizer.step()
            self._precision_meter[DataMode.train].update(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            if idx % self._config.visualization_frequency[DataMode.train] == 0:
                for batch_idx in range(img.shape[0]):
                    fig = self._get_progress_plot(img, labels, outputs, batch_idx=batch_idx)
                    self._visualizer.add_figure(f"VISUALIZATION_TRAIN/{idx}/{batch_idx}", fig, epoch)
        self._logger.info(f"Training precision: {self._precision_meter[DataMode.train].precision}")

    @staticmethod
    def _get_progress_plot(img, labels, outputs, batch_idx):
        output_numpy = outputs[batch_idx, 0, :, :].detach().cpu().numpy()
        fig = plt.figure(1, dpi=200, figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.resize(img.detach().permute([0, 2, 3, 1]).cpu().numpy()[batch_idx, :, :, :], output_numpy.shape))
        plt.subplot(1, 3, 2)
        plt.imshow(output_numpy, vmin=0, vmax=1)
        plt.plot(*np.unravel_index(np.argmax(output_numpy, axis=None), shape=output_numpy.shape)[::-1], "+r",
                 markersize=15)
        plt.subplot(1, 3, 3)
        labels_numpy = labels[batch_idx, 0, :, :].detach().cpu().numpy()
        plt.imshow(labels_numpy)
        plt.plot(np.amax(np.argmax(labels_numpy, axis=1)), np.amax(np.argmax(labels_numpy, axis=0)), "+r",
                 markersize=15)
        return fig


if __name__ == "__main__":
    Trainer(Config()).train()
