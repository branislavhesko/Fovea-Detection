import logging
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, DataMode
from data_loader import get_data_loader
from focal_loss import TotalLoss
from models.focal_net import FocalNet
from models.fovea_net import FoveaNet
from utils.precision_meter import PrecisionMeter


class Trainer:

    def __init__(self, config: Config):
        logging.basicConfig(format='%(levelname)s: [%(asctime)s] [%(name)s:%(lineno)d-%(funcName)20s()] %(message)s',
                            level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S')
        self._config = config
        self._model = FoveaNet(num_classes=config.num_classes).to(self._config.device)
        self._optimizer = torch.optim.AdamW(params=self._model.parameters(), lr=2e-4, weight_decay=1e-4)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._visualizer = SummaryWriter()
        self._data_loader = {DataMode.train: get_data_loader(config, mode=DataMode.train),
                             DataMode.eval: get_data_loader(config, mode=DataMode.eval)}
        self._loss = TotalLoss(config)
        self._precision_meter = {
            DataMode.train: PrecisionMeter(postprocess_fn=self._config.post_processing_fn, config=config),
            DataMode.eval: PrecisionMeter(postprocess_fn=self._config.post_processing_fn, config=config)
        }
        self._load_checkpoint(os.path.join(self._config.path_to_checkpoints, self._config.checkpoint_name))

    def train(self):
        self.validate(-1)
        for epoch in range(self._config.num_epochs):
            self._train_single_epoch(epoch)
            if epoch % self._config.validation_frequency == 0:
                self.validate(epoch)
                self._save_checkpoint(self._config.path_to_checkpoints, epoch)

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
                                        epoch * len(self._data_loader[DataMode.eval]) + idx)
            self._visualizer.add_scalar(f"eval/precision_y", self._precision_meter[DataMode.eval].last_precision[1],
                                        epoch * len(self._data_loader[DataMode.eval]) + idx)
            self._visualizer.add_scalar(f"eval/precision_x", self._precision_meter[DataMode.eval].last_precision[0],
                                        epoch * len(self._data_loader[DataMode.eval]) + idx)
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
            self._visualizer.add_scalar(f"train/precision_x", self._precision_meter[DataMode.train].last_precision[0],
                                        epoch * len(self._data_loader[DataMode.train]) + idx)
            self._visualizer.add_scalar(f"train/precision_y", self._precision_meter[DataMode.train].last_precision[1],
                                        epoch * len(self._data_loader[DataMode.train]) + idx)
            if idx % self._config.visualization_frequency[DataMode.train] == 0:
                for batch_idx in range(img.shape[0]):
                    fig = self._get_progress_plot(img, labels, outputs, batch_idx=batch_idx)
                    self._visualizer.add_figure(f"VISUALIZATION_TRAIN/{idx}/{batch_idx}", fig, epoch)
        self._logger.info(f"Training precision: {self._precision_meter[DataMode.train].precision}")

    @staticmethod
    def _get_progress_plot(img, labels, outputs, batch_idx):
        output_numpy = outputs[batch_idx, 0, :, :].detach().cpu().numpy()
        labels_numpy = labels[batch_idx, 0, :, :].detach().cpu().numpy()
        fig = plt.figure(1, dpi=200, figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.resize(img.detach().permute([0, 2, 3, 1]).cpu().numpy()[batch_idx, :, :, :], output_numpy.shape))
        plt.plot(np.amax(np.argmax(labels_numpy, axis=1)), np.amax(np.argmax(labels_numpy, axis=0)), "+r",
                 markersize=15)
        plt.subplot(1, 3, 2)
        plt.imshow(output_numpy, vmin=0, vmax=1)
        plt.plot(*np.unravel_index(np.argmax(output_numpy, axis=None), shape=output_numpy.shape)[::-1], "+r",
                 markersize=15)
        plt.subplot(1, 3, 3)
        plt.imshow(labels_numpy)
        return fig

    def _save_checkpoint(self, path, epoch, include_optimizer=True):
        self._check_and_mkdir(path)
        path = os.path.join(path, f"{self._precision_meter[DataMode.eval].precision:.2f}_{epoch}.ckpt")
        state_dict = {
            "model": self._model.state_dict(),
        }
        if include_optimizer:
            state_dict["optimizer"] = self._optimizer.state_dict()
        torch.save(state_dict, path)

    def _load_checkpoint(self, path):
        if self._config.checkpoint_name and os.path.exists(path):
            state_dict = torch.load(path)
            self._model.load_state_dict(state_dict["model"])
            if "optimizer" in state_dict:
                self._optimizer.load_state_dict(state_dict["optimizer"])
            print("Checkpoint loaded!")

    @staticmethod
    def _check_and_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    from config import Config
    Trainer(Config()).train()
