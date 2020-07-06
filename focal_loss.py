import logging

import torch


class FocalLoss(torch.nn.Module):

    def __init__(self, alfa=2., beta=4.):
        super().__init__()
        self._alfa = alfa
        self._beta = beta
        self._logger = logging.getLogger(self.__class__.__name__)

    def forward(self, labels, output):
        loss_point = torch.sum((1 - output[
            labels == 1.]) ** self._alfa * torch.log(output[labels == 1.]))
        loss_background = torch.mean((1 - labels) ** self._beta * output ** self._alfa * torch.log(1 - output))
        self._logger.info("Losses: point: {}, background: {}.".format(loss_point.item(), loss_background.item()))
        return -1 * (loss_point + loss_background)


class TotalLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self._focal_loss = FocalLoss(alfa=config.alfa, beta=config.beta)

    def forward(self, labels, output):
        return self._focal_loss(labels, output)


if __name__ == "__main__":
    x = FocalLoss()
    labels = torch.zeros(1, 32, 32)
    labels[0, 11, 10] = 1.
    output = torch.zeros(1, 32, 32)
    output[0, 11, 10] = 10.

    print(x(labels, output))