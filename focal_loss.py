import torch


class FocalLoss(torch.nn.Module):

    def __init__(self, alfa=2., beta=4.):
        super().__init__()
        self._alfa = alfa
        self._beta = beta

    def forward(self, labels, output):
        loss_point = torch.sum((1 - output[
            labels == 1.]) ** self._alfa * torch.nn.functional.logsigmoid(output[labels == 1.]))
        loss_background = torch.mean((1 - labels[labels != 1]) ** self._beta * output[
            labels != 1.] ** self._alfa * torch.nn.functional.logsigmoid(1 - output[labels != 1.]))
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