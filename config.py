from enum import auto, Enum


class DataMode(Enum):
    train = auto()
    eval = auto()


class Config:

    num_classes = 1
    num_epochs = 10
    learning_rate = 1e-4
    validation_frequency = 1
    alfa = 2.
    beta = 4.
    path = "/home/brani/STORAGE/DATA/refugee/"
    shape = (256, 256)
    output_stride = 2
