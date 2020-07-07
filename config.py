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
    shape = (512, 512)
    output_stride = 2
    visualization_frequency = {
        DataMode.train: 50,
        DataMode.eval: 10
    }
    device = "cuda:0"
    kernel_size = 5
    shuffle = True
    batch_size = 2
