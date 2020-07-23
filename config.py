from enum import auto, Enum

from utils.post_processing import max_postprocess, center_of_gravity_postprocess


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
    shuffle = {DataMode.train: True, DataMode.eval: False}
    batch_size = 2
    post_processing_fn = center_of_gravity_postprocess
    limit_size = 0.8
    path_to_checkpoints = "/home/brani/STORAGE/DATA/refugee/checkpoints/fovea"
    checkpoint_name = "2.18_7.ckpt"
