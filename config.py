from enum import auto, Enum

from augmentations import Compose, HorizontalFlip, Rotate, VerticalFlip
from utils.post_processing import max_postprocess, center_of_gravity_postprocess


class DataMode(Enum):
    train = auto()
    eval = auto()


class Config:

    num_classes = 1
    num_epochs = 30
    learning_rate = 1e-4
    validation_frequency = 1
    alfa = 2.
    beta = 4.
    path = "/home/brani/DATA/DATASETS/refugee/"
    shape = (512, 512)
    output_stride = 1
    visualization_frequency = {
        DataMode.train: 50,
        DataMode.eval: 10
    }
    device = "cuda:0"
    kernel_size = 19
    shuffle = {DataMode.train: True, DataMode.eval: False}
    batch_size = 2
    post_processing_fn = center_of_gravity_postprocess
    limit_size = 0.8
    path_to_checkpoints = "/home/brani/DATA/DATASETS/refugee/checkpoints/fovea"
    checkpoint_name = ""
    subfolder = {
        DataMode.eval: "fovea_eval",
        DataMode.train: "fovea_train"
    }
    def __init__(self):
        self.augmentations = {
            DataMode.eval: Compose([]),
            DataMode.train: Compose([Rotate(10, self.output_stride),
                                     HorizontalFlip(0.5, self.output_stride),
                                     VerticalFlip(0.5, self.output_stride)])
        }


class ConfigPrecisingNetwork(Config):
    shape = (256, 256)
    batch_size = 12
    path_to_checkpoints = "/home/brani/DATA/DATASETS/refugee/checkpoints/fovea_precising_small"
    path = "/home/brani/DATA/DATASETS/refugee/fovea_centered_dataset_small"
    subfolder = {
        DataMode.eval: "fovea_centered_dataset_small",
        DataMode.train: "fovea_centered_dataset_small"
    }
    checkpoint_name = "6.40_6.ckpt"
    output_stride = 2
    kernel_size = 7
    post_processing_fn = max_postprocess
