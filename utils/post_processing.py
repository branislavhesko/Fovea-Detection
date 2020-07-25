import cv2
import numpy as np


def max_postprocess(output, _):
    return np.unravel_index(np.argmax(output, axis=None), shape=output.shape)


def center_of_gravity_postprocess(output, config):
    output[:100, :] = 0
    output[250:] = 0
    output = cv2.blur(output, ksize=(7, 7))
    maximum = np.amax(output) * config.limit_size
    maximum_loc = np.unravel_index(np.argmax(output, axis=None), shape=output.shape)
    y, x = np.meshgrid(np.arange(output.shape[0]), np.arange(output.shape[1]))
    detected_object = (output > maximum) & (np.abs(x - maximum_loc[0]) < config.kernel_size) & \
                      (np.abs(y - maximum_loc[1]) < config.kernel_size)
    output[detected_object is False] = 0
    center_x = np.sum(x[detected_object] * output[detected_object]) / np.sum(output[detected_object])
    center_y = np.sum(y[detected_object] * output[detected_object]) / np.sum(output[detected_object])
    return center_x, center_y


if __name__ == "__main__":
    from config import Config
    print(center_of_gravity_postprocess(np.random.rand(128, 128), Config()))
