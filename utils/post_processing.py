import numpy as np


def max_postprocess(output):
    return np.unravel_index(np.argmax(output, axis=None), shape=output.shape)[::-1]


def center_of_gravity_postprocess(output):
    pass
