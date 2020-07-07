import numpy as np


class PrecisionMeter:

    def __init__(self, postprocess_fn):
        self._num_images = 0
        self._precision = []
        self._postprocess_fn = postprocess_fn.__func__

    def update(self, labels, outputs):
        for idx in range(labels.shape[0]):
            gt_point = np.amax(np.argmax(labels[idx, 0, :, :], axis=1)), \
                       np.amax(np.argmax(labels[idx, 0, :, :], axis=0))
            output_point = self._postprocess_fn(outputs[idx, 0, :, :])
            self._precision.append((abs(gt_point[0] - output_point[0]), abs(gt_point[1] - output_point[1])))

    @property
    def precision(self):
        return np.mean(self._precision)

    def reset(self):
        self._precision = []
        self._num_images = 0
