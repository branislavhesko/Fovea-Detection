import numpy as np
from skimage.transform import rotate
import torch


class Compose:

    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, image, points):
        for transform in self._transforms:
            image, mask = transform(image, points)


class Rotate:

    def __init__(self, std_angle):
        self._std_angle = std_angle

    def __call__(self, image, points):
        angle = torch.randn(1) * self._std_angle
        image = rotate(image, angle, resize=False)
        rotated_points = []
        matrix = torch.tensor([[torch.cos(angle), - torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]])
        center = torch.tensor([image.shape[0] // 2, image.shape[1] // 2])
        for point in points:
            centered_point = point - center
            rotated_point = matrix @ centered_point[::-1]
            rotated_points.append(rotated_point[::-1] + center)
        return image, rotated_points


class VerticalFlip:

    def __init__(self, probability):
        self._probability = probability

    def __call__(self, image, points):
        if np.random.rand(1) > self._probability:
            return image, points
        shape = image.shape[:2]
        return np.copy(image[::-1, :, :]), [(shape[1] - point[1], point[0]) for point in points]


class HorizontalFlip:

    def __init__(self, probability):
        self._probability = probability

    def __call__(self, image, points):
        if np.random.rand(1) > self._probability:
            return image, points
        shape = image.shape[:2]
        return np.copy(image[:, ::-1, :]), [(point[1], shape[0] - point[0]) for point in points]


class Scale:

    def __init__(self, interval):
        self._range = interval

    def __call__(self, image, points):
        return image, points
