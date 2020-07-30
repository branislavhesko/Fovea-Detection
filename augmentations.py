import numpy as np
from skimage.transform import rotate
import torch


class Compose:

    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, image, points):
        for transform in self._transforms:
            image, points = transform(image, points)
        return image, points


class Rotate:

    def __init__(self, std_angle, stride=2):
        self._std_angle = std_angle
        self._stride = stride

    def __call__(self, image, points):
        angle = torch.randn(1) * self._std_angle
        image = rotate(image, angle, resize=False)
        rotated_points = []
        angle = angle / 180. * np.pi
        matrix = torch.tensor([[torch.cos(angle), - torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]])
        center = torch.tensor([image.shape[0] // 2 // self._stride, image.shape[1] // 2 // self._stride])
        for point in points:
            centered_point = torch.tensor(point) - center
            rotated_point = matrix @ torch.flip(centered_point, dims=[0]).float()
            rotated_points.append(torch.flip(rotated_point, dims=[0]) + center)
        return image, rotated_points


class VerticalFlip:

    def __init__(self, probability, stride):
        self._probability = probability
        self._stride = stride

    def __call__(self, image, points):
        if np.random.rand(1) > self._probability:
            return image, points
        shape = torch.tensor(image.shape[:2]) // self._stride
        return np.copy(image[::-1, :, :]), [(point[0], shape[1] - point[1]) for point in points]


class HorizontalFlip:

    def __init__(self, probability, stride):
        self._probability = probability
        self._stride = stride

    def __call__(self, image, points):
        if np.random.rand(1) > self._probability:
            return image, points
        shape = torch.tensor(image.shape[:2]) // self._stride
        return np.copy(image[:, ::-1, :]), [(shape[0] - point[0], point[1]) for point in points]


class Scale:

    def __init__(self, interval):
        self._range = interval

    def __call__(self, image, points):
        return image, points


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import cv2
    image = cv2.imread("/home/brani/STORAGE/DATA/refugee/test/V0100.jpg") / 255.
    image = cv2.resize(image, (512, 512))
    point = (76, 234)
    x = Rotate(20, 1)
    for i in range(100):
        k, p = x(image, [point,])
        plt.imshow(k)
        plt.plot(p[0][0], p[0][1], "+g", markersize=25)
        plt.show()