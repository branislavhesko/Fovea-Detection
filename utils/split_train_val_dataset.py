import cv2
import os
import glob
import shutil
from random import shuffle


def process_image(src, destination, shape):
    cv2.imwrite(destination, cv2.resize(cv2.imread(src, cv2.IMREAD_COLOR), shape))


def process_mask(src, destination, shape):
    cv2.imwrite(destination, cv2.resize(cv2.imread(src, cv2.IMREAD_GRAYSCALE), shape, interpolation=cv2.INTER_NEAREST))


SHAPE = (1024, 1024)
RATIO = 0.85
images = sorted(glob.glob("./data/images/*.jpg"))
masks = sorted(glob.glob("./data/GT_disc_cup/*.bmp"))

os.makedirs(os.path.join("./data/train"), exist_ok=True)
os.makedirs(os.path.join("./data/validate"), exist_ok=True)

for run_mode in ["train", "validate"]:
    for type_ in ["images", "masks"]:
        if os.path.exists(os.path.join("./data", run_mode, type_)):
            os.remove(os.path.join("./data", run_mode, type_))
        os.makedirs(os.path.join("./data", run_mode, type_), exist_ok=True)

assert len(images) == len(masks)

combined = list(zip(images, masks))
shuffle(combined)
to_train_number = int(len(combined) * RATIO)
for idx, (image, mask) in enumerate(combined):
    print("Processing image: {}, mask: {}".format(image, mask))
    if idx < to_train_number:
        process_image(image, os.path.join("./data/train/images", os.path.split(image)[1]), SHAPE)
        process_mask(mask, os.path.join("./data/train/masks", os.path.split(mask)[1]), SHAPE)
    else:
        process_image(image, os.path.join("./data/validate/images", os.path.split(image)[1]), SHAPE)
        process_mask(mask, os.path.join("./data/validate/masks", os.path.split(mask)[1]), SHAPE)
