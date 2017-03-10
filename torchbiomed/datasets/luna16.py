import numpy as np

import torch
import torch.utils.data as data
import torchbiomed.utils as utils

import os
import os.path
from torchbiomed.utils import Z_MAX, Y_MAX, X_MAX

image_dict = {}
label_dict = {}
test_split = []
train_split = []

def train_test_split(full, positive):
    negative = full - positive
    test_neg_count = (len(negative) // 5) + 1
    test_pos_count = (len(positive) // 5) + 1
    negative_list = list(negative)
    positive_list = list(positive)
    np.random.shuffle(positive_list)
    np.random.shuffle(negative_list)
    test_negative = set()
    test_positive = set()
    for i in range(test_neg_count):
        test_negative |= set([negative_list[i]])
    for i in range(test_pos_count):
        test_positive |= set([positive_list[i]])
    train_negative = negative - test_negative
    train_positive = positive - test_positive
    train = list(train_positive | train_negative)
    test = list(test_positive | test_negative)
    np.random.shuffle(train)
    np.random.shuffle(test)
    return (train, test)

def make_dataset(dir, images, targets, seed, train):
    global image_dict, label_dict, test_split, train_split
    if len(image_dict) == 0:
        image_dict = utils.npz_load(dir + "/" + images)
    if len(label_dict) == 0:
        label_dict = utils.npz_load(dir + "/" + targets)

    for key in image_dict.keys():
        image_dict[key] = image_dict[key].reshape((1, Z_MAX, Y_MAX, X_MAX))
    for key in label_dict.keys():
        label = label_dict[key]
        label_dict[key] = label.astype(np.uint8).reshape((1, Z_MAX, Y_MAX, X_MAX))
    np.random.seed(seed)
    positives = set(label_dict.keys())

    assert len(positives) > 5

    full = set(image_dict.keys())
    if len(test_split) == 0:
        train_split, test_split = train_test_split(full, positives)
    if train:
        keys = train_split
    else:
        keys = test_split

    labels = []
    images = []
    zero_tensor = torch.ByteTensor(1, Z_MAX, Y_MAX, X_MAX).zero_()
    for key in keys:
        if key not in label_dict.keys():
            labels.append(zero_tensor)
        else:
            label = label_dict[key]
            labels.append(torch.from_numpy(label))

    for key in keys:
        image = image_dict[key]
        image = image.astype(np.float32)
        image = torch.from_numpy(utils.normalize(image))
        image_dict[key] = image
        images.append(image)

    results = list(zip(images, labels))
    return results


class LUNA16(data.Dataset):
    def __init__(self, root='.', images=None, targets=None, transform=None,
                 target_transform=None, co_transform=None,
                 train=True, seed=1):
        if images is None or targets is None:
            raise(RuntimeError("both images and targets must be set"))

        imgs = make_dataset(root, images, targets, seed, train)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 targets: " + root +
                               "/" + targets + "\n"))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

    def __getitem__(self, index):
        img, target = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.co_transform is not None:
            img, target = self.co_transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
