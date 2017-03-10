import numpy as np
import torch.utils.data as data
import torchbiomed.utils as utils

import os
import os.path


def make_dataset(dir, images, targets, seed, train):
    image_dict = utils.npz_load(dir + "/" + images)
    label_dict = utils.npz_load(dir + "/" + targets)

    negatives = []
    positives = np.array(list(label_dict.keys()))
    assert len(positives) > 5

    for key in image_dict.keys():
        image = utils.normalize(image_dict[key])
        image_dict[key] = image
        if key not in label_dict.keys():
            label_dict[key] = np.zeros(utils.shape_max, dtype=np.int8)
            negatives.append(key)
    np.random.seed(seed)
    negatives = np.array(negatives)
    np.random.shuffle(positives)
    if len(negatives) > 5:
        np.random.shuffle(negatives)
        neg_test_count = len(negatives) // 5

    neg_test_count = 0
    pos_test_count = len(positives) // 5

    train_series = []
    test_series = []
    count = 0
    for series in positives:
        if count < pos_test_count:
            test_series.append(series)
        else:
            train_series.append(series)
        count += 1
    count = 0
    for series in negatives:
        if count < pos_test_count:
            test_series.append(series)
        else:
            train_series.append(series)
        count += 1

    if train:
        np.random.shuffle(train_series)
        image_series = train_series
    else:
        np.random.shuffle(test_series)
        image_series = test_series
    images = []
    for series in image_series:
        label = label_dict[series]
        image = image_dict[series]
        images.append((series, image, label))
    return images


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
        _, img, target = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.co_transform is not None:
            img, target = self.co_transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
