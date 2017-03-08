import numpy as np
import torch.utils.data as data

import os
import os.path

def make_dataset(dir, images, targets):
    image_dict = npz_load(dir + "/" + images)
    label_dict = npz_load(dir + "/" + targets)

    for key in image_dict.keys():
        image = normalize(image_dict[key])
        image_dict[key] = image
    
    images = []
    for key in label_dict.keys():
        label = label_dict[key]
        image = image_dict[key]
        images.append((key, image, label))
    return images

class LUNA16(data.Dataset):
    def __init__(self, root, images, targets, transform=None, target_transform=None,
                 co_transform=None):
        imgs = make_dataset(root, images, targets)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 targets: " + root + "/" + targets + "\n"))

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
