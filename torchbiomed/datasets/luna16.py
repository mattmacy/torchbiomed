import numpy as np

import torch
import torch.utils.data as data
import torchbiomed.utils as utils
from glob import glob
import os
import os.path
import SimpleITK as sitk

MIN_BOUND = -1000
MAX_BOUND = 400

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
    test_positive = set()
    for i in range(test_pos_count):
        test_positive |= set([positive_list[i]])
    train_positive = positive - test_positive
    if test_neg_count > 1:
        test_negative = set()
        for i in range(test_neg_count):
            test_negative |= set([negative_list[i]])
        train_negative = negative - test_negative
        train = list(train_positive | train_negative)
        test = list(test_positive | test_negative)
    else:
        train = list(train_positive)
        test = list(test_positive)
    np.random.shuffle(train)
    np.random.shuffle(test)
    return (train, test)

def load_image(root, series):
    if series in image_dict.keys():
        return image_dict[series]
    img_file = root + "/" + series + ".mhd"
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    z, y, x = np.shape(img)
    img = img.reshape((1, z, y, x))
    image_dict[series] = utils.truncate(img, MIN_BOUND, MAX_BOUND)
    return img

def load_label(root, series):
    if series in label_dict.keys():
        return label_dict[series]
    img_file = root + "/" + series + ".mhd"
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    img[img != 0] = 1
    z, y, x = np.shape(img)
    img = img.reshape((1, z, y, x))
    label_dict[series] = img
    return img

def make_dataset(dir, images, targets, seed, train, allow_empty):
    global image_dict, label_dict, test_split, train_split
    zero_tensor = None

    label_path = dir + "/" + targets
    label_files = glob(label_path + "/*.mhd")
    label_list = []
    for name in label_files:
        label_list.append(os.path.basename(name)[:-4])

    if len(test_split) == 0:
        sample_label = load_label(label_path, label_list[0])
        shape = np.shape(sample_label)
        zero_tensor = np.zeros(shape, dtype=np.uint8)
        image_list = []
        image_path = dir + "/" + images
        file_list=glob(image_path + "/*.mhd")
        for img_file in file_list:
            series = os.path.basename(img_file)[:-4]
            if not allow_empty and series not in label_list:
                continue
            image_list.append(series)
            if series not in label_list:
                label_dict[series] = zero_tensor
        np.random.seed(seed)
        full = set(image_list)
        positives = set(label_list) & full
        train_split, test_split = train_test_split(full, positives)
    if train:
        keys = train_split
    else:
        keys = test_split

    return keys

def normalize_lung_CT(**kwargs):
    mean_values = []
    var_values = []
    MIN_BOUND = -1000
    MAX_BOUND = 400
    Z_MAX, Y_MAX, X_MAX = kwargs['Z_MAX'], kwargs['Y_MAX'], kwargs['X_MAX']
    vox_spacing = kwargs['vox_spacing']
    utils.init_dims3D(Z_MAX, Y_MAX, X_MAX, vox_spacing)
    luna_subset_path = kwargs['src']
    luna_save_path = kwargs['dst']
    file_list=glob(luna_subset_path+"*.mhd")
    img_spacing = (vox_spacing, vox_spacing, vox_spacing)

    for img_file in file_list:
        itk_img = sitk.ReadImage(img_file)
        (x_space, y_space, z_space) = itk_img.GetSpacing()
        spacing_old = (z_space, y_space, x_space)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        img, mu, var = utils.resample_volume(img_array, spacing_old, img_spacing, bounds=(MIN_BOUND, MAX_BOUND))
        utils.save_updated_image(img, itk_img, luna_save_path+os.path.basename(img_file), img_spacing)
        mean_values.append(mu)
        var_values.append(var)
    dataset_mean = np.mean(mean_values)
    dataset_stddev = np.sqrt(np.mean(var_values))
    return (dataset_mean, dataset_stddev)


def normalize_lung_mask(**kwargs):
    mean_values = []
    var_values = []
    MIN_BOUND = -1000
    MAX_BOUND = 400
    Z_MAX, Y_MAX, X_MAX = kwargs['Z_MAX'], kwargs['Y_MAX'], kwargs['X_MAX']
    vox_spacing = kwargs['vox_spacing']
    utils.init_dims3D(Z_MAX, Y_MAX, X_MAX, vox_spacing)
    luna_seg_lungs_path = kwargs['src']
    luna_seg_lungs_save_path = kwargs['dst']
    file_list=glob(luna_seg_lungs_path+"*.mhd")
    img_spacing = (vox_spacing, vox_spacing, vox_spacing)
    for img_file in file_list:
        itk_img = sitk.ReadImage(img_file)
        (x_space, y_space, z_space) = itk_img.GetSpacing()
        spacing_old = (z_space, y_space, x_space)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        img, _, _ = utils.resample_volume(img_array, spacing_old, img_spacing)
        utils.save_updated_image(img, itk_img, luna_seg_lungs_save_path+os.path.basename(img_file), img_spacing)


class LUNA16(data.Dataset):
    def __init__(self, root='.', images=None, targets=None, transform=None,
                 target_transform=None, co_transform=None,
                 train=True, seed=1, allow_empty=True):
        if images is None or targets is None:
            raise(RuntimeError("both images and targets must be set"))

        imgs = make_dataset(root, images, targets, seed, train, allow_empty)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 targets: " + root +
                               "/" + targets + "\n"))

        self.root = root
        self.imgs = imgs
        self.targets = self.root + "/" + targets
        self.images = self.root + "/" + images
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

    def __getitem__(self, index):
        series = self.imgs[index]
        target = load_label(self.targets, series)
        target = torch.from_numpy(target.astype(np.int64))
        image = load_image(self.images, series)
        img = image.astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.co_transform is not None:
            img, target = self.co_transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
