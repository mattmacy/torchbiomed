import numpy as np

import torch
import torch.utils.data as data
import torchbiomed.utils as utils
from glob import glob
import os
import os.path
import SimpleITK as sitk
import pandas as pd

MIN_BOUND = -1000
MAX_BOUND = 400

image_dict = {}
label_dict = {}
mask_dict = {}
stats_dict = {}
test_split = []
train_split = []


def train_test_split(full, positive, test_fraction):
    negative = full - positive
    test_neg_count = int(np.ceil(len(negative)*test_fraction))
    test_pos_count = int(np.ceil(len(positive)*test_fraction))
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
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    z, y, x = np.shape(img)
    img = img.reshape((1, z, y, x))
    image_dict[series] = utils.truncate(img, MIN_BOUND, MAX_BOUND)
    stats_dict[series] = itk_img.GetOrigin(), itk_img.GetSpacing()
    return img


def load_label(root, series):
    if series in label_dict.keys():
        return label_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    if np.max(img) > 3400:
        img[img <= 3480] = 0
        img[img > 3480] = 1
    else:
        img[img != 0] = 1
    label_dict[series] = img.astype(np.uint8)
    return img


def load_mask(root, series):
    if series in mask_dict.keys():
        return mask_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    img[img != 0] = 1
    mask_dict[series] = img
    return img

def full_dataset(dir, images):
    image_path = os.path.join(dir, images)
    image_files = glob(os.path.join(image_path, "*.mhd"))
    image_list = []
    for name in image_files:
        image_list.append(os.path.basename(name)[:-4])
    return image_list

def make_dataset(dir, images, targets, seed, train, class_balance, partition, nonempty, test_fraction, mode):
    global image_dict, label_dict, test_split, train_split
    zero_tensor = None

    train = mode == "train"
    label_path = os.path.join(dir, targets)
    label_files = glob(os.path.join(label_path, "*.mhd"))
    label_list = []
    for name in label_files:
        label_list.append(os.path.basename(name)[:-4])

    sample_label = load_label(label_path, label_list[0])
    shape = np.shape(sample_label)
    if len(test_split) == 0:
        zero_tensor = np.zeros(shape, dtype=np.uint8)
        image_list = []
        image_path = os.path.join(dir, images)
        file_list=glob(image_path + "/*.mhd")
        for img_file in file_list:
            series = os.path.basename(img_file)[:-4]
            if series not in label_list:
                continue
            image_list.append(series)
            if series not in label_list:
                label_dict[series] = zero_tensor
        np.random.seed(seed)
        full = set(image_list)
        positives = set(label_list) & full
        train_split, test_split = train_test_split(full, positives, test_fraction)
    if train:
        keys = train_split
    else:
        keys = test_split
    part_list = []
    z, y, x = shape
    if partition is not None:
        z_p, y_p, x_p = partition
        z, y, x = shape
        z_incr, y_incr, x_incr = z // z_p, y // y_p, x // x_p
        assert z % z_p == 0
        assert y % y_p == 0
        assert x % x_p == 0
        for zi in range(z_p):
            zstart = zi*z_incr
            zend = zstart + z_incr
            for yi in range(y_p):
                ystart = yi*y_incr
                yend = ystart + y_incr
                for xi in range(x_p):
                    xstart = xi*x_incr
                    xend = xstart + x_incr
                    part_list.append(((zstart, zend), (ystart, yend), (xstart, xend)))
    else:
        part_list = [((0, z), (0, y), (0, x))]
    result = []
    target_means = []
    for key in keys:
        for part in part_list:
            target = load_label(label_path, key)
            if nonempty:
                if np.sum(utils.get_subvolume(target, part)) == 0:
                    continue
            target_means.append(np.mean(target))
            result.append((key, part))

    target_mean = np.mean(target_means)
    return (result, target_mean)


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
    file_list=glob(luna_subset_path + "/" + "*.mhd")
    img_spacing = (vox_spacing, vox_spacing, vox_spacing)

    for img_file in file_list:
        itk_img = sitk.ReadImage(img_file)
        (x_space, y_space, z_space) = itk_img.GetSpacing()
        spacing_old = (z_space, y_space, x_space)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        img, mu, var = utils.resample_volume(img_array, spacing_old, img_spacing, bounds=(MIN_BOUND, MAX_BOUND))
        utils.save_updated_image(img, luna_save_path+os.path.basename(img_file), itk_img.GetOrigin(), img_spacing)
        mean_values.append(mu)
        var_values.append(var)
    dataset_mean = np.mean(mean_values)
    dataset_stddev = np.sqrt(np.mean(var_values))
    return (dataset_mean, dataset_stddev)


def normalize_lung_mask(**kwargs):
    Z_MAX, Y_MAX, X_MAX = kwargs['Z_MAX'], kwargs['Y_MAX'], kwargs['X_MAX']
    vox_spacing = kwargs['vox_spacing']
    utils.init_dims3D(Z_MAX, Y_MAX, X_MAX, vox_spacing)
    luna_seg_lungs_path = kwargs['src']
    luna_seg_lungs_save_path = kwargs['dst']
    file_list=glob(os.path.join(luna_seg_lungs_path, "*.mhd"))
    img_spacing = (vox_spacing, vox_spacing, vox_spacing)
    for img_file in file_list:
        itk_img = sitk.ReadImage(img_file)
        (x_space, y_space, z_space) = itk_img.GetSpacing()
        spacing_old = (z_space, y_space, x_space)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        img, _, _ = utils.resample_volume(img_array, spacing_old, img_spacing)
        img[img < 1] = 0
        utils.save_updated_image(img, os.path.join(luna_seg_lungs_save_path, os.path.basename(img_file)),
                                 itk_img.GetOrigin(), img_spacing)


def build_nodule_offset_tables(file_list, save_path, **kwargs):
    offset_dict = {}
    origin_dict = {}
    X_MAX, Y_MAX, Z_MAX = kwargs['X_MAX'], kwargs['Y_MAX'], kwargs['Z_MAX']
    vox_spacing = kwargs['vox_spacing']
    for img_file in file_list:
        series = os.path.basename(img_file)[:-4]
        itk_img = sitk.ReadImage(img_file)
        (x_space, y_space, z_space) = itk_img.GetSpacing()
        img_spacing = (z_space, y_space, x_space)
        x_size, y_size, z_size = itk_img.GetSize()
        img_size = (z_size, y_size, x_size)
        resize_factor = np.array(img_spacing) / [vox_spacing, vox_spacing, vox_spacing]
        (new_z, new_y, new_x) = np.round(img_size * resize_factor)
        z_off = int((new_z - Z_MAX)/2)
        y_off = int((new_y - Y_MAX)/2)
        x_off = int((new_x - X_MAX)/2)
        offset_dict[series] = np.array((z_off, y_off, x_off))
        x_orig, y_orig, z_orig = itk_img.GetOrigin()
        origin_dict[series] = np.array((z_orig, y_orig, x_orig))
    utils.npz_save(os.path.join(save_path, "origin_table"), origin_dict)
    utils.npz_save(os.path.join(save_path, "offset_table"), offset_dict)


def normalize_nodule_mask(**kwargs):
    luna_path = kwargs['orig']
    pixel_count = 0
    mask_count = 0
    annotations = kwargs['annotations']
    Z_MAX, Y_MAX, X_MAX = kwargs['Z_MAX'], kwargs['Y_MAX'], kwargs['X_MAX']
    shape_max = (Z_MAX, Y_MAX, X_MAX)
    vox_spacing = kwargs['vox_spacing']
    tables_path = kwargs['tables']
    utils.init_dims3D(Z_MAX, Y_MAX, X_MAX, vox_spacing)
    x_list, y_list, z_list = [], [], []
    luna_normal_path = kwargs['src']
    luna_mask_path = kwargs['dst']

    def get_boundaries(origin, offsets, params):
        diam, center = params
        diam3 = np.array((diam, diam, diam))
        diamu = diam + vox_spacing
        diam3u = np.array((diamu, diamu, diamu))
        v_center = np.rint((center - origin)/vox_spacing)
        v_lower = np.rint((center - diam3 - origin)/vox_spacing)
        v_upper = np.rint((center + diam3u - origin)/vox_spacing)
        v_center -= offsets
        v_lower -= offsets
        v_upper -= offsets
        x_list.append(v_upper[2])
        y_list.append(v_upper[1])
        z_list.append(v_upper[0])
        x_list.append(v_lower[2])
        y_list.append(v_lower[1])
        z_list.append(v_lower[0])
        return (v_lower, v_center, v_upper)

    def l2_norm(pointA, pointB):
        point = pointA - pointB
        return np.sqrt(np.dot(point, point))

    def get_filename(case):
        for f in file_list:
            if case in f:
                return(f)

    def update_mask(mask, CT, bounds):
        v_lower, v_center, v_upper = bounds
        z_min, y_min, x_min = v_lower
        z_max, y_max, x_max = v_upper
        pixel_count = 0
        radius = np.rint((z_max - z_min + vox_spacing)/2)
        ct_thresh = -1000
        bit_count = 0
        #print(v_center)
        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if l2_norm(np.array((z, y, x)), v_center) > radius:
                        break
                    if CT[z][y][x] > ct_thresh:
                        mask[z][y][x] = 4
                        pixel_count += 1
                        bit_count += 1
        #assert bit_count != 0
        if bit_count == 0:
            print(v_center)

    origin_dict = utils.npz_load(os.path.join(tables_path, "origin_table"))
    offset_dict = utils.npz_load(os.path.join(tables_path, "offset_table"))

    file_list=glob(luna_normal_path+"*.mhd")
    df_node = pd.read_csv(annotations)
    df_node["file"] = df_node["seriesuid"].apply(get_filename)
    df_node = df_node.dropna()
    img_spacing = (vox_spacing, vox_spacing, vox_spacing)
    count = 0
    for img_file in file_list:
        mask_count = 0
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        if len(mini_df) == 0:
            continue
        mask = np.zeros(shape_max, dtype=np.int16)
        series = os.path.basename(img_file)[0:-4]
        origin = origin_dict[series]
        if origin[1] > 0 and origin[2] > 0:
            origin[1] = -origin[1]
            origin[2] = -origin[2]
        offsets = offset_dict[series]
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)
        for i in range(len(mini_df)):
            node_x = mini_df["coordX"].values[i]
            node_y = mini_df["coordY"].values[i]
            node_z = mini_df["coordZ"].values[i]
            diam = mini_df["diameter_mm"].values[i]
            params = (diam, np.array((node_z, node_y, node_x)))
            bounds = get_boundaries(origin, offsets, params)
            _, v_center, _ = bounds
            if np.min(v_center) < 0:
                print("origin: {} offsets: {}\n params: {} v_center: {}".format(
                    origin, offsets, params, v_center))
                continue
            bounds = np.array(bounds).astype(np.int16)
            update_mask(mask, img_array, bounds)
            mask_count += 1
        assert mask_count != 0
        itk_mask_img = sitk.GetImageFromArray(mask, isVector=False)
        itk_mask_img.SetSpacing(img_spacing)
        itk_mask_img.SetOrigin(origin)
        sitk.WriteImage(itk_mask_img, luna_mask_path+'/'+os.path.basename(img_file))


class LUNA16(data.Dataset):
    def __init__(self, root='.', images=None, targets=None, transform=None,
                 target_transform=None, co_transform=None, mode="train", seed=1,
                 class_balance=False, split=None, masks=None, nonempty=True,
                 test_fraction=0.15):
        if images is None:
            raise(RuntimeError("images must be set"))
        if targets is None and mode != "infer":
            raise(RuntimeError("both images and targets must be set if mode is not 'infer'"))
        if mode == "infer":
            imgs = full_dataset(root, images)
        else:
            imgs, target_mean = make_dataset(root, images, targets, seed, mode, class_balance, split, nonempty, test_fraction, mode)
            self.data_mean = target_mean
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images: " + os.path.join(root, images) + "\n"))

        self.mode = mode
        self.root = root
        self.imgs = imgs
        self.masks = None
        self.split = split
        if masks is not None:
            self.masks = os.path.join(self.root, masks)
        if targets is not None:
            self.targets = os.path.join(self.root, targets)
        self.images = os.path.join(self.root, images)
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

    def target_mean(self):
        return self.data_mean

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "test":
            return self.__getitem_dev(index)
        elif self.mode == "infer":
            return self.__getitem_prod(index)

    def __getitem_prod(self, index):
        series = self.imgs[index]
        image = load_image(self.images, series)
        origin, spacing = stats_dict[series]
        image = image.astype(np.float32)
        if self.split is not None:
            batches = utils.partition_image(image, self.split)
        else:
            batches = [image]
        if self.transform is not None:
            batches = map(self.transform, batches)
            batches = [*batches]
        batches = torch.cat(batches)
        return batches, series, origin, spacing

    def __getitem_dev(self, index):
        series, bounds = self.imgs[index]
        (zs, ze), (ys, ye), (xs, xe) = bounds
        target = load_label(self.targets, series)
        target = target[zs:ze, ys:ye, xs:xe]
        target = torch.from_numpy(target.astype(np.int64))
        image = load_image(self.images, series)
        image = image[0, zs:ze, ys:ye, xs:xe]
        # optionally mask out lung volume from image
        if self.masks is not None:
            mask = load_mask(self.masks, series)
            mask = mask[zs:ze, ys:ye, xs:xe]
            image -= MIN_BOUND
            image = np.multiply(mask, image)
            image += MIN_BOUND
        image = image.reshape((1, ze-zs, ye-ys, xe-xs))
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
