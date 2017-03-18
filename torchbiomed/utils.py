import SimpleITK as sitk

import os.path
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import scipy.ndimage

Z_MAX = None
Y_MAX = None
X_MAX = None
vox_spacing = None
shape_max = None

def init_dims3D(z, y, x, spacing):
    global Z_MAX, Y_MAX, X_MAX, vox_spacing, shape_max
    vox_spacing = spacing
    Z_MAX, Y_MAX, X_MAX = z, y, x
    shape_max = (z, y, x)

def debug_img(img):
    plt.hist(img.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

def plot_3d(image, threshold=-300):
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    #p = image
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def npz_save(name, obj):
    keys = list(obj.keys())
    values = list(obj.values())
    np.savez(name+".npz", keys=keys, values=values)

def npz_save_compressed(name, obj):
    keys = list(obj.keys())
    values = list(obj.values())
    np.savez_compressed(name+"_compressed.npz", keys=keys, values=values)

def npz_load(filename):
    npzfile = np.load(filename+".npz")
    keys = npzfile["keys"]
    values = npzfile["values"]
    return dict(zip(keys, values))

def npz_load_compressed(filename):
    npzfile = np.load(filename+"_compressed.npz")
    keys = npzfile["keys"]
    values = npzfile["values"]
    return dict(zip(keys, values))

def copy_slice_centered(dst, src, dim):
    if dim <= Y_MAX:
        x_start = int((X_MAX - dim) / 2)
        y_start = int((Y_MAX - dim) / 2)
        for y in range(dim):
            for x in range(dim):
                dst[y_start+y][x_start+x] = src[y][x]
    elif dim <= X_MAX:
        x_start = int((X_MAX - dim) / 2)
        y_start = int((dim - Y_MAX) / 2)
        for y in range(Y_MAX):
            for x in range(dim):
                dst[y][x_start+x] = src[y_start+y][x]
    else:
        x_start = int((dim - X_MAX) / 2)
        y_start = int((dim - Y_MAX) / 2)
        for y in range(Y_MAX):
            for x in range(X_MAX):
                dst[y][x] = src[y_start+y][x_start+x]

def copy_normalized(src, dtype=np.int16):
    src_shape = np.shape(src)
    if src_shape == shape_max:
        return src
    
    (z_axis, y_axis, x_axis) = src_shape
    assert x_axis == y_axis
    new_img = np.full(shape_max, np.min(src), dtype=dtype)
    if z_axis < Z_MAX:
        start = int((Z_MAX - z_axis) / 2)
        for i in range(z_axis):
            copy_slice_centered(new_img[start + i], src[i], x_axis)
    else:
        start = int((z_axis - Z_MAX) / 2)
        for i in range(Z_MAX):
            copy_slice_centered(new_img[i], src[start+i], x_axis)            
    return new_img

def truncate(image, min_bound, max_bound):
    image[image < min_bound] = min_bound
    image[image > max_bound] = max_bound
    return image

def resample_volume(img, spacing_old, spacing_new, bounds=None):
    (z_axis, y_axis, x_axis) = np.shape(img)
    resize_factor = np.array(spacing_old) / spacing_new 
    new_shape = np.round(np.shape(img) * resize_factor)
    real_resize_factor = new_shape / np.shape(img)
    img_rescaled = scipy.ndimage.interpolation.zoom(img, real_resize_factor, mode='nearest').astype(np.int16)
    img_array_normalized = copy_normalized(img_rescaled)
    img_tmp = img_array_normalized.copy()
    # determine what the mean will be on the anticipated value range
    mu, var = 0., 0.
    if bounds is not None:
        min_bound, max_bound = bounds
        img_tmp = truncate(img_tmp, min_bound, max_bound)
        mu = np.mean(img_tmp)
        var = np.var(img_tmp)
    return (img_array_normalized, mu, var)

def save_updated_image(img_arr, itk_img_orig, path, spacing):
    itk_scaled_img = sitk.GetImageFromArray(img_arr, isVector=False)
    itk_scaled_img.SetSpacing(spacing)
    itk_scaled_img.SetOrigin(itk_img_orig.GetOrigin())
    sitk.WriteImage(itk_scaled_img, path)

