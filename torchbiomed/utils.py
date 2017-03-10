import os.path
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

Z_MAX = 192
Y_MAX = 160
X_MAX = 192
vox_spacing = 2
shape_max = (Z_MAX, Y_MAX, X_MAX)

image_dict = {}
mask_dict = {}
def img_store(image, name):
    fname = os.path.basename(name)[0:-4]    
    print("storing %s as %s"%(name, fname))
    image_dict[fname] = image.astype(np.int16)

def mask_store(image, name):
    fname = os.path.basename(name)[0:-4]    
    print("storing %s as %s"%(name, fname))
    mask_dict[fname] = image.astype(np.int8)
    

def debug_img(img):
    plt.hist(img.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

MIN_BOUND = -1000
MAX_BOUND = 400
UPPER_BOUND = 1400

# re-scale image on load for training
def normalize(image):
    image = image.astype(np.float32) / UPPER_BOUND
    return image
    
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

def npz_save_uncompressed(name, obj):
    keys = list(obj.keys())
    values = list(obj.values())
    np.savez(name+"_uncompressed.npz", keys=keys, values=values) 

def npz_save(name, obj):
    keys = list(obj.keys())
    values = list(obj.values())
    np.savez_compressed(name+".npz", keys=keys, values=values) 

def mask_save(name):
    npz_save(name, mask_dict)
    
def img_save(name):
    npz_save(name, image_dict)
    
def npz_load(filename):
    npzfile = np.load(filename+".npz")
    keys = npzfile["keys"]
    values = npzfile["values"]
    return dict(zip(keys, values))

def npz_load_uncompressed(filename):
    npzfile = np.load(filename+"_uncompressed.npz")
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
    new_img = np.zeros(shape_max, dtype=dtype)
    if z_axis < Z_MAX:
        start = int((Z_MAX - z_axis) / 2)
        for i in range(z_axis):
            copy_slice_centered(new_img[start + i], src[i], x_axis)
    else:
        start = int((z_axis - Z_MAX) / 2)
        for i in range(Z_MAX):
            copy_slice_centered(new_img[i], src[start+i], x_axis)            
    return new_img

