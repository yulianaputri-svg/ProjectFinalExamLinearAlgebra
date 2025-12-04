import numpy as np
from PIL import Image

# ------------ MATRIX BUILDERS ------------

def translation_matrix(tx, ty):
    return np.array([[1,0,tx],[0,1,ty],[0,0,1]], float)

def scaling_matrix(sx, sy, center=None):
    M = np.array([[sx,0,0],[0,sy,0],[0,0,1]], float)
    if center:
        cx, cy = center
        return translation_matrix(cx, cy) @ M @ translation_matrix(-cx, -cy)
    return M

def rotation_matrix(angle_deg, center=None):
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    M = np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
    if center:
        cx, cy = center
        return translation_matrix(cx, cy) @ M @ translation_matrix(-cx, -cy)
    return M

def shear_matrix(shx=0.0, shy=0.0):
    return np.array([[1,shx,0],[shy,1,0],[0,0,1]], float)

def reflection_matrix(axis="y"):
    if axis == "x":
        return np.array([[1,0,0],[0,-1,0],[0,0,1]], float)
    if axis == "y":
        return np.array([[-1,0,0],[0,1,0],[0,0,1]], float)
    return np.eye(3)

# ------------ IMAGE HELPERS ------------

def pil_to_numpy(img):
    return np.array(img.convert("RGBA"), float)

def numpy_to_pil(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))

# ------------ BILINEAR INTERPOLATION ------------

def bilinear_interpolate(c, x, y):
    h, w = c.shape
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0,0,w-1)
    x1 = np.clip(x1,0,w-1)
    y0 = np.clip(y0,0,h-1)
    y1 = np.clip(y1,0,h-1)

    Ia = c[y0, x0]
    Ib = c[y1, x0]
    Ic = c[y0, x1]
    Id = c[y1, x1]

    wa = (x1 - x)*(y1 - y)
    wb = (x1 - x)*(y - y0)
    wc = (x - x0)*(y1 - y)
    wd = (x - x0)*(y - y0)

    return Ia*wa + Ib*wb + Ic*wc + Id*wd

# ------------ AFFINE TRANSFORM ------------

def apply_affine_transform(img, M, out_shape=None):
    h, w = img.shape[:2]
    if out_shape is None:
        oh, ow = h, w
    else:
        oh, ow = out_shape

    Minv = np.linalg.inv(M)
    xx, yy = np.meshgrid(np.arange(ow), np.arange(oh))
    ones = np.ones_like(xx)
    coords = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3).T

    src = Minv @ coords
    xs, ys = src[0], src[1]

    out = np.zeros((oh * ow, img.shape[2]), float)
    for c in range(img.shape[2]):
        out[:, c] = bilinear_interpolate(img[:, :, c], xs, ys)

    out = out.reshape(oh, ow, img.shape[2])
    return out
