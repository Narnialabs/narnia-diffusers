import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def show_imgs(imgs=[], titles=[], r=1, cmap='gray', axis=False, vrange=[0.,1.]):  
    c = len(imgs) // r
    plt.figure(figsize=(c*3, r*3))
    for i in range(len(imgs)):
        plt.subplot(r, c, i+1)
        if bool(vrange): plt.imshow(imgs[i], cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        else: plt.imshow(imgs[i], cmap=cmap)
        if len(titles) == len(imgs):
            plt.title(titles[i])
        if not axis:
            plt.axis('off')
    plt.show()

def path2arr(path, img_size=None):
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if img_size is not None:
      img = cv2.resize(img, dsize=(img_size,img_size))
  img_arr = np.array(img)
  return img_arr

def cal_img_diff(img1, img2):
  diff = np.sum(np.abs(img1 - img2)) / img1.size
  return diff


def remove_bkg(arr, crop_ratio=0.8):
  x = np.copy(arr)
  img_size=x.shape[1]
  crop_mask = get_circle(crop_ratio, img_size=img_size, isbool=True)
  x[crop_mask != True] = 0.
  return x

def zoom_and_resize(arr, resize=128, crop_ratio=0.8):
  x = np.copy(arr)
  img_size = x.shape[1]
  center = int(img_size//2)
  crop_size = int(img_size * crop_ratio)
  start = center - crop_size // 2
  crop_slice = np.s_[start:start+crop_size, start:start+crop_size]
  cropped = x[crop_slice]
  resized = cv2.resize(cropped, dsize=(resize, resize), interpolation=cv2.INTER_AREA)
  resized = np.where(resized > 0., 1.0, 0.0)
  return resized
  
get_angle = lambda x : 360 // x

def rotate(arr, angle, thres=None):
  (h, w) = arr.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  img_rotated = cv2.warpAffine(arr, M, (w, h), borderValue=(0.))
  img_rotated = cv2.GaussianBlur(img_rotated, (5, 5), 0)
  if thres is not None:
    img_rotated = np.where(img_rotated > thres, 1.0, 0.0)
  return img_rotated


def get_rnd_beta(mn, mx, a=2., b=1.2, size=1000):
    z = (mx-mn)/mx
    xs = (np.random.beta(a, b, size=size))*z+mn
    return xs

def get_slice_label(arr):
  diff_mn = np.inf
  slices = range(2,16)
  for s in slices:
    angle = 360  / s
    rot = rotate(arr, angle)      
    diff_tmp = cal_img_diff(arr, rot)
    if diff_mn > diff_tmp: 
        diff_mn = diff_tmp
        label = s
  return label

def get_norm_img(img, slices, thres=0.65):
    norm_img = np.zeros(shape=img.shape)
    for i in range(1, 1+slices):
        angle = 360 * i / slices
        rot_img = rotate(img, angle)
        norm_img+=rot_img
    norm_img /=slices
    norm_img = np.where(norm_img > thres, 1.0, 0.0)
    return norm_img

def get_volume_img(volume, radius_ratio=1., img_size=128):
    image = np.full((img_size, img_size), 0.)
    center = img_size / 2.
    y, x = np.indices((img_size, img_size))
    distances = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    image[distances < center * radius_ratio] = volume/0.78
    return image
