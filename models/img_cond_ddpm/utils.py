import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import glob, os

def make_mask(paths, save_dir):
    for path in paths:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        img_name = path.split('/')[-1]
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 이진화 수행
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.drawContours(image.copy(), contours, -1, (0, 0, 0), 6)
        
        color = (255,255, 255)
        for cont in contours:
            pts = cont.reshape(-1, 2)
            cv2.fillPoly(result, [pts], color)
        
        cv2.imwrite(f'{save_dir}/{img_name}', result)
        
def grid_imgs(imgs, titles=[], row=1, col=None, size=2, save=None):
    if row == 1: 
        col = len(imgs)
    assert row * col == len(imgs)
    
    fig, axes = plt.subplots(row,col,figsize=(col*size, row*size))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.set_axis_off()
        ax.imshow(imgs[i], cmap='gray')
        if len(titles)==len(imgs):
            ax.set_title(titles[i])
            
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    plt.show()

def erode_img(image, ratio=2):
    img = np.array(image, dtype=np.uint8)
    _, binary_image = cv2.threshold(img , 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((ratio,ratio), np.uint8)
    eroded_image = cv2.dilate(binary_image, kernel, iterations=1)

    return eroded_image

def rotate_arr(arr, angle=90):
    x = arr.copy()
    x = np.rot90(x, k=angle // 90)
    return x.astype(float)
    
def get_vf_from_img(img, color='w', binary=True):
    arr = np.array(img, dtype=np.uint8).copy()
    if binary:
        _, binary_image = cv2.threshold(arr , 127, 255, cv2.THRESH_BINARY)
        arr = np.array(binary_image, dtype=np.uint8)
    
    vf = np.mean(arr)
    vf = vf / 255.
    
    if color != 'w':
        vf = 1-vf
    
    return vf

def pad_img(image, target_size=128):
    width, height = image.size

    # Calculate the aspect ratio
    aspect_ratio = width / height
    
    # Calculate the new dimensions while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    
    # Resize the image while preserving the aspect ratio
    resized_image = image.resize((new_width, new_height))
    
    # Create a new blank image with the target size
    pad = Image.new('L', (target_size, target_size), 0)
    
    # Calculate the position to paste the resized image
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    
    # Paste the resized image onto the blank image
    pad.paste(resized_image, (x_offset, y_offset))
    
    return pad
    

def ratio_img(image, ratio=1.):
    width, height = image.size
    size = max(width, height)

    aspect_ratio = width / height
    aspect_ratio = aspect_ratio*ratio
    # Calculate the new dimensions while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_width = int(size * aspect_ratio)
        new_height = size
    
    # Resize the image while keeping the aspect ratio unchanged
    resized_image = image.resize((new_width, new_height))
    
    return resized_image