import cv2
import numpy as np

def min_max_normalize(img):
    normalize_img = (img-np.min(img)) / np.max(img) 
    normalize_img *= 2**8-1
    normalize_img = normalize_img.astype(np.uint8)
    return normalize_img

def mean_std_normalize(img):
    return (img - np.mean(img)) / np.std(img)

# def clahe_normalization(img):
#     img = (img*255).astype(np.uint8)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     img2 = clahe.apply(img)

#     return img2