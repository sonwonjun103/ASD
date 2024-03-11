import math
import numpy as np
import nibabel as nib

def image_entropy(path, split, direction):
    entropy = {}
    
    # read image
    nii = nib.load(path)
    img_volume = nii.get_fdata()
    
    if direction == 'axial':
        img_volume = img_volume.transpose(2,1,0)
    elif direction == 'sagittal':
        img_volume = img_volume.transpose(1,2,0)
    elif direction == 'coronal':
        img_volume = img_volume.transpose(0,2,1)
    
    slice = img_volume.shape[0]
    cnt = 0
    
    for i in range(slice):
        d=[]
        img=img_volume[i]
        
        # pixel reshape
        img_pixel = np.reshape(img, -1)
        set_img_pixel = list(set(img_pixel))
        set_img_pixel = sorted(set_img_pixel)
        
        n = len(set_img_pixel)
        
        for i in range(0, split):
            Ij = (i*n) / 10
            v = (set_img_pixel[math.floor(Ij)]) + (Ij - math.floor(Ij)) * ( (set_img_pixel[math.ceil(Ij)]) - (set_img_pixel[math.floor(Ij)]) )
            d.append(v)
            
        # the largest value
        d.append(max(img_pixel))
        
        # d1 < x < d9
        
        reslice_pixel = []
        
        for pixel in img_pixel:
            if pixel > d[1] and pixel < d[9]:
                reslice_pixel.append(pixel)
                
        set_pixel = list(set(reslice_pixel))
        
        slice_entropy = 0
        
        for i in range(len(set_pixel)):
            pik = (reslice_pixel.count(set_pixel[i])) / (len(img_pixel))
            spik = pik*np.log(pik)
            slice_entropy-=spik
            
        cnt +=1
        entropy[cnt] = slice_entropy
        
    return entropy, img_volume