import torch
import numpy as np
import cv2
import nibabel as nib
from PIL import Image

from torch.utils.data import Dataset
from utils.normalize import *

class CustomDataset(Dataset):
    def __init__(self, args, index, axial, sagittal, coronal, label, transform=None):
        self.index=index

        self.args = args
        self.axial=axial
        self.sagittal=sagittal
        self.coronal=coronal
        self.label=label
        self.transform=transform

    def get_img(self, info, index, direction):
        nii=nib.load(info.iloc[index].values[-1])
        volume = nii.get_fdata()

        entropy = info.iloc[index].values[0:20]

        if direction == 'axial':
            img = volume.transpose(2,1,0)
            slice = img.shape[0]
            img = img[slice//4 : -slice//4]

            return [img[e,:,:] for e in entropy]
        
        elif direction == 'sagittal':
            img = volume.transpose(0,2,1)
            img = np.flip(img, axis=1)
            slice = img.shape[0]
            img = img[slice//4 : -slice//4]

            return [img[e,:,:] for e in entropy]
        
        elif direction == 'coronal':
            img = volume.transpose(1,2,0)
            img = np.flip(img, axis=1)
            slice = img.shape[0]
            img = img[slice//4 : -slice//4]

            return [img[e,:,:] for e in entropy]

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        dataindex = self.index[index]
        label = self.label[index]

        axial_imgs=self.get_img(self.axial, dataindex, 'axial')
        sagittal_imgs=self.get_img(self.sagittal, dataindex, 'sagittal')
        coronal_imgs=self.get_img(self.coronal, dataindex, 'coronal')

        # concat per direction (axial, sagittal, coronal)
        data=[]
        for i in range(20):
            images=[]
            axial_entropy_slice = cv2.resize(axial_imgs[i], (self.args.image_size, self.args.image_size))
            sagittal_entropy_slice = cv2.resize(sagittal_imgs[i], (self.args.image_size, self.args.image_size))
            coronal_entropy_slice = cv2.resize(coronal_imgs[i], (self.args.image_size, self.args.image_size))

            images.append(min_max_normalize(axial_entropy_slice))
            images.append(min_max_normalize(sagittal_entropy_slice))
            images.append(min_max_normalize(coronal_entropy_slice))

            if self.transform is not None:
                # 3,256,256 => 256,256,3
                images = np.array(images).transpose(1,2,0).astype('uint8')
                images = Image.fromarray(images)
                images = self.transform(images)

            data.append(images)

        data = torch.stack(data, 0)

        return data, torch.from_numpy(label)