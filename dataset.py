'''
This code is to load images and masks: data loader

Input:
-Images and Masks  (H,W,CH)

Output:
- Images after transformations and convert to float tensor (CH,H,W)
''' 

import torch
import numpy as np
from torch.utils.data import Dataset



class ImagesDataset(Dataset):
    def __init__(self, img_paths: list, channels:list, transform=None, mode='train', limit=None):
        self.img_paths = img_paths
        self.channels =channels
        self.transform = transform
        self.mode = mode
        self.limit = limit

    def __len__(self):
        if self.limit is None:
            return len(self.img_paths)
        else:
            return self.limit

    def __getitem__(self, idx):
        if self.limit is None:
            img_file_name = self.img_paths[idx]
        else:
            img_file_name = np.random.choice(self.img_paths)
            

        img = load_image(img_file_name,self.channels )
        #print(self.mode)

        if self.mode == 'train':
            mask = load_mask(img_file_name)

            img, mask = self.transform(img, mask)

            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            mask = np.zeros(img.shape[:2])
            img, mask = self.transform(img, mask)

            return to_float_tensor(img), str(img_file_name) 


def to_float_tensor(img):
    img=torch.from_numpy(np.moveaxis(img, -1, 0)).float()  
    return img


def load_image(path,channels): #in CH, H,W  out: H,W,CH
    img = np.load(str(path))
    img=img.transpose((1, 2, 0))  
    dimsH=img.shape[0]
    dimsW=img.shape[1]
    dimsCH=len(channels)


    imga = np.zeros((dimsH,dimsW,dimsCH))

    
    for i,ch in enumerate(channels):
        imga[:,:,i] =img[:,:,ch]
        
        
    ##TRAIN RGB 3 o RGBNIR 4
    #img = img[:,:,:4]
    #TRAIN R 0 o NIR 4
    #imga = np.zeros((160,160,2))
    #imga[:,:,0] = img[:,:,0]
    #imga[:,:,1] = img[:,:,3]

    return  imga 

def load_mask(path):   #H,W,CH   
    mask = np.load(str(path).replace('images', 'masks').replace(r'.npy', r'_a.npy'), 0)
    #mask=mask.reshape(mask.shape[1],-1)
#    mask =np .max(mask, axis=2)  #convert of 3 channel to 1 channel
#    mask=(mask > 0).astype(np.uint8)
#    return mask
    mask=mask.transpose(1, 2, 0).reshape(mask.shape[1],-1)
    mask=(mask > 0).astype(np.uint8)
    return mask
