import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import glob
import unlabeled_helper
from collections import defaultdict
from torch.utils.data import DataLoader
from dataset import WaterDataset
import torch.nn.functional as F
from models import UNet11
import numpy as np
import torch
import glob

from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)


def make_loader(file_names, shuffle=False, transform=None, limit=None):
        return DataLoader(
            dataset=WaterDataset(file_names, transform=transform, limit=limit, mode='test'),
            shuffle=shuffle,            
            batch_size=1,
            pin_memory=torch.cuda.is_available() 
        )
    
def unlabel_prediction(PATH_model,unlabel_name_file):
    num_classes = 1 
    model = UNet11(num_classes=num_classes)
    model.cuda()
    model.load_state_dict(torch.load(PATH_model))
    model.eval()   
    ######################### setting all data paths#######
    outfile_path = 'predictions_VHR/unlabel_test/'
    data_path = 'data_VHR'
    test_path= "data_VHR/unlabel/" + unlabel_name_file  

    get_files_path = test_path + "/*.npy"
    test_file_names = np.array(sorted(glob.glob(get_files_path)))
    ###################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_transform = DualCompose([
            CenterCrop(512),
            ImageOnly(Normalize())
        ])
    
    test_loader = make_loader(test_file_names, shuffle=False, transform=test_transform)
    metrics = defaultdict(float)

    count_img=0
    input_vec= []
    pred_vec = []
    for inputs , name in test_loader:
        inputs = inputs.to(device)            
        with torch.set_grad_enabled(False):
            input_vec.append(inputs.data.cpu().numpy())
            pred = model(inputs)
            pred=torch.sigmoid(pred) 

            pred_vec.append(pred.data.cpu().numpy())
            count_img += 1
    print(count_img)
    name_imgs=  outfile_path +unlabel_name_file+ "_inputs_unlab_" + str(count_img) + ".npy"         
    name_preds=  outfile_path +unlabel_name_file+ "pred_unlab_" +  str(count_img) + ".npy"     
 
    np.save(name_imgs, np.array(input_vec))
    np.save(name_preds, np.array(pred_vec))
    return  name_imgs,name_preds


def plot_prediction(path_model,unlabel_name_file): 
    val_file,pred_file= unlabel_prediction(path_model,unlabel_name_file)

    val_images = np.load(val_file)
    pred_images = np.load(pred_file)
    pred_images[0,0,:,:,:].shape

    input_images_rgb = [unlabeled_helper.reverse_transform(x) for x in val_images[:,0,:3,:,:]]  #[:10,0,:3,:,:]limited to 10 images
    pred_rgb = [unlabeled_helper.masks_to_colorimg(x) for x in pred_images[:,0,:,:,:]]#[:10,0,:3,:,:]limited to 10 images
    unlabeled_helper.plot_side_by_side([input_images_rgb, pred_rgb],save=1)
    

