# prediction mask in HR images downscale 

#import math
import helper
#from pathlib import Path
from collections import defaultdict
from helper import reverse_transform2
from torch.utils.data import DataLoader
from loss import dice_loss,metric_jaccard  #this is loss
from dataset import WaterDataset
import torch.nn.functional as F
from models import UNet11
import numpy as np
import torch
import glob
import os
import torch.nn as nn

from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        Normalize2,                            
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)




def make_loader(file_names, shuffle=False, transform=None,mode='train', limit=None):  #mode ='train' with labels
    return DataLoader(
        dataset=WaterDataset(file_names, transform=transform, limit=limit),
        shuffle=shuffle,            
        batch_size=1,
        pin_memory=torch.cuda.is_available() #### in process arguments
    )


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    #pred=(pred >0).float()  #!!no o!!!!
    jaccard_loss = metric_jaccard(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    # convering tensor to numpy to remove from the computationl graph 
    metrics['bce'] = bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] = 1-dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] = loss.data.cpu().numpy() * target.size(0)
    metrics['jaccard'] = 1-jaccard_loss.data.cpu().numpy() * target.size(0)
    return loss

def print_metrics(metrics, epoch_samples, f):    
    outputs = []
    for k in metrics.keys():
        #print(k , metrics[k])
        outputs.append("{}: {:4f}".format(k, metrics[k] ))#/ epoch_samples))
        #outputs.append(k + " " + str(metrics[k]))
        #print(outputs)
    print("{}".format(", ".join(outputs)))
    f.write("{}".format(",".join(outputs)))


def test_predition(out_file='HR',dataset_file='HR' ,name_file='_HR_dist' ):
    #PATH = ('logs_{}/mapping/model_40epoch{}.pth').format(out_file,name_file) 
    PATH = ('logs_{}/mapping/model_40epoch_HR_916.pth').format(out_file)
    outfile_path = ('predictions_{}').format(out_file)       
    f = open(("predictions_{}/metric{}.txt").format(out_file,name_file), "w+")
 
    #Initialise the model
    num_classes = 1 
    model = UNet11(num_classes=num_classes)
    model.cuda()
    model.load_state_dict(torch.load(PATH))
    model.eval()   # Set model to evaluate mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######################### setting all data paths#######
    test_path=("data_{}/test{}/images").format(dataset_file,name_file) #.format(dataset_file) 
    get_files_path3 = test_path + "/*.npy"

    test_file_names = np.array(sorted(glob.glob(get_files_path3)))
    ###################################

    if(dataset_file == 'HR'):
        test_transform = DualCompose([
                CenterCrop(512),
                ImageOnly(Normalize())
            ])

    if(dataset_file =='LR'):
        test_transform = DualCompose([
                CenterCrop(64),
                ImageOnly(Normalize2())
            ])


    test_loader = make_loader(test_file_names, transform=test_transform)

    metrics = defaultdict(float)
    
    count=0
    input_vec= []
    labels_vec = []
    pred_vec = []
    epoch_samples = 0  #########

    result_dice = []
    result_jaccard = []

        #------------------------------------------------------------
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)              
        with torch.set_grad_enabled(False):
            input_vec.append(inputs.data.cpu().numpy())
            labels_vec.append(labels.data.cpu().numpy())
            pred = model(inputs)

            epoch_samples += inputs.size(0) #### 

            loss = calc_loss(pred, labels, metrics)
            print_metrics(metrics,epoch_samples, f)

            pred=torch.sigmoid(pred) #####   
            pred_vec.append(pred.data.cpu().numpy())    

            result_dice1 += [metrics['dice']]
            
            if((metrics['jaccard'] == 0 )and (metrics['dice'] > 0.8)): 
                result_jaccard += [metrics['dice']]  #because  is all water IOU ==1
            elif((metrics['jaccard'] == 0 )and (metrics['dice'] < 0.8)): 
                result_jaccard += [metrics['jaccard'] ]
            else:           
                result_jaccard += [metrics['jaccard'] ]

            count += 1
            #print(count)

    print(("Test_{}").format(out_file))
    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ',np.mean(result_jaccard), np.std(result_jaccard),'\n')

    f.write(("Test_{}\n").format(out_file))
    f.write("dice_metric: {:4f}, std: {:4f} \n".format(np.mean(result_dice),np.std(result_dice)))
    f.write("jaccard_metric: {:4f}, std: {:4f}  \n".format(np.mean(result_jaccard), np.std(result_jaccard))) 

            #final_layer_npy_outpath.format(int(epoch)


    np.save(str(os.path.join(outfile_path,"inputs_testHR{}.npy".format(int(count)))), np.array(input_vec))
    np.save(str(os.path.join(outfile_path,"labels_testHR{}.npy".format(int(count)))), np.array(labels_vec))
    np.save(str(os.path.join(outfile_path,"pred_testHR{}.npy".format(int(count)))), np.array(pred_vec))


test_predition(out_file='HR',dataset_file='HR' ,name_file='_HR_dist' )