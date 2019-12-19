import helper
from collections import defaultdict
from helper import reverse_transform2
from torch.utils.data import DataLoader
from loss import dice_loss,metric_jaccard
from dataset import WaterDataset
import torch.nn.functional as F
from models import UNet11
import numpy as np
import torch
import glob
import os
from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)



PATH = 'logs/mapping/HR_model_temp2.pth'

#Initialise the model
num_classes = 1 
model = UNet11(num_classes=num_classes)
model.cuda()
model.load_state_dict(torch.load(PATH))
model.eval()   # Set model to evaluate mode
######################### setting all data paths#######
outfile_path = 'predictions'
data_path = 'data_HR'

test_path= "data_HR/test/images" ##

get_files_path = test_path + "/*.npy"
test_file_names = np.array(sorted(glob.glob(get_files_path)))
###################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

f = open("predictions/pred_loss_dist.txt", "w+")
f = open("predictions/pred_loss_dist_400.txt", "w+")
f = open("predictions/pred_loss_dist_100.txt", "w+")

test_transform = DualCompose([
        CenterCrop(512),
        ImageOnly(Normalize())
    ])

def make_loader(file_names, shuffle=False, transform=None,mode='train',batch_size=1, limit=None):
        return DataLoader(
            dataset=WaterDataset(file_names, transform=transform, limit=limit),
            shuffle=shuffle,            
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available() #### in process arguments
        )
    
test_loader = make_loader(test_file_names, shuffle=False, transform=test_transform, mode='train', batch_size = 1)
        
metrics = defaultdict(float)
def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
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


count=0
input_vec= []
labels_vec = []
pred_vec = []
epoch_samples = 0  #########

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

        count += 1
        print(count)


np.save(str(os.path.join(outfile_path,"inputs_test_dist_p_{}.npy".format(int(count)))), np.array(input_vec))
np.save(str(os.path.join(outfile_path,"labels_test_distp_p_{}.npy".format(int(count)))), np.array(labels_vec))
np.save(str(os.path.join(outfile_path,"pred_test_dist_p_{}.npy".format(int(count)))), np.array(pred_vec))

