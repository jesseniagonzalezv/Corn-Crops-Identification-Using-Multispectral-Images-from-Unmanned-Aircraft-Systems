from datetime import datetime
from pathlib import Path
import random
import numpy as np
import time
import torch
import copy      
from collections import defaultdict
import torch.nn.functional as F
import os

from loss import dice_loss,metric_jaccard
from metrics_prediction_2 import calc_loss,print_metrics 


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
if (device=='cpu'):
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


def train_model(dataset_file,name_file, model, optimizer, scheduler,dataloaders,fold_out='0', fold_in='0', name_model='UNet11',num_epochs=25):
    hist_lst = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    f = open("history/{}/history_model{}_{}_foldout{}_foldin{}_{}epochs.txt".format(dataset_file,name_file,name_model,fold_out,fold_in,num_epochs), "w+")
   
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        f.write('Epoch {}/{}'.format(str(epoch), str(num_epochs - 1)) + "\n")
        print('-' * 10)
        f.write(str('-' * 10) + "\n") 
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    f.write("LR" +  str(param_group['lr']) + "\n") 

                model.train()  # Set model to training mode
            else:
                    model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            itr = 0
            print("dataloader:",len(dataloaders[phase]) )
            f.write("dataloader:" + str(len(dataloaders[phase])) + "\n") 
            for inputs, labels in dataloaders[phase]:
               # print(device)
                inputs = inputs.to(device)
                labels = labels.to(device)             

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics,f, phase, epoch_samples)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                f.write("saving best model" + "\n")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        f.write('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + "\n")
    print('Best val loss: {:4f}'.format(best_loss))
    f.write('Best val loss: {:4f}'.format(best_loss)  + "\n")
    f.close()

    model.load_state_dict(best_model_wts)
    return model