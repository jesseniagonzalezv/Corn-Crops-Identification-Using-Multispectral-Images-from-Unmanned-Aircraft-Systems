from datetime import datetime
from pathlib import Path
import random
import numpy as np
import time
import torch
#import tqdm
import copy      

from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss,metric_jaccard
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if (device=='cpu'):
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


#def cuda(x):
#    return x.cuda(async=True) if torch.cuda.is_available() else x


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    pred=(pred >0.5).float()  #!!!!!!  th 0.55 
    jaccard_loss = metric_jaccard(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    # convering tensor to numpy to remove from the computationl graph 
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice_loss'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    metrics['jaccard_loss'] += jaccard_loss.data.cpu().numpy() * target.size(0)
    return loss

def print_metrics(metrics, epoch_samples, phase, f):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))
    f.write("{}: {}".format(phase, ", ".join(outputs)))    ### f

def train_model(name_file, model, optimizer, scheduler,dataloaders,fold_out, name_model='UNet11',num_epochs=25):

    ##name depend
    #name_save='_400' 
    #finally_path = Path('logs_HR')

    #final_layer = finally_path/'mapping'/'final_layer'
    #final_layer = Path('logs/mapping/final_layer')
    #final_layer_npy_outpath = 'final_layer_{}.npy'

    hist_lst = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    #f = open("history_model1.txt", "w+")
    #f = open("history_model1_100.txt", "w+")
    #f = open("history_model1_400.txt", "w+")
   # f = open("history_modelHR_fake.txt", "w+")
    f = open("history_HR/history_model{}_{}_fold{}.txt".format(name_file,name_model,fold_out), "w+")

   
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
                inputs = inputs.to(device)
                labels = labels.to(device)             

                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs, conv1 = model(inputs)
                    outputs = model(inputs)
                    
                    # saving batch prediction _final_layer 
                    #if itr == 0:
                    #    final_layer_data = outputs.data.cpu().numpy()
                    #    outpath_final_layer=str(os.path.join(final_layer,final_layer_npy_outpath.format(int(epoch))))

                        #np.save(final_layer/"final_layer_" + str(epoch) + ".npy" , final_layer_data)
                     #   np.save(outpath_final_layer, final_layer_data)
                        
                    #saving conv1_data
                    #conv1_data = conv1.data.cpu().numpy()
                    #np.save(str(conv_path) + "conv1_" + str(epoch) + "_" + str(itr) + ".npy", conv1_data )
                    #itr = itr + 1
                    #print(itr)
                    loss = calc_loss(outputs, labels, metrics)
                    
                    #print("I am here 2")
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
                #print(epoch_samples)
            #print("I am here 3")
            print_metrics(metrics, epoch_samples, phase, f)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                f.write("saving best model" + "\n")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        #hist_lst.append(metrics)
        #with open(final_layer/'loss.txt', "w") as file:
        #    file.write(str(hist_lst))
        
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        f.write('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + "\n")
    print('Best val loss: {:4f}'.format(best_loss))
    f.write('Best val loss: {:4f}'.format(best_loss)  + "\n")
    f.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model