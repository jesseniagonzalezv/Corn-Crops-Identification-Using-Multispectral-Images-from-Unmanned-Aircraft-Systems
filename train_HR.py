'''
This is the main code 
Ask the argument
Make the loaders
Make the train
'''
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
import json
from models import UNet11,UNet, AlbuNet34, SegNet
from dataset import WaterDataset
from torch.optim import lr_scheduler   ####
import utilsTrain_HR 
import torch.optim as optim 
import numpy as np 
import glob  ###
import os

from get_train_test_kfold import get_split_out, percent_split, get_split_in

from split_train_val import get_files_names
from scalarmeanstd import meanstd
from metrics_prediction_2 import find_metrics

from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)



def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold-out', type=int, help='fold train test', default=0)
    arg('--fold-in', type=int, help='fold train val', default=0)
    arg('--percent', type=float, help='percent of data', default=1)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=4)
    arg('--limit', type=int, default=10000, help='number of images in epoch')
    arg('--n-epochs', type=int, default=40)
    arg('--lr', type=float, default=1e-3)
    arg('--model', type=str, default='UNet11', choices=['UNet11','UNet','AlbuNet34','SegNet'])

    args = parser.parse_args()
    
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1 
    if args.model == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    elif args.model == 'AlbuNet34':
        model = AlbuNet34(num_classes=num_classes, num_input_channels=4, pretrained=False)
    elif args.model == 'SegNet':
        model = SegNet(num_classes=num_classes, num_input_channels=4, pretrained=False)
    else:
        model = UNet11(num_classes=num_classes, input_channels=4)

    
    if torch.cuda.is_available():
        if args.device_ids:#
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()# to run the code in multiple gpus
    #loss = utilsTrain.calc_loss(pred, target, metrics, bce_weight=0.5) #check it is utilstrain

    cudnn.benchmark = True


    out_path = Path('logs_HR/mapping/')


    ####################Change the files_names ######################################
    data_path = Path('data_HR') # change the data path here 
    
    name_file = '_'+ str(int(args.percent*100))+'_percent'
    data_all='data' ##file que contiene todo la data 
    #name_file='_HR_100'
    #name_file='_HR_400'
    #name_file='_HR_916' 
    #name_file='_HR_dist'
    #name_file='_dist_60'
    #name_file='_dist_60_2'
    #name_file='_HR_60_fake'
    #name_file='_HR_116_fake'

        
    #print("data_path:",data_path) 
    #train_path= str(data_path/'train{}'/'images').format(name_file)+ "/*.npy"
    #val_path= str(data_path/'val{}'/'images').format(name_file)+ "/*.npy" 
    #train_file_names = np.array(sorted(glob.glob(train_path)))
    #val_file_names = np.array(sorted(glob.glob(val_path)))
    
    #################################################################################  
    # Nested cross validation K-fold train test
    #train_val_file_names, test_file_names = get_split_out(data_path,data_all,args.fold_out)
    #################################################################################  

    train_val_file_names=np.array(sorted(glob.glob(str(data_path/'data_850'/'images')+ "/*.npy")))
    test_file_names =  np.array(sorted(glob.glob(str(data_path/'test_850'/'images') + "/*.npy")))
    if args.percent !=1:
        extra, train_val_file_names= percent_split(train_val_file_names, args.percent) 

    #################################################################################  
    
 
    
    train_file_names,val_file_names = get_split_in(train_val_file_names,args.fold_in)    #train_file_names, val_file_names = get_files_names(data_path,name_file)


    np.save(str(os.path.join(out_path,"train_files{}_{}_fold{}_{}.npy".format(name_file,args.model,args.fold_out,args.fold_in))), train_file_names)
    np.save(str(os.path.join(out_path,"val_files{}_{}_fold{}_{}.npy".format(name_file,args. model,args.fold_out, args.fold_in))), val_file_names)
    
    ######## 733
    #train_path= data_path/'train'/'images'
    #val_path= data_path/'val'/'images'
    
    ######## 100
    #train_path= data_path/'train_100'/'images'
    #val_path= data_path/'val_100'/'images'
    
    ##############400
    #train_path= data_path/'train_400'/'images'
    #val_path= data_path/'val_400'/'images'
    ##############
    #train_path= data_path/'dist_per'/'train_HR'/'images'
    #val_path= data_path/'dist_per'/'val_HR'/'images'



    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    
    def make_loader(file_names, shuffle=False, transform=None,mode='train',batch_size=4, limit=None):
        return DataLoader(
            dataset=WaterDataset(file_names, transform=transform,mode=mode, limit=limit),
            shuffle=shuffle,            
            batch_size=batch_size, #args.batch_size,
            pin_memory=torch.cuda.is_available() #### in process arguments
        )
    ########return value of mean_std_train
    max_values, mean_values, std_values=meanstd(train_file_names, val_file_names,test_file_names,str(data_path)) #_60 --data_HR, data_LR

    train_transform = DualCompose([
        CenterCrop(512),
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),    
        ImageOnly(Normalize(mean_values, std_values))
    ])

    val_transform = DualCompose([
        CenterCrop(512),
        ImageOnly(Normalize(mean_values, std_values))
    ])
#albunet 34 with only 3 batch_size
    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform, mode='train', batch_size = 4) 
    valid_loader = make_loader(val_file_names, transform=val_transform, batch_size = 4, mode = "train")


    dataloaders = {
    'train': train_loader, 'val': valid_loader
    }

    dataloaders_sizes = {
    x: len(dataloaders[x]) for x in dataloaders.keys()
    }

    
    root.joinpath('params_HR.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr= args.lr)  #
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1) 
    
    
    utilsTrain_HR.train_model(
   
        name_file=name_file,
        model=model,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
        dataloaders=dataloaders,
        fold_out=args.fold_out,
        name_model=args.model,
        num_epochs=args.n_epochs 
        )

  #  torch.save(model.module.state_dict(), out_path/'modelHR_40epoch.pth')
    #torch.save(model.module.state_dict(), out_path/'modelHR_40epoch_100.pth')
    #torch.save(model.module.state_dict(), out_path/'modelHR_40epoch_400.pth')
    #torch.save(model.module.state_dict(), out_path/'modelHR_40epoch_fake.pth')
    torch.save(model.module.state_dict(),(str(out_path)+'/model_40epoch{}_{}_fold{}.pth').format(name_file,args.model,args.fold_out)) #I am saving the last model of k_fold
    
    print(args.model)
    max_values_all_data=3521
    

    find_metrics(train_file_names, val_file_names, test_file_names, max_values_all_data, mean_values, std_values, args.fold_out, args.fold_in,model, args.model, out_file='HR', dataset_file='HR',name_file=name_file)   
    #out_file=the file of the outputs: HR, LR, distil, dataset_file=ubication of the data:HR, LR, name_files: HR_916,HR_100, HR_60_fake
        
if __name__ == '__main__':
    main()
