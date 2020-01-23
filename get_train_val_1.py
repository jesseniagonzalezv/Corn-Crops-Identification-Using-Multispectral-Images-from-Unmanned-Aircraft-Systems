#####################
'''
Create Train and val with all the dataset 

original_dataset_dir= "data_HR/unlabel/images"  ubication of all the dataset
base_dir = "data_HR/unlabel" ubication of the main directory where copy all the dataset in the files train and val
validation_split = 0.2 size of the validation 
train_file='train' ubication of the train
val_file='val      ubication of the valid
'''
#####################

import numpy as np
import shutil
import glob
import os
from pathlib import Path

########################################################################
def split_train_val(original_dataset_dir= "data_HR/unlabel/images" ,base_dir = "data_HR/unlabel",validation_split = 0.2, train_file='train',val_file='val'):

    

    get_files_path = original_dataset_dir + "/*.npy"
    fpath_list = sorted(glob.glob(get_files_path))

    # make the directories
    train_dir0 = os.path.join(base_dir, train_file) 
    if not os.path.exists(train_dir0):
            os.mkdir(train_dir0)

    validation_dir0 = os.path.join(base_dir, val_file)
    if not os.path.exists(validation_dir0):
            os.mkdir(validation_dir0)
            
    train_dir = os.path.join(base_dir, train_file,'images') #only create the last-images
    if not os.path.exists(train_dir):
            os.mkdir(train_dir)

    validation_dir = os.path.join(base_dir, val_file,'images')
    if not os.path.exists(validation_dir):
            os.mkdir(validation_dir)

    ###############################################################################
    

    dataset_size = len(fpath_list)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))


    if 1 :
        #np.random.seed(1337)
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]
    print(dataset_size,len(train_indices), len(valid_indices))
    ############################################################

    for i in train_indices:
        fname = fpath_list[i]
        fname = fname.split("/")
        fname = (fname[-1])
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dir, fname)
        shutil.copyfile(src, dst)


    for i in valid_indices:
        fname = fpath_list[i]
        fname = fname.split("/")
        fname = (fname[-1])
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dir, fname)
        shutil.copyfile(src, dst)    





