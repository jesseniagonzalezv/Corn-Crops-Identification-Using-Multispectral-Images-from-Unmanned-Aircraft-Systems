#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:59:37 2019
split train and val
@author: debjani
python split_train_val.py
"""
import numpy as np
import shutil
import glob
import os
from pathlib import Path

########################################################################

original_dataset_dir= "data_HR/data/images" 
get_files_path = original_dataset_dir + "/*.npy"
fpath_list = sorted(glob.glob(get_files_path))


# make the directories
base_dir = "data_HR" 
train_dir = os.path.join(base_dir, 'train','images')
if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    
validation_dir = os.path.join(base_dir, 'val','images')
if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
        
        
###############################################################################
validation_split = 0.2
dataset_size = len(fpath_list)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if 1 :
    np.random.seed(1337)
    np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]
print(len(train_indices), len(valid_indices))
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





