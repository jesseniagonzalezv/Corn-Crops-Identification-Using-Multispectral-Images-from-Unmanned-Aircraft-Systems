#####################
'''
Create Test with the data of the train file

original_dataset_dir_total= "data_HR/unlabel/images"  ubication of the all the dataset
original_dataset_dir_train= "data_HR/unlabel/train/images" ubication of the train images  ## qutie
base_dir = "data_HR/unlabel",                 ubication of the main directory of train and test
test_split = 0.1                              size of the test 
train_file='train',                           ubication of the train file
test_file='test                               ubication of the test file
'''
#####################
import numpy as np
import shutil
import glob
import os
from pathlib import Path

########################################################################
def split_train_test(original_dataset_dir_total= "data_HR/unlabel/images",  base_dir = "data_HR/unlabel",test_split = 0.1, train_file='train',test_file='test'):
#original_dataset_dir_train= "data_HR/unlabel/train/images", 
    get_files_path = original_dataset_dir_total + "/*.npy"
    fpath_list_total = sorted(glob.glob(get_files_path))


    # make the directories


    train_dir = os.path.join(base_dir, train_file,'images')
    get_files_path = str(train_dir) + "/*.npy"
    fpath_list = sorted(glob.glob(get_files_path))
    
            
    test_dir0 = os.path.join(base_dir, test_file)
    if not os.path.exists(test_dir0):
            os.mkdir(test_dir0)
            
    test_dir = os.path.join(base_dir, test_file,'images')
    if not os.path.exists(test_dir):
            os.mkdir(test_dir)

    ###############################################################################


    total_size = len(fpath_list_total)

    dataset_size = len(fpath_list)
    indices = list(range(dataset_size))

    split = int(np.floor(test_split * total_size))

    if 1 :
        np.random.seed(1337)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    print(len(train_indices), len(test_indices))
    ############################################################

    for i in test_indices:
        fname = fpath_list[i]
        fname = fname.split("/")
        fname = (fname[-1])
        src = os.path.join(original_dataset_dir_train, fname)
        dst = os.path.join(test_dir, fname)
        shutil.move(src, dst)    





