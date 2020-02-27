#####################
'''
This code create the files to train and test(10% percent)

Input:
    original_dataset_dir= "data_HR/data/images"  ubication of all the dataset
    base_dir = "data_HR/data" ubication of the main directory where copy all the dataset 
    test_split = 0.1 size of the test 
    train_file='train' ubication of the train
    test_file='test'      ubication of the test
Output:
    File train and test each with 
    from pathlib import Path
    │   └── train
    │           ├── images
    │           └── masks
    │   └── test
    │           ├── images
    │           └── masks
'''
#####################

import numpy as np
import shutil
import glob
import os
from pathlib import Path
from transfer_maks_4 import obtained_mask

########################################################################
def split_train_test(original_dataset_dir= "data_HR/data/images" ,base_dir = "data_HR",test_split = 0.1, train_file='train',test_file='test'):

    

    get_files_path = original_dataset_dir + "/*.npy"
    fpath_list = sorted(glob.glob(get_files_path))

    # make the directories
    train_dir0 = os.path.join(base_dir, train_file) 
    if not os.path.exists(train_dir0):
            os.mkdir(train_dir0)

    test_dir0 = os.path.join(base_dir, test_file)
    if not os.path.exists(test_dir0):
            os.mkdir(test_dir0)
            
    train_dir = os.path.join(train_dir0,'images') #only create the last-images
    if not os.path.exists(train_dir):
            os.mkdir(train_dir)

    test_dir = os.path.join(test_dir0,'images')
    if not os.path.exists(test_dir):
            os.mkdir(test_dir)

    ###############################################################################
    

    dataset_size = len(fpath_list)
    indices = list(range(dataset_size))

    split = int(np.floor(test_split * dataset_size))


    if 1 :
        #np.random.seed(1337)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    print(dataset_size,len(train_indices), len(test_indices))
    ############################################################

    for i in train_indices:
        fname = fpath_list[i]
        fname = fname.split("/")
        fname = (fname[-1])
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dir, fname)
        shutil.copyfile(src, dst)


    for i in test_indices:
        fname = fpath_list[i]
        fname = fname.split("/")
        fname = (fname[-1])
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dir, fname)
        shutil.copyfile(src, dst)    


    labels_dir = os.path.join(base_dir,'data', 'masks') #only create the last-images

    obtained_mask(mode="test",
                  in_label_dir=labels_dir,
                  in_images_dir_train=train_dir,
                  out_label_dir_train=os.path.join(train_dir0,'masks') ,
                  in_images_dir_test=test_dir, 
                  out_label_dir_test=os.path.join(test_dir0,'masks') )


