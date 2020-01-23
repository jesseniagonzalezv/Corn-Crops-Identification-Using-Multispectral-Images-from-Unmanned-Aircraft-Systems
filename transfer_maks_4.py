#####################
'''
Script: This is for making the validation mask data

original_dataset_dir  ubication of the images that need labels
label_dir ubication in which the labels are
data_dir  ubication in which copy the masks

'''
#####################
# ----------------Importing required packages---------------------------
import shutil
import glob
import os
# ----------------transferring masks corresponding to correct image-------------
def transfer_masks(original_dataset_dir,label_dir,data_dir): 
    #original_dataset_dir = "data_LR/train/images/"
    #original_dataset_dir = "data_LR/val/images/"
    #original_dataset_dir = "data_LR/test/images/"


    print(original_dataset_dir)
    #label_dir  = "data_LR/data/masks/"

    #data_dir  = "data_LR/train/masks/"
    #data_dir  = "data_LR/val/masks/"
    #data_dir  = "data_LR/test/masks/"

    # copying files to their correspondig folder-------------------------
    ## Getting names of all files in the folder------------------------------------
    get_files_path = original_dataset_dir + "/*.npy"
    #print(len(get_files_path))
    fpath_list = sorted(glob.glob(get_files_path))
    

            
    if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            
    
    for file in fpath_list:
        file = file.split("/")[-1]
        fname = str(file[:-4] + ".npy") #+'_a+ npy'
        #print(fname)
        src = os.path.join(label_dir, fname)
        dst = os.path.join(data_dir, fname)
        shutil.copyfile(src, dst) 
#____________________________________________________

def obtained_mask(mode='all',original_dataset_dir_train="data_LR/train/images/",label_dir_train="data_LR/data/masks/",data_dir_train="data_LR/train/masks/" ,original_dataset_dir_val="data_LR/val/images/",label_dir_val="data_LR/data/masks/",data_dir_val="data_LR/val/masks/" ,original_dataset_dir_test="data_LR/test/images/",label_dir_test="data_LR/data/masks/",data_dir_test="data_LR/test/masks/"):
    
    
    transfer_masks(original_dataset_dir_train,label_dir_train,data_dir_train)
    if (mode=='val'):
        transfer_masks(original_dataset_dir_val,label_dir_val,data_dir_val )
    if (mode=='all'):
        transfer_masks(original_dataset_dir_test,label_dir_test, data_dir_test )


#   
    
    
