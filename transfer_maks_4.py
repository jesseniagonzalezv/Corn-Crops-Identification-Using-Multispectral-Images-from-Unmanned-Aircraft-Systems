#####################
'''
Script: This is for for copy in a new file the labels needed

Input:
in_images_dir  ubication of the images that need labels
in_label_dir ubication of the labels 

Output:
out_label_dir:  ubication where the copied labels will be placed

'''
#####################
# ----------------Importing required packages---------------------------
import shutil
import glob
import os

# ----------------transferring masks corresponding to correct input-------------
def transfer_masks(in_images_dir="data_HR/train/images",in_label_dir="data_HR/data/masks", out_label_dir="data_HR/train/masks/"): 

    print(in_images_dir)


    # copying files to their correspondig folder-------------------------
    ## Getting names of all files in the folder------------------------------------
    get_files_path = in_images_dir + "/*.npy"
    #print(len(get_files_path))
    fpath_list = sorted(glob.glob(get_files_path))
    

            
    if not os.path.exists(out_label_dir):
            os.mkdir(out_label_dir)
            
    
    for file in fpath_list:
        file = file.split("/")[-1]
        fname = str(file[:-4]+ "_a" + ".npy")
        #print(fname)
        src = os.path.join(in_label_dir, fname)
        dst = os.path.join(out_label_dir, fname)
        shutil.copyfile(src, dst) 
#____________________________________________________

def obtained_mask(mode='test',in_label_dir= "data_HR/data/masks/",in_images_dir_train="data_HR/train/images/", out_label_dir_train="data_HR/train/masks/" ,in_images_dir_val="data_HR/val/images/",out_label_dir_val="data_HR/val/masks/" ,in_images_dir_test="data_HR/test/images/",out_label_dir_test="data_HR/test/masks/"):
    
    
    transfer_masks(in_images_dir_train,in_label_dir,out_label_dir_train)
    if (mode=='val'):
        transfer_masks(in_images_dir_val,in_label_dir,out_label_dir_val )
    if (mode=='test'):
        transfer_masks(in_images_dir_test,in_label_dir, out_label_dir_test )


#   
    
    
