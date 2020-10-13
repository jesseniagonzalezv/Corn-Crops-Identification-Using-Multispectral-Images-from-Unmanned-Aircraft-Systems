"""
Create the Perusat dataset 
Images with RGBNIR bands

Input:
Original dataset: 
The images from PeruSAT-1 satellite in .TIF format with a size approximately of 6000X6000size, 4 bands
Labels were create with QGIS and defined by hand and also using the filter_mask_1step.py



Output:
Dataset: 915 images of 6 bands  
Image output: C X H X W  
C: 0 red, 1 green, 2 blue, 3 nir
Label:C X H X W  
"""

import argparse
import numpy as np
import glob 
import timeit
import csv
import os
import shutil
from pathlib import Path
from cropImages import splits_images
from cropMasks import splits_masks

 

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
####### Images ###############################################################################
def create_crop_dataset_images(out_path_images='data_512/images',input_filename='images/IMG_PER1_20170422154946_ORT_MS_003749.TIF',index_imgs=0):
    start = timeit.default_timer()
    
 
    
    output_filename = 'images_tif/{}_{}-{}.tif'
    output_filename_npy = '{}_{}-{}.npy'
    output_filename_npyblack = 'images_black/{}_{}-{}.npy'
    img_black_paths=splits_images(out_path_images,input_filename,output_filename,output_filename_npy,output_filename_npyblack, index_imgs)
    end = timeit.default_timer()
    print("Images-elapsed time: {}".format(end-start))
    return img_black_paths
 
########Masks ###############################################################################
def create_crop_dataset_masks(out_path_mask='data_512/masks', mask_filename='maks/img_d4534.tif',index_imgs=0):
    start = timeit.default_timer()

    output_filename = 'masks_tif/{}_{}-{}_a.tif'
    output_filename_npy = '{}_{}-{}_a.npy'

    splits_masks(out_path_mask,mask_filename,output_filename,output_filename_npy, index_imgs)

  
    end = timeit.default_timer()
    print("Masks-elapsed time: {}".format(end-start))
 
#############################################################################################

    
    


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--dataset-path', type=str, default='dataset', help='main file,in which the dataset is')
    arg('--dataset-file', type=str, default='data_512', help='dataset of a specific size')
    args = parser.parse_args()    
    
    
    print('Cortando pathes----------------')
        ## falta crear automaticamente el masks tif and images _160 pituputs tif and crops
    ####### Images File ##################
    data_path = Path(args.dataset_path) 
    
    out_path_images = str(data_path/args.dataset_file/'images')
    print('out_path_images',out_path_images)

    myData = [["input_id", "source_id", "coordinates(rows,col)", "porcentaje"]]              
    myFile = open('splits_images.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)  #list

    ####### Masks File ##################
    out_path_mask= str(data_path/args.dataset_file/'masks')
    myData = [["input_id", "source_id", "coordinates(rows,col)"]] 

    myFile = open('splits_masks.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)      

    mkdirs(out_path_images + '/images_black')
    mkdirs(out_path_mask + '/masks_black')
    mkdirs(out_path_images + '/images_tif')
    mkdirs(out_path_mask +'/masks_tif')

   
    
    dataset_images=np.array(sorted(glob.glob(str(data_path/'images')+ "/*.tif")))

    
    #np.array(sorted(list(predict_list.glob('*_fake.png'))))

    print('images founded:', dataset_images.shape)
    for i, input_path in enumerate(dataset_images):
        
            print(i,input_path)
            mask_path =input_path.replace('images','masks')
            print(i,mask_path)        
            index_imgs=i
            img_black_paths=create_crop_dataset_images(out_path_images,input_path,index_imgs)
            create_crop_dataset_masks(out_path_mask,mask_path,index_imgs)
            
            print('black images >15%',np.shape(img_black_paths))        

            for img_black in img_black_paths:
                    dst =img_black.replace('images/images_black','masks/masks_black').replace(r'.npy', r'_a.npy')
                    src =img_black.replace('images/images_black','masks').replace(r'.npy', r'_a.npy')
                    shutil.move(src, dst) 
        
        
if __name__ == '__main__':
    main()