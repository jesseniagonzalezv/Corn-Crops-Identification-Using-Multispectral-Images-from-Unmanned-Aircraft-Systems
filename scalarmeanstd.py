"""
Adaptate from https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
Calculate Mean and Std per channel
and standard deviation in the training set, 
suggestions: http://cs231n.github.io/neural-networks-2/#datapre
Input:images HxWxCH
Output: mean, std 
"""

import numpy as np
from os import listdir
from os.path import join, isdir
import glob 
import cv2
import timeit
from pathlib import Path


def find_max(im_pths):
    
    
    minimo_pixel=[]
    maximo_pixel=[]
    size=len(im_pths) 

    for i in im_pths:
        img = np.load(str(i)).transpose((1, 2, 0))
           
        minimo_pixel.append(np.min(img))
        maximo_pixel.append(np.max(img))

    return   np.min(minimo_pixel),np.max(maximo_pixel), size
        

    
def cal_dir_stat(im_pths, maximunValue,CHANNEL_NUM): ##give the names 
    pixel_num = 0
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)



    for path in im_pths:
        im = np.load(path).transpose(1,2,0) #  because input imgs is CHXHXW
        im= im[:,:,:CHANNEL_NUM]
        im = im/maximunValue
        pixel_num += (im.size/CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))


    rgb_mean = channel_sum / pixel_num
    rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))
       
    return rgb_mean, rgb_std


def meanstd(train_root,val_root, test_root,rootdata='dataset/data_512',CHANNEL_NUM='6'): #name_file,
    data_path = Path(rootdata)
 
    data_all_raiz=str(data_path/'images')  
    get_files_path = str(data_all_raiz) + "/*.npy"
    data_all_root = np.array(sorted(glob.glob(get_files_path)))
    
    minimo_pixel_train,maximo_pixel_train,size_train = find_max(train_root)
    minimo_pixel_val,maximo_pixel_val,size_val = find_max(val_root)
    minimo_pixel_test,maximo_pixel_test,size_test = find_max(test_root)
    
    
    minimo_pixel_all,maximo_pixel_all,size_all = find_max(data_all_root)
    
    print('Train:',str(data_path),size_train, 'min ',np.min(minimo_pixel_train),'max ',maximo_pixel_train) # 0-1
    print('Val:',str(data_path),size_val,'min ',np.min(minimo_pixel_val),'max ',maximo_pixel_val) # 0-1
    print('Test:',str(data_path),size_test,'-min ',np.min(minimo_pixel_test),'max ',maximo_pixel_test) # 0-1        
    print('All:',str(data_path),size_all,'min ',np.min(minimo_pixel_all),'max ',maximo_pixel_all) # 0-1
    
    print(CHANNEL_NUM)
    start = timeit.default_timer()
    mean_train, std_train = cal_dir_stat(train_root,maximo_pixel_all,CHANNEL_NUM) #max 3521
    mean_val, std_val = cal_dir_stat(val_root,maximo_pixel_all,CHANNEL_NUM)
    mean_test, std_test = cal_dir_stat(test_root,maximo_pixel_all,CHANNEL_NUM) 
    mean_all, std_all = cal_dir_stat(data_all_root,maximo_pixel_all,CHANNEL_NUM)

    end = timeit.default_timer()
    print("elapsed time: {}".format(end-start))
    ''' 
    print('Train:',str(data_path),size_train, 'min ',np.min(minimo_pixel_train),'max ',maximo_pixel_train) # 0-1
    print('Val:',str(data_path),size_val,'min ',np.min(minimo_pixel_val),'max ',maximo_pixel_val) # 0-1
    print('Test:',str(data_path),size_test,'-min ',np.min(minimo_pixel_test),'max ',maximo_pixel_test) # 0-1
    print('All:',str(data_path),size_all,'min ',np.min(minimo_pixel_all),'max ',maximo_pixel_all) # 0-1
'''
    print("Train mean:{}\nstd:{}".format(mean_train, std_train))
    print("Val mean:{}\nstd:{}".format(mean_val, std_val))
    print("Test mean:{}\nstd:{}".format(mean_test, std_test))
    print("All mean:{}\nstd:{}".format(mean_all, std_all))
    
    return maximo_pixel_all, mean_train, std_train




