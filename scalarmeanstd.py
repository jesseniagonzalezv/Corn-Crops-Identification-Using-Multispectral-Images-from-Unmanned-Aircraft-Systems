"""
Adaptate from https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
Calculate Mean and Std per channel
in this script, we calculate the image per channel mean and standard
deviation in the training set, do not calculate the statistics on the
whole dataset, as per here http://cs231n.github.io/neural-networks-2/#datapre
"""

import numpy as np
from os import listdir
from os.path import join, isdir
import glob 
import cv2
import timeit
from pathlib import Path

# number of channels of the dataset image, 4 channels (RGB NIR)


def find_max(im_pths):

    #get_files_path = str(data_root) + "/*.npy"
    #im_pths = np.array(sorted(glob.glob(get_files_path)))
    
    minimo_pixel=[]
    maximo_pixel=[]
    #mean_pixel= []
    size=len(im_pths)
    #print('fid_max_size',size)

    for i in im_pths:
        img = np.load(str(i))
        #print(np.max(img))
        img=img.transpose((2, 1, 0))
        #print(np.shape(img))
        minimo_pixel.append(np.min(img))
        maximo_pixel.append(np.max(img))
        #mean_pixel.append(np.mean(img[:,:,0]))
    #print(np.mean(mean_pixel))
    return   np.min(minimo_pixel),np.max(maximo_pixel), size
        

    
CHANNEL_NUM = 4


def cal_dir_stat(im_pths, maximunValue): ##give the names 
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    #get_files_path = str(data_root) + "/*.npy"
    #im_pths = np.array(sorted(glob.glob(get_files_path)))
    
    #print('cal_shape',np.shape(im_pths))


    for path in im_pths:
        im = np.load(path).transpose(1,2,0) # 
        #print(np.shape(im))

        im = im/maximunValue
        pixel_num += (im.size/CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    rgb_mean = channel_sum / pixel_num
    rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))
       
    return rgb_mean, rgb_std


def meanstd(train_root,val_root, test_root,rootdata='data_HR'): #name_file,
    data_path = Path(rootdata)
    #train_root= str(data_path/'train{}'/'images').format(name_file)
    #val_root= str(data_path/'val{}'/'images').format(name_file)
 #************************************************************ 
    #test_raiz= str(data_path/'test{}'/'images').format(name_file)
    #test_raiz= str(data_path/'test_HR_916'/'images')

    #get_files_path = str(test_raiz) + "/*.npy"
    #test_root = np.array(sorted(glob.glob(get_files_path)))
 #************************************************************ 
    
    data_all_raiz=str(data_path/'data'/'images')  #'/home/jgonzalez/Test_2019/Test_PreProcessing/data' #all the dataset
    get_files_path = str(data_all_raiz) + "/*.npy"
    data_all_root = np.array(sorted(glob.glob(get_files_path)))
    
    minimo_pixel_train,maximo_pixel_train,size_train = find_max(train_root)
    minimo_pixel_val,maximo_pixel_val,size_val = find_max(val_root)
    minimo_pixel_test,maximo_pixel_test,size_test = find_max(test_root)
    
    
    minimo_pixel_all,maximo_pixel_all,size_all = find_max(data_all_root)


    start = timeit.default_timer()
    mean_train, std_train = cal_dir_stat(train_root,maximo_pixel_all) #max 3521
    mean_val, std_val = cal_dir_stat(val_root,maximo_pixel_all)
    mean_test, std_test = cal_dir_stat(test_root,maximo_pixel_all) 
    mean_all, std_all = cal_dir_stat(data_all_root,maximo_pixel_all)

    end = timeit.default_timer()
    print("elapsed time: {}".format(end-start))
    print('Train:',str(data_path),size_train, 'min ',np.min(minimo_pixel_train),'max ',maximo_pixel_train) # 0-1
    print("mean:{}\nstd:{}".format(mean_train, std_train))

    print('Val:',str(data_path),size_val,'min ',np.min(minimo_pixel_val),'max ',maximo_pixel_val) # 0-1
    print("mean:{}\nstd:{}".format(mean_val, std_val))

    print('Test:',str(data_path),size_test,'-min ',np.min(minimo_pixel_test),'max ',maximo_pixel_test) # 0-1
    print("mean:{}\nstd:{}".format(mean_test, std_test))

        
    print('All:',str(data_path),size_all,'min ',np.min(minimo_pixel_all),'max ',maximo_pixel_all) # 0-1
    
    print("mean:{}\nstd:{}".format(mean_all, std_all))
    return maximo_pixel_all, mean_train, std_train