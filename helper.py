import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import itertools
import torchvision.utils

def values_metric(filedata,name_metric):
    m_metric = []
    for i in filedata:
        i = i.strip(" ")
        if str(i).startswith(name_metric):
            i = i.split(" ")
            m_metric.append(float(i[1]))
    return m_metric


def plot_img_array(img_array, filedata,save,out_file, name_output, ncol=3):
    
    m_jaccard=values_metric(filedata,"jaccard")
    m_dice=values_metric(filedata,"dice")
    #print(len(m_jaccard))

    nrow = len(img_array)  // ncol    
    plt.close('all')
    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4)) 
    #f.suptitle('----Image---------Label---------Predictions---------')

    count = 0
    for i in range(len(img_array)):

        plots[i // ncol, i % ncol]
        if (i % ncol == 1):
            temp = str(m_dice[count])
            temp1 = str(m_jaccard[count])
            #print(temp,count)
            count += 1
            plots[i // ncol, i % ncol].set_title("Nº:"+str(count) + " dice: " + temp +" IoU: " + temp1)
        plots[i // ncol, i % ncol].imshow(img_array[i])
    if(save==1):
        f.savefig(("predictions/{}/prediction_{}.pdf").format(out_file,name_output), bbox_inches='tight') #last same this


def plot_side_by_side(img_arrays,filedata,out_file, name_output,save=0):
    #print('ini',len(img_arrays))
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))
    #print('flat',len(img_arrays))

    plot_img_array(np.array(flatten_list),filedata,save,out_file, name_output, ncol=len(img_arrays))

def masks_to_colorimg_3clases(masks):

    colors = np.asarray([(240, 0, 0), (0, 240, 0), (0, 0, 240)])
    colorimg = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.2]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)


def masks_to_colorimg(masks):

    colors = np.asarray([(0, 140, 150)])
    colorimg = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)

def reverse_transform(inp,out_file='512'):
    inp = inp.transpose(1,2,0)
    if out_file=='512':                   
        mean = np.array([0.22651606, 0.28676218, 0.23280623 ])
        std = np.array([0.11427596, 0.1197355,  0.12576549] )  
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)*65535 #
        inp = (inp/inp.max()*255).astype(np.uint8)

    else:
        mean = np.array([0.23026181, 0.29209857, 0.23458897])
        std = np.array([0.1102557,  0.11778646, 0.12171536] ) 
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)*65535 #
        inp = (inp/inp.max())
        inp = (inp*255).astype(np.uint8)

    return inp
