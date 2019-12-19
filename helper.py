import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import itertools
import torchvision.utils




def values_jaccard(filedata):
    m_jaccard = []
    for i in filedata:
        i = i.strip(" ")
        if str(i).startswith("jaccard"):
            i = i.split(" ")
            m_jaccard.append(float(i[1]))
    return m_jaccard

def values_dice(filedata):
    m_dice = []
    for i in filedata:
        i = i.strip(" ")
        if str(i).startswith("dice"):
            i = i.split(" ")
            m_dice.append(float(i[1]))
    return m_dice


def reverse_transform2(inp): 
    inp = inp.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.11239524, 0.101936, 0.11311523])
    std = np.array([0.08964322, 0.06702993, 0.05725554]) 
 
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)* 3521#3415  #check value maximun
    inp = (inp/inp.max()).astype(np.float32)  

    return inp

def plot_img_array(img_array, filedata,save,out_file, name_output, ncol=3):
    
    m_jaccard=values_jaccard(filedata)
    m_dice=values_dice(filedata)
    #print(len(m_jaccard))

    nrow = len(img_array)  // ncol
    
    plt.close('all')

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))    
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
        f.savefig(("predictions_{}/prediction_{}.pdf").format(out_file,name_output), bbox_inches='tight') #last same this


def plot_side_by_side(img_arrays,filedata,out_file, name_output,save=0):
    #print('ini',len(img_arrays))
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))
    #print('flat',len(img_arrays))

    plot_img_array(np.array(flatten_list),filedata,save,out_file, name_output, ncol=len(img_arrays))


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

def reverse_transform(inp):     
    inp = inp.transpose(1,2,0)
    mean = np.array([0.11239524, 0.101936, 0.11311523])
    std = np.array([0.08964322, 0.06702993, 0.05725554]) 
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)*3521 #3415
    #inp = (inp/inp.max()).astype(np.float32)
    inp = (inp/inp.max())#.astype(np.float32)
    inp = (inp*255).astype(np.uint8)
    return inp
