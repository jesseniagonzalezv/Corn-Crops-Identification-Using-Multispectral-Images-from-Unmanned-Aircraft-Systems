import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import glob
import helper
import argparse


def read_metric(out_file,stage,name_file, name,name_model,fold_out,fold_in,epochs):
    loss_file = open(("predictions/{}/pred_loss_{}{}_{}_foldout{}_foldin{}_{}epochs.txt").format(out_file,stage,name_file,name_model,fold_out,fold_in,epochs))
    filedata = loss_file.read()
    filedata = filedata.replace("bce",",bce")
    filedata = filedata.split(",")
    metric=[]
    for i in filedata:
        i = i.strip(" ")
        if str(i).startswith(name):
            i = i.split(" ")
            metric.append(float(i[1]))
            
    plt.close('all')
    f = plt.figure()
    y_axe = np.asarray(metric)
    
    x =np.asarray(list(range(0,len(metric))))
    plt.xlabel(("Number of {} images").format(stage))
    plt.ylabel(name)
    plt.title(("predictions_with_{}").format(name) )
    plt.plot(x,y_axe,label = name_file)
    plt.show()
    f.savefig(("predictions/{}/metric_{}_{}{}_{}_foldout{}_foldin{}_{}epochs.pdf").format(out_file,name,stage,name_file,name_model,fold_out,fold_in,epochs), bbox_inches='tight')
    print('Ctd:',len(x),name,np.mean(metric))
    plt.close()

    #return metric


def plot_history_train(out_file,name_file,name_model,fold_out,fold_in,epochs):
    file = open(("history/{}/history_model{}_{}_foldout{}_foldin{}_{}epochs.txt").format(out_file,name_file,name_model,fold_out,fold_in,epochs), "r")
    

        
    filedata = file.read() 
    filedata = filedata.replace("dataloader",",dataloader")
    filedata = filedata.replace("saving",",saving")
    filedata = filedata.replace("\n",", \n")



    filedata = filedata.split(",")
    loss = []
    for i in filedata:
        i = i.strip(" ")

        if str(i).startswith("loss:"):
            i = i.split(":")
            loss.append(float(i[1]))
            
    plt.close('all')
    f= plt.figure()  
    y_train = np.asarray(loss[0::2])
    y_val = np.asarray(loss[1::2])

    x =np.asarray(list(range(0,len(y_val))))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("loss graph")
    plt.plot(x[1:120:],y_train[1:120:],label = 'train_loss') 
    plt.plot(x[1:120:],y_val[1:120:],'k', label = 'val_loss')
    plt.legend()
    plt.show()
    f.savefig(("history/{}/loss_convergence{}_{}_foldout{}_foldin{}_{}epochs.pdf").format(out_file,name_file,name_model,fold_out,fold_in,epochs), bbox_inches='tight') 
    plt.close()
 #  


def plot_prediction(stage='test',name_file='_VHR_60_fake',out_file='VHR',name_model='UNet11',fold_out=0,fold_in=0,epochs=40, count=30): # #HR •dist
    loss_file = open(("predictions/{}/pred_loss_{}{}_{}_foldout{}_foldin{}_{}epochs.txt").format(out_file,stage,name_file,name_model,fold_out,fold_in,epochs))
  
    filedata = loss_file.read()
    filedata = filedata.replace("bce",",bce")
    filedata = filedata.split(",")

    val_file = (("predictions/{}/inputs_{}{}_{}_foldout{}_foldin{}_{}epochs_{}.npy").format(out_file, stage, name_file,name_model,fold_out,fold_in,epochs, count))
    pred_file =(("predictions/{}/pred_{}{}_{}_foldout{}_foldin{}_{}epochs_{}.npy").format(out_file, stage, name_file,name_model,fold_out,fold_in,epochs, count))
    label_file = (("predictions/{}/labels_{}{}_{}_foldout{}_foldin{}_{}epochs_{}.npy").format(out_file, stage, name_file,name_model,fold_out,fold_in,epochs, count))

    val_images = np.load(val_file)
    pred_images = np.load(pred_file)
    val_label = np.load(label_file)
    print(val_images.shape,val_label.shape,pred_images.shape)
    input_images_rgb = [helper.reverse_transform(x,out_file) for x in val_images[:95,0,:3,:,:]]   #new metrics
    # Map each channel (i.e. class) to each color
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in val_label[:95,0,:3,:,:]]
    pred_rgb = [helper.masks_to_colorimg(x) for x in pred_images[:95,0,:,:,:]]

    name_output=("{}{}_{}_foldout{}_foldin{}_{}epochs").format(stage, name_file,name_model,fold_out,fold_in,epochs)
  
   # stage + name_file + name_model+'_foldin' +str(fold_in)
    helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb],filedata, out_file, name_output, save=1)
    

    
def main():   
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--out-file', type=str, default='160', help='For example 160 or 512')
    arg('--stage', type=str, default='test', help='For example test or val')
    arg('--name-file', type=str, default='_8_percent', help='For example _6_percent')
    arg('--name-model', type=str, default='UNet11', choices=['UNet11','UNet','AlbuNet34','SegNet'])
    arg('--fold-out', type=int, help='fold train test', default=0)
    arg('--fold-in', type=int, help='fold train val', default=0)  
    arg('--n-epochs', type=int, help='epochs in which the model was trained', default=40)  
    arg('--count', type=int, help='number of img to plot', default=94)        
    
    args = parser.parse_args()
    

    plot_history_train(args.out_file, args.name_file,args.name_model,args.fold_out,args.fold_in,args.n_epochs)   
    plot_prediction(args.stage, args.name_file, args.out_file, args.name_model,args.fold_out,args.fold_in,args.n_epochs,args.count) 
 

if __name__ == '__main__':
    main()
