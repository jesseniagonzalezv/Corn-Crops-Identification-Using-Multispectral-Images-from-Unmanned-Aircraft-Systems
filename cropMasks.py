#Cortar las imagenes TIF RGBNIR
import os
from itertools import product
import rasterio as rio
from rasterio import windows
from pathlib import Path
import csv
import numpy as np

def splits_masks(out_path,input_filename,output_filename,output_filename_npy,index_imgs):

    coordinates='{}-{},{}-{}'

    def get_tiles2(ds, width=512, height=512):
        nols, nrows = ds.meta['width'], ds.meta['height']
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in  offsets:
            window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
            transform = windows.transform(window, ds.transform)  #partir
            yield window, transform


    with rio.open(input_filename) as inds:
        tile_width, tile_height = 512, 512
        meta = inds.meta.copy()


        for window, transform in get_tiles2(inds):

            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            
            outpath = os.path.join(out_path,output_filename.format(index_imgs,int(window.col_off), int(window.row_off)))
            outpath_npy = str(os.path.join(out_path,output_filename_npy.format(index_imgs,int(window.col_off), int(window.row_off))))

            
            if((int(window.width)==512) and (int(window.height)==512)):
                with rio.open(outpath, 'w', **meta) as outds:                
                    array=inds.read(window=window)
                    outds.write(array)     
                    #print(array.shape)
                    np.save(outpath_npy,array)
                                             


                input_id=str(output_filename.format(index_imgs,int(window.col_off), int(window.row_off)))     
                source_id=str(os.path.join(out_path,output_filename.format(index_imgs,int(window.col_off), int(window.row_off))))
                coordinates2= str(coordinates.format(int(window.row_off),int(window.row_off)+int(window.width),int(window.col_off),int(window.col_off)+int(window.height)))

                myData = [[input_id, source_id, coordinates2,]]              
                myFile = open('splits_masks.csv', 'a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(myData)  #list    