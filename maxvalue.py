import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# read all bands
def maxvalue(root):
    datasetRGBNIR = rasterio.open(root)
    print(datasetRGBNIR)
    datasetRGBNIR.indexes

    array = datasetRGBNIR.read()


    stats = []
    for band in array:
        stats.append({
            'min': band.min(),
            'mean': band.mean(),
            'median': np.median(band),
            'max': band.max()})

    # Mostrar stado de cada banda
    #print(stats)

    maxValue=np.max(array)
    #print('type:',array.dtype,'max', maxValue)
    #print('shape',array.shape)
    return maxValue
    
    
in_path = Path('imagenes')

input_filename0  = 'imagen0/IMG_PER1_20170422154946_ORT_MS_003749.TIF' #ok
input_filename1  = 'imagen1/IMG_PER1_20170422154946_ORT_MS_003131.TIF'
input_filename2  = 'imagen2/IMG_PER1_20170422154946_ORT_MS_002513.TIF'
input_filename3  = 'imagen3/IMG_PER1_20190703144250_ORT_MS_000672.TIF'
input_filename4  = 'imagen4/IMG_PER1_20190703144250_ORT_MS_001290.TIF'
input_filename5 = 'imagen5/IMG_PER1_20190703144250_ORT_MS_002526.TIF'
input_filename6 = 'imagen6_conida/IMG_PER1_20170410154322_ORT_MS_000659.TIF'

array_max=[]

root0=str(os.path.join(in_path, input_filename0))
max0= maxvalue(root0)
array_max.append(max0)

root1=str(os.path.join(in_path, input_filename1))
max1= maxvalue(root1)
array_max.append(max1)

root2=str(os.path.join(in_path, input_filename2))
max2= maxvalue(root2)
array_max.append(max2)

root3=str(os.path.join(in_path, input_filename3))
max3= maxvalue(root3)
array_max.append(max3)

root4=str(os.path.join(in_path, input_filename4))
max4= maxvalue(root4)
array_max.append(max4)

root5=str(os.path.join(in_path, input_filename5))
max5= maxvalue(root5)
array_max.append(max5)


root6=str(os.path.join(in_path, input_filename6))
max6= maxvalue(root6)
array_max.append(max6)

print('array', array_max, 'max', np.max(array_max))
