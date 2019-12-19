#!/bin/ bash
#bash trainHR.sh
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0 #,1,2,3
echo hola

for i in 0 1 2 3 4
do
  for j in 0 1 2 3 4
    do
    python train_HR.py --fold-out $i  --fold-in $j --percent 0.06
    done
done