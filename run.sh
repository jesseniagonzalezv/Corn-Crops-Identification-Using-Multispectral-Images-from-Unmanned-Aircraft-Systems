#!/bin/ bash
#bash trainHR.sh
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0,1,2,3
echo hola

for i in 0 
do
  for j in 4
    do
	python train.py   --model UNet  --n-epochs 100 --lr 1e-6 --batch-size 4 --fold-out $i  --fold-in $j  #--channels 0,1,2,3,4
	python plotting.py --out-file '512' --stage 'test' --name-file '_100_percent_512' --name-model 'UNet' --count 227 --n-epochs  100 --fold-out $i  --fold-in $j 


    done
done


