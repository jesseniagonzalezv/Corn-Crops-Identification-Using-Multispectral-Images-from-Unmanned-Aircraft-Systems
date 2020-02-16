#!/bin/ bash
#bash trainHR.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo hola


python train_HR.py   --model UNet  --n-epochs 40 --lr 1e-4

python plotting.py --out-file 'HR' --stage 'test' --name-file '_100_percent' --name-model 'UNet' --count 2747 
