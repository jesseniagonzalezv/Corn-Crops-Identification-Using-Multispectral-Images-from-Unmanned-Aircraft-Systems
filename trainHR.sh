#!/bin/ bash
#bash trainHR.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo hola

#40epech
python train.py   --model UNet  --n-epochs 40 --lr 1e-4
python plotting.py --out-file '160' --stage 'test' --name-file '_100_percent_160' --name-model 'UNet' --n-epochs 40 --count 2747 

#100epochs batch 8
#python train_HR.py   --model UNet  --n-epochs 60 --lr 1e-5 --batch-size 16
#python plotting.py --out-file 'HR' --stage 'test' --name-file '_100_percent' --name-model 'UNet' --count 2747 

#100epochs batch 16
#python train_HR.py   --model UNet11  --n-epochs 60 --lr 1e-4 --batch-size 16
#python plotting.py --out-file 'HR' --stage 'test' --name-file '_100_percent' --name-model 'UNet11' --count 2747 

#python train_HR.py   --model SegNet  --n-epochs 60 --lr 1e-4 --batch-size 16
#python plotting.py --out-file 'HR' --stage 'test' --name-file '_100_percent' --name-model 'SegNet' --count 2747 

#SOLO RGB
#python train_HR.py   --model UNet  --n-epochs 1 --lr 1e-4 --batch-size 16
#python plotting.py --out-file 'HR' --stage 'test' --name-file '_100_percent' --name-model 'UNet' --count 2747 
