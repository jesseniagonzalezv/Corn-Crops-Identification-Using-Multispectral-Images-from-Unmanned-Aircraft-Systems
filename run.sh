#!/bin/ bash
#bash trainHR.sh
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0,1,2,3
echo hola

python train.py   --model UNet11  --n-epochs 30 --lr 1e-5 --batch-size 4 #--channels 0,1,2,3,4

python train.py   --model UNet11  --n-epochs 31 --lr 1e-5 --batch-size 4 --channels 3,4,2

python train.py   --model UNet11  --n-epochs 29 --lr 1e-5 --batch-size 4 --channels 0,1,2

python train.py   --model UNet11  --n-epochs 32 --lr 1e-5 --batch-size 4 --channels 1,3,4


python plotting.py --out-file '512' --stage 'test' --name-file '_100_percent_512' --name-model 'UNet' --count 227 --n-epochs  30

python plotting.py --out-file '512' --stage 'test' --name-file '_100_percent_512' --name-model 'UNet' --count 227 --n-epochs  31

python plotting.py --out-file '512' --stage 'test' --name-file '_100_percent_512' --name-model 'UNet' --count 227 --n-epochs  29

python plotting.py --out-file '512' --stage 'test' --name-file '_100_percent_512' --name-model 'UNet' --count 227 --n-epochs  32