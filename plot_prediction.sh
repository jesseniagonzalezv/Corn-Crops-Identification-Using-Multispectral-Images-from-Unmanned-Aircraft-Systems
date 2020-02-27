#!/bin/ bash
#Run as bash plot_prediction.sh

echo ploting

#python plotting.py --out-file 'HR' --stage 'test'  --name-model 'UNet' --count 277

python plotting.py --out-file 'HR' --stage 'test' --name-file '_100_percent' --name-model 'UNet' --count 277 
