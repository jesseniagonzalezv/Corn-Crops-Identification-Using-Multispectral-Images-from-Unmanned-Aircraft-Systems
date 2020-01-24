#!/bin/ bash
#Run as bash plot_prediction.sh

echo ploting

python plotting.py --out-file 'HR' --stage 'test'  --name-model 'UNet11' --count 277
