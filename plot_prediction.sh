#!/bin/ bash
#Run as bash plot_prediction.sh

echo ploting

for i in 0 #1 2 3 4
do
    for j in 4 #0 1 2 3 4
       do
         python plotting.py --out-file 'HR' --stage 'test' --name-file '_8_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94
         #python plotting.py --out-file 'paral' --stage 'test' --name-file '_8_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94
         #python plotting.py --out-file 'HR' --stage 'test' --name-file '_20_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94
         #python plotting.py --out-file 'paral' --stage 'test' --name-file '_20_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94
         #python plotting.py --out-file 'HR' --stage 'test' --name-file '_40_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94
         #python plotting.py --out-file 'paral' --stage 'test' --name-file '_40_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94
         #python plotting.py --out-file 'HR' --stage 'test' --name-file '_80_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94
         #python plotting.py --out-file 'paral' --stage 'test' --name-file '_80_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94
         #python plotting.py --out-file 'HR' --stage 'test' --name-file '_100_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94
         #python plotting.py --out-file 'paral' --stage 'test' --name-file '_100_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94
    done
done
