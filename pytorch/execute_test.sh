#!/bin/bash
source activate py35
for i in {1..5000}
do
   python deploy.py --model_path unet_cars_1_13.pkl --dataset cars --img_path $i --out_path /home/user/carseg/carsegmentation/pytorch-semseg-master/out
done
