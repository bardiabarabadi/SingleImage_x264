#!/bin/bash


for QP in QP30 QP38 QP42 QP46
do
    echo $QP
    python train.py --LR_dir ./RAW/train/ --NR_dir ./${QP}/train/ --model_save_dir ./model/${QP}/ 
    
done


