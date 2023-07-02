#!/bin/bash

# 如何运行：
# chmod +x run.sh 
# ./run.sh

# first_epoch=85
# continue_times=20
# step=5

# echo "first_epoch: $first_epoch";
# python train_IEMOCAP.py --GAN-epochs=$first_epoch --continue-train-GAN-step=0;

# for((i=1;i<=continue_times;i++)); 
# do
#     echo "continue_times: {$first_epoch + $i * $step}}";
#     current_epoch=$(($first_epoch + $i * $step));
#     python train_IEMOCAP.py --GAN-epochs=$current_epoch --use-trained-GAN --continue-train-GAN-step=$step;
# done

python train_IEMOCAP.py --GAN-epochs=120 --continue-train-GAN-step=0;
python train_IEMOCAP.py --GAN-epochs=150 --continue-train-GAN-step=0;
