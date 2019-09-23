#!/bin/sh
#
# PROGRAMMER: Brian Pederson
# DATE CREATED: 05/20/2018
# REVISED DATE: 05/20/2018  -
# PURPOSE: Runs various variants of train.py
#
# Usage: sh run_train_batch.sh    -- will run program from commandline within Project Workspace
#
#python train.py flowers checkpoint_densenet121_n1.pth --arch densenet121 --hidden_sizes 768 256 128 --epochs 1 --gpu
#python train.py flowers checkpoint_densenet121_n1.pth --epochs 1 --gpu
python train.py flowers checkpoint_densenet121_n1.pth --epochs 1 --gpu

#python train.py flowers checkpoint_vgg16_n1.pth --arch vgg16 --hidden_sizes 2048 512 256 --epochs 1 --gpu

