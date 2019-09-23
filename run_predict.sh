#!/bin/sh
#
# PROGRAMMER: Brian Pederson
# DATE CREATED: 05/20/2018
# REVISED DATE: 05/20/2018
# PURPOSE: Runs various variants of predict.py
#
# Usage: sh run_predict_batch.sh    -- will run program from commandline within Project Workspace
#
python predict.py flowers/test/28/image_05230.jpg  checkpoint_densenet121_n1.pth  --gpu
python predict.py flowers/test/59/image_05020.jpg  checkpoint_densenet121_n1.pth  --gpu
python predict.py flowers/test/6/image_07182.jpg  checkpoint_densenet121_n1.pth  --gpu
python predict.py flowers/test/6/image_07181.jpg  checkpoint_densenet121_n1.pth  --gpu

