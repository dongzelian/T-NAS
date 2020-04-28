#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2, python -u train_auto_maml_miniimagenet.py  --update_step 5  --update_step_test 10 --layers 2 --meta_batch_size 4 --batch_size 15000  --epoch 15 --arch AUTO_MAML_2  --k_spt 1
#CUDA_VISIBLE_DEVICES=1, python -u train_auto_maml_miniimagenet.py  --update_step 5  --update_step_test 10 --layers 2 --meta_batch_size 4 --batch_size 15000  --epoch 15 --arch AUTO_MAML_3  --k_spt 5
