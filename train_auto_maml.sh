#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3, python -u auto_maml.py  --update_step 5  --update_step_test 10 --batch_size 5000  --layers 2 --meta_batch_size 4  --k_spt 1
