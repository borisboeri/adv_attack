#!/bin/sh

python test_input.py \
    --train_log_dir ./log \
    --data_dir /mnt/dataX/assia/ \
    --data cifar \
    --h 28 \
    --w 28 \
    --batch_size 8 \
    --xp_name testinput \


