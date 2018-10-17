#!/bin/sh

LOG_ROOT_DIR=log/

if [ "$1" = '-h' ]; then 
  echo "Usage"
  echo "  1. Xp source"
  echo "  2. Xp name"
  echo "  3. Data"
  echo "  4. Number of epochs to train"
  echo "  5. Number of epochs between tests"
  echo "  6. Model"
  exit 0
fi

if [ "$#" -ne 6 ]; then
  echo "Error: bad number of arguments"
  echo "Usage"
  echo "  1. Xp source"
  echo "  2. Xp name"
  echo "  3. Data"
  echo "  4. Number of epochs to train"
  echo "  5. Number of epochs between tests"
  echo "  6. Model"
  exit 1
fi

xp_source="$1"
xp_name="$2"
data="$3"
max_train_epoch="$4"
eval_interval_epoch="$5"
num_iter=$((max_train_epoch/eval_interval_epoch))
model="$6"

echo "Experiment recap:"
echo "Train for "$max_train_epoch" epochs"
echo "Test every "$eval_interval_epoch" epochs"
# Create log directories
source_log_dir=""$LOG_ROOT_DIR""$xp_source"/train/"
train_log_dir=""$LOG_ROOT_DIR""$xp_name"/train/"
val_log_dir=""$LOG_ROOT_DIR""$xp_name"/val/"
echo ""$train_log_dir""

if [ -d "$train_log_dir" ]; then
    while true; do
        read -p ""$train_log_dir" already exists. Do you want to overwrite it (y/n) ?" yn
        case $yn in
            [Yy]* ) rm -rf "$train_log_dir"; mkdir -p "$train_log_dir"; break;;
            [Nn]* ) exit;;
            * ) * echo "Please answer yes or no.";;
        esac
    done
else
    mkdir -p "$train_log_dir"
fi

if [ -d "$val_log_dir" ]; then
    while true; do
        read -p ""$val_log_dir" already exists. Do you want to overwrite it (y/n) ?" yn
        case $yn in
            [Yy]* ) rm -rf "$val_log_dir"; mkdir -p "$val_log_dir"; break;;
            [Nn]* ) exit;;
            * ) * echo "Please answer yes or no.";;
        esac
    done
else
    mkdir -p "$val_log_dir"
fi

# Write the xp info
#echo "$max_train_epoch" >> "$train_log_dir"/xp.log
#echo "$eval_interval_epoch" >> "$train_log_dir"/xp.log
#echo "$dataset_id" >> "$train_log_dir"/xp.log

# Run the bench
i=0
echo "num_iter = "$num_iter""
while [ "$i" -lt "$num_iter" ]
do
    python test_restore.py \
        --source_log_dir ./log/ \
        --xp_source "$xp_source" \
        --train_log_dir ./log/ \
        --xp_name "$xp_name" \
        --display_interval 10 \
        --summary_interval 10 \
        --data_dir /mnt/dataX/assia/ \
        --data "$data" \
        --h 28 \
        --w 28 \
        --train_set_size 60000 \
        --test_set_size 10000 \
        --model "$model" \
        --epochs "$eval_interval_epoch" \
        --batch_size 16 \
        --lr 1e-4 \
        --adam_b1 0.9 \
        --adam_b2 0.999 \
        --adam_eps 1e-08 \
        --moving_average_decay 0.9999 \
        --start "$i"

    if [ $? -ne 0 ]; then
        echo "Error in training "$i" "
        exit 1
    fi

#    python eval.py \
#        --train_log_dir ./log \
#        --xp_name "$xp_name" \
#        --model "$model" \
#        --data_dir /mnt/dataX/assia/ \
#        --data "$data" \
#        --h 28 \
#        --w 28 \
#        --train_set_size 50000 \
#        --test_set_size 10000 \
#        --batch_size 10 \
#        --moving_average_decay 0.9999 
#    if [ $? -ne 0 ]; then
#        echo "Error in eval "$i" "
#        exit 1
#    fi

    i=$((i+1))
done


