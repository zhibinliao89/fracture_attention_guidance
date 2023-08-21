#!/usr/bin/env bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
PROJECT_ROOT=$(pwd)/..
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT}

mkdir -p "${PROJECT_ROOT}"/data
mkdir -p "${PROJECT_ROOT}"/log

# general parameters
batch_size=32
num_epochs=95
lr=1e-3
network='resnet50'
seed=0
num_training_images_per_class=5
mkdir -p "${PROJECT_ROOT}"/log/n${num_training_images_per_class}

# model specific parameters
enable_attn_loss=False
attn_loss_scalar=0
identifier=cub_original_${network}_lr${lr}_b${batch_size}_ls${attn_loss_scalar}_s${seed}
echo ${identifier}
python3 -u train.py --num_workers 8 \
                    --save_interval 30 \
                    --seed "${seed}" \
                    --network ${network} \
                    --batch_size ${batch_size} \
                    --save_path "${PROJECT_ROOT}"/data/n${num_training_images_per_class}/${identifier} \
                    --num_epochs ${num_epochs} \
                    --learning_rate ${lr} \
                    --optimizer sgd \
                    --reload_from_checkpoint False \
                    --enable_attn_loss ${enable_attn_loss} \
                    --num_training_images_per_class ${num_training_images_per_class} \
                    --attn_loss_scalar ${attn_loss_scalar} 2>&1 | tee "${PROJECT_ROOT}"/log/n${num_training_images_per_class}/${identifier}.txt
