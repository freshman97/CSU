#!/bin/bash
DATA=./DATA

DATASET=office_home_ab
D1=art
D2=clipart
D3=product
D4=real_world
SEED=2
model=uresnet18_compare
method=corruncertainty_compare
cuda_device=5

(CUDA_VISIBLE_DEVICES=$cuda_device python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--backbone ${model} \
--source-domains ${D2} ${D3} ${D4} \
--target-domains ${D1} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1} &)

(CUDA_VISIBLE_DEVICES=$cuda_device python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--backbone ${model} \
--source-domains ${D1} ${D3} ${D4} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} &)

(CUDA_VISIBLE_DEVICES=$cuda_device python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--backbone ${model} \
--source-domains ${D1} ${D2} ${D4} \
--target-domains ${D3} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D3} &)

(CUDA_VISIBLE_DEVICES=$cuda_device python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--backbone ${model} \
--source-domains ${D1} ${D2} ${D3} \
--target-domains ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D4} &)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
