#!/bin/bash
DATA='./DATA'

DATASET=office_home_dg
D1=art
D2=clipart
D3=product
D4=real_world
SEED=2
method=corruncertaintcheck
cuda_device=1
is_corrb=True

(CUDA_VISIBLE_DEVICES=$cuda_device python tools/train.py \
--root ${DATA} \
--uncertainty 0.5 \
--is_corr $is_corrb \
--trainer Vanilla \
--source-domains ${D1} ${D3} ${D4} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2})
