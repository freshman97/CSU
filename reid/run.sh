#!/bin/bash

DATA=/home/zze3980/project/UncertaintyGeneral/Mixstyle/mixstyle-release/reid/data
model1=resnet50_fc512_csu012345_a0d3
model2=osnet_x1_0_csu234_a0d3
###########
# resnet50
###########
CUDA_VISIBLE_DEVICES=4 python main.py \
--config-file cfgs/cfg_r50.yaml \
-s market1501 \
-t dukemtmcreid \
--root ${DATA} \
model.name ${model1} \
data.save_dir output/${model1}/market2duke &

CUDA_VISIBLE_DEVICES=5 python main.py \
--config-file cfgs/cfg_r50.yaml \
-s dukemtmcreid \
-t market1501 \
--root ${DATA} \
model.name ${model1} \
data.save_dir output/${model1}/duke2market &

###########
# osnet
###########
CUDA_VISIBLE_DEVICES=6 python main.py \
--config-file cfgs/cfg_osnet.yaml \
-s market1501 \
-t dukemtmcreid \
--root ${DATA} \
model.name ${model2} \
data.save_dir output/${model2}/market2duke &

CUDA_VISIBLE_DEVICES=7 python main.py \
--config-file cfgs/cfg_osnet.yaml \
-s dukemtmcreid \
-t market1501 \
--root ${DATA} \
model.name ${model2} \
data.save_dir output/${model2}/duke2market &