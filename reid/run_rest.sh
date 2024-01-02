#!/bin/bash

DATA=/home/zze3980/project/UncertaintyGeneral/Mixstyle/mixstyle-release/reid/data
#model1_list=(osnet_x1_0_csu12_a0d1 osnet_x1_0_csu23_a0d1 osnet_x1_0_csu34_a0d1 \
#             osnet_x1_0_csu123_a0d1 osnet_x1_0_csu234_a0d1)
#model2_list=(osnet_x1_0_dsu osnet_x1_0_pada uresnet50_dsu uresnet50_pada \
#             resnet50_fc512_csu0123_a0d1  osnet_x1_0_csu1234_a0d1)
model1_list=(osnet_x1_0_csu012345_a0d1 osnet_x1_0_csu12345_a0d1 osnet_x1_0_csu1234_a0d1)
model2_list=(resnet50_fc512_csu012345_a0d1 resnet50_fc512_csu12345_a0d1 resnet50_fc512_csu1234_a0d1 resnet50_fc512_csu0123_a0d1)
int=0
while (( $int<4 ))
do
    echo $int
    model1=${model1_list[$int]}
    model2=${model2_list[$int]}
    echo $model1
    echo $model2
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
    data.save_dir output/${model2}/duke2market

    let "int++"
    wait
done
