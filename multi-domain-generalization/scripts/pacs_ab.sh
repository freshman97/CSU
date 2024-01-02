#!/bin/bash
DATA=./DATA
DATASET=pacs_ab
method_list=(uresnet18_a01d3 uresnet18_a12d3 uresnet18_a23d3 uresnet18_a34d3 uresnet18_a45d3 \
             uresnet18_a012d3 uresnet18_a123d3 uresnet18_a234d3 uresnet18_a345d3 \
             uresnet18_a0123d3 uresnet18_a1234d3 uresnet18_a2345d3 \
             uresnet18_a01234d3 uresnet18_a12345d3 \
             uresnet18_a012345d3 \
             uresnet18_a012345d1 uresnet18_a012345d2 \
             uresnet18_a012345d4 uresnet18_a012345d5 uresnet18_a012345d7 uresnet18_a012345d9)


#method_list=(uresnet18_a012345d1 uresnet18_a012345d2 uresnet18_a012345d3 \
#             uresnet18_a012345d4 uresnet18_a012345d5 uresnet18_a012345d7 uresnet18_a012345d9)
D1=art_painting
D2=cartoon
D3=photo
D4=sketch
SEED=11
prexis=corruncertainty
cuda_device=1
int=0

while (( $int < 31))
do
    wait
    method=${method_list[$int]}

    (CUDA_VISIBLE_DEVICES=2 python tools/train.py \
    --root ${DATA} \
    --trainer Vanilla \
    --backbone ${method} \
    --uncertainty 0.5 \
    --source-domains ${D2} ${D3} ${D4} \
    --target-domains ${D1} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/dg/${DATASET}/"${prexis}_${method}"/${D1} &)

    (CUDA_VISIBLE_DEVICES=2 python tools/train.py \
    --root ${DATA} \
    --trainer Vanilla \
    --uncertainty 0.5 \
    --backbone ${method} \
    --source-domains ${D1} ${D3} ${D4} \
    --target-domains ${D2} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/dg/${DATASET}/"${prexis}_${method}"/${D2} &)

    (CUDA_VISIBLE_DEVICES=3 python tools/train.py \
    --root ${DATA} \
    --trainer Vanilla \
    --uncertainty 0.5 \
    --backbone ${method} \
    --source-domains ${D1} ${D2} ${D4} \
    --target-domains ${D3} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/dg/${DATASET}/"${prexis}_${method}"/${D3} &)

    (CUDA_VISIBLE_DEVICES=3 python tools/train.py \
    --root ${DATA} \
    --trainer Vanilla \
    --uncertainty 0.5 \
    --backbone ${method} \
    --source-domains ${D1} ${D2} ${D3} \
    --target-domains ${D4} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/dg/${DATASET}/"${prexis}_${method}"/${D4})

    let "int++"
    echo "Finished ${method}"
    wait
done

echo "Running scripts in parallel"
echo "Script done running"