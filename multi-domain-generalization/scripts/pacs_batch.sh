#!/bin/bash
DATA='./DATA'

DATASET=pacs_batch
method=uresnet18_a012345d3
batch_list=(16 32 64 128 256 512)

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


while (( $int < 6))
do
    wait
    batch=${batch_list[$int]}

    (CUDA_VISIBLE_DEVICES=3 python tools/train.py \
    --root ${DATA} \
    --trainer Vanilla \
    --backbone ${method} \
    --batch_size ${batch} \
    --uncertainty 0.5 \
    --source-domains ${D2} ${D3} ${D4} \
    --target-domains ${D1} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/dg/${DATASET}/"${prexis}_${method}_${batch}"/${D1} &)

    (CUDA_VISIBLE_DEVICES=4 python tools/train.py \
    --root ${DATA} \
    --trainer Vanilla \
    --uncertainty 0.5 \
    --backbone ${method} \
    --batch_size ${batch} \
    --source-domains ${D1} ${D3} ${D4} \
    --target-domains ${D2} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/dg/${DATASET}/"${prexis}_${method}_${batch}"/${D2} )
    wait

    (CUDA_VISIBLE_DEVICES=3 python tools/train.py \
    --root ${DATA} \
    --trainer Vanilla \
    --uncertainty 0.5 \
    --backbone ${method} \
    --batch_size ${batch} \
    --source-domains ${D1} ${D2} ${D4} \
    --target-domains ${D3} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/dg/${DATASET}/"${prexis}_${method}_${batch}"/${D3} &)

    (CUDA_VISIBLE_DEVICES=4 python tools/train.py \
    --root ${DATA} \
    --trainer Vanilla \
    --uncertainty 0.5 \
    --backbone ${method} \
    --batch_size ${batch} \
    --source-domains ${D1} ${D2} ${D3} \
    --target-domains ${D4} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/dg/${DATASET}/"${prexis}_${method}_${batch}"/${D4})

    let "int++"
    echo "Finished ${method}"
    wait
done

echo "Running scripts in parallel"
echo "Script done running"