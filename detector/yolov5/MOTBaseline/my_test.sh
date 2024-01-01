#!/bin/bash

TrackOneSeq(){
    local seq=$1
    local config=$2
    echo "Tracking $seq with ${config}"
    python3 -W ignore fair_app.py \
        --min_confidence=0.1 \
        --display=False \
        --max_frame_idx -1 \
        --nms_max_overlap 0.99 \
        --min-box-area 750 \
        --cfg_file ${config} \
        --seq_name ${seq} \
        --max_cosine_distance 0.5

    echo "fair_app.py done."
    cd ./post_processing
    # 后处理
    python3 main.py ${seq} pp ${config}
    cd ..
}

seqs=("c041" "c042" "c043" "c044" "c045" "c046")
# seqs=("c042")
config_file="aic_all.yml"

for seq in "${seqs[@]}"
do 
    echo "Processing sequence: ${seq}"
    TrackOneSeq "${seq}" "${config_file}"
done
wait
