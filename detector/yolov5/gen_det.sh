#!/bin/bash
seqs=(c041 c042 c043 c044 c045 c046)
# seqs=(c041)
# gpu_id=0
for seq in "${seqs[@]}"
do
    python3 detect2img.py --name ${seq} --weights yolov5s.pt --conf 0.1 --agnostic --save-txt --save-conf --img-size 1280 --classes 2 5 7 --cfg_file $1
    # gpu_id=$(($gpu_id+1))
    echo $seq
done
wait


#bash gen_det.sh "aic_all.yml"
# python3 detect2img.py --name c041 --weights yolov5s.pt --conf 0.1 --agnostic --save-txt --save-conf --img-size 1280 --classes 2 5 7 --cfg_file aic_all.yml