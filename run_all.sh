MCMT_CONFIG_FILE="aic_all.yml"
#### Run Detector.####
# 视频转成图片
cd detector
python gen_images_aic.py ${MCMT_CONFIG_FILE}

# 得到检测框
cd yolov5
sh gen_det.sh ${MCMT_CONFIG_FILE}

#### Extract reid feautres.####
# 对每个检测框的图片进行编码
cd ../../reid/
python3 extract_image_feat.py "aic_reid1.yml"
python3 extract_image_feat.py "aic_reid2.yml"
python3 extract_image_feat.py "aic_reid3.yml"
python3 merge_reid_feat.py ${MCMT_CONFIG_FILE}

#### MOT. ####
# 单视频目标追踪
cd ../tracker/MOTBaseline
sh run_aic.sh ${MCMT_CONFIG_FILE}
wait
#### Get results. ####
cd ../../reid/reid-matching/tools
python trajectory_fusion.py ${MCMT_CONFIG_FILE}
python sub_cluster.py ${MCMT_CONFIG_FILE}
python gen_res.py ${MCMT_CONFIG_FILE}

#### Vis. (optional) ####
# python viz_mot.py ${MCMT_CONFIG_FILE}
# python viz_mcmt.py ${MCMT_CONFIG_FILE}
