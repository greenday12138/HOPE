import os
from utils.datasets import LoadStreams, LoadImages
import cv2
import numpy as np

imgsz = 1280
cams_dir = 'datasets/AIC22_Track1_MTMC_Tracking/test/S06'

def load_video():
    cams = os.listdir(cams_dir)
    datasets, trackers, f_nums = {}, {}, []
    # roi_masks, overlap_regions = {}, {}
    for cam in cams:
        video_dir = os.path.join(cams_dir, cam) + '/vdo.avi'
        datasets[cam] = LoadImages(video_dir, img_size=imgsz, stride=32)
        # print(type(datasets[cam])) <class 'utils.datasets.LoadImages'>
    for cam in cams: 
        for path, img, im0s, vid_cap in datasets[cam]:
            cv2.imshow('Video Stream', im0s)
            cv2.waitKey(0)  # 等待用户按下键盘上的任意键
            break

def gather_sequence_info(sequence_dir):
    image_dir = os.path.join(sequence_dir, "img1")
    # image_filenames = {index : 图片名}
    image_filenames = {
            int(os.path.splitext(f)[0][3:]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')
    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None
    cv2.imshow("x",image)
    cv2.waitKey(0)
    print(image_size)

a = '/mnt/c/Users/83725/Desktop/AIC21-MTMC/datasets/detection/images/test/S06/c041'
gather_sequence_info(a)
