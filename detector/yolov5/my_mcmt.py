import argparse
import time
from pathlib import Path
import numpy as np
import pickle
import os
import logging
import sys
sys.path.append(os.getcwd())
# print(os.getcwd())
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from numpy import random
from PIL import Image
from config import cfg
from yacs.config import CfgNode

from torchvision.ops import nms
from MOTBaseline.src.fm_tracker.multitracker import JDETracker
from MOTBaseline.src.post_processing.post_association import associate
from MOTBaseline.src.post_processing.track_nms import track_nms
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from reid.matching.tools.utils.zone_intra import zone
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, get_gpu_mem_info, get_cpu_mem_info
from reid.reid_inference.reid_model import build_reid_model

# Namespace(agnostic_nms=True, augment=False, cfg_file='aic_all.yml', 
# classes=[2, 5, 7], conf_thres=0.1, device='', exist_ok=False, img_size=1280, 
# iou_thres=0.45, name='c041', project='/mnt/c/Users/83725/Desktop/AIC21-MTMC/datasets/detect_merge/', 
# save_conf=True, save_txt=True, source='/mnt/c/Users/83725/Desktop/AIC21-MTMC/datasets/detection/images/test/S06//c041/img1/', 
# update=False, view_img=False, weights=['yolov5s.pt'])

formatted_date = time.strftime("%Y-%m-%d", time.localtime())
log_name = f'town5/{formatted_date}_detect_res.log'
logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(message)s')

global GPU_ID
GPU_ID = 0
conf_thres = 0.25
iou_thres = 0.45
vdo_frame_ratio = 10
min_confidence =  0.1 # 置信度
g_tid = 0

# 通过当前轨迹的cam和最后一帧zone_num，找到其应当存在哪个cam的哪个区域的待匹配队列中
next_cams_zone = {'c001':{1:[['c002',1],['c003',1],['c004',1],['c005',1]], 
                          2:[['c005',2]], 
                          3:[], 
                          4:[['c002',3]]},
                  'c002':{1:[['c001',1],['c003',1],['c004',1],['c005',1]], 
                          2:[['c001',4],['c003',2],['c004',2],['c005',3]],
                          3:[['c001',4],['c002',2],['c003',2],['c004',2],['c005',3]], 
                          4:[['c003',3]]},
                  'c003':{1:[['c001',1],['c002',1],['c004',1],['c005',1]], 
                          2:[['c001',4],['c002',2],['c004',2],['c005',3]], 
                          3:[['c002',4]], 
                          4:[['c004',3]]},
                  'c004':{1:[['c001',1],['c002',1],['c003',1],['c005',1]], 
                          2:[['c001',4],['c002',2],['c003',2],['c005',3]], 
                          3:[['c003',4]], 
                          4:[['c005',3]]},
                  'c005':{1:[['c001',1],['c002',1],['c003',1],['c004',1]], 
                          2:[['c001',2]], 
                          3:[['c001',4],['c002',2],['c003',2],['c004',2],['c004',4]], 
                          4:[]}}
# key1摄像头, key2为start_zone, 即该轨迹只能从这个zone邻接的摄像头中寻找匹配轨迹
# 1 白色 2 红色 3 绿色 4 蓝色

# 通过当前轨迹的起始zone, 定位到需要进行跨视频匹配的轨迹列表
# 再基于out_time二分查找，快速定位到最可能匹配的位置，向两侧匹配，直至超过规定值
Track_to_be_matched = {'c001':{1:[], 2:[], 3:[], 4:[]},
                       'c002':{1:[], 2:[], 3:[], 4:[]},
                       'c003':{1:[], 2:[], 3:[], 4:[]},
                       'c004':{1:[], 2:[], 3:[], 4:[]},
                       'c005':{1:[], 2:[], 3:[], 4:[]}}
# c001 4表示 区域4邻接的摄像头到c001的时间差，即c002的区域3的out_time - c001的区域4的in_time
avg_times = {'c001':{1:0.0, 2:0.0, 4:22.8},
             'c002':{1:0.0, 2:0.0, 3:15.0, 4:49.3},
             'c003':{1:0.0, 2:0.0, 3:52.6, 4:27.8},
             'c004':{1:0.0, 2:0.0, 3:50.3, 4:2.8},
             'c005':{1:0.0, 2:0.0, 3:33.4}}

def cfg_extract():
    cfg = CfgNode()
    cfg.REID_MODEL= 'detector/yolov5/reid/reid_model/resnet101_ibn_a_2.pth'
    cfg.REID_BACKBONE= 'resnet101_ibn_a'
    cfg.REID_SIZE_TEST= [384, 384]
    cfg.freeze()
    return cfg

extract_cfg = cfg_extract()

def cal_similarity(vector1,vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def find_closest_element(data, target_time):
    low, high = 0, len(data) - 1
    
    while low <= high:
        mid = (low + high) // 2
        mid_time = data[mid][0]

        if mid_time < target_time:
            low = mid + 1
        elif mid_time > target_time:
            high = mid - 1
        else:
            return mid

    if low > 0 and (high == len(data) - 1 or abs(data[low][0] - target_time) < abs(data[high][0] - target_time)):
        return low
    else:
        return high

def cross_cam_match(cam, start_zone, io_time, new_feat, max_similarity = 0.7):
    global g_tid
    pre_similarity = 0.5
    cross_match = False
    match_list = Track_to_be_matched[cam][start_zone]
    match_list_size = len(match_list)
    if match_list_size == 0:
        g_tid += 1
        matched_tid = g_tid
        return matched_tid
    out_time = io_time[0]-avg_times[cam][start_zone]
    closest_idx = find_closest_element(match_list, out_time)
    closest_idx = max(0, min(closest_idx, match_list_size - 1))
    cur_similarity = cal_similarity(new_feat, match_list[closest_idx][2])
    if cur_similarity > pre_similarity:
        cross_match = True
        pre_similarity = cur_similarity
        matched_tid = match_list[closest_idx][1]
        if pre_similarity > max_similarity:
            return matched_tid
    max_range = max(closest_idx, match_list_size - 1 - closest_idx)

    for search_i in range(1,max_range):
        left, right = closest_idx - search_i, closest_idx + search_i
        left_similarity = cal_similarity(new_feat, match_list[left][2]) if left >= 0 else 0
        if left_similarity > pre_similarity:
            cross_match = True
            pre_similarity = left_similarity
            matched_tid = match_list[left][1]
            if pre_similarity > max_similarity:
                return matched_tid
        right_similarity = cal_similarity(new_feat, match_list[right][2]) if right <= (match_list_size - 1) else 0
        if right_similarity > pre_similarity:
            cross_match = True
            pre_similarity = right_similarity
            matched_tid = match_list[right][1]
            if pre_similarity > max_similarity:
                return matched_tid

    if not cross_match:
        g_tid += 1
        matched_tid = g_tid
    return matched_tid

def read_data_from_txt(file_path):
    data_dict = {}  # 最外层字典，key为tid

    with open(file_path, 'r') as file:
        for line in file:
            # 从每一行提取tid、fid和bbox
            fid, tid, x1,y1,w,h,a,b,c,d = map(int, line.strip().split(','))
            bbox = [x1, y1, w, h]

            # 将数据存储到字典中
            if tid not in data_dict:
                data_dict[tid] = {}  # 内层字典，key为fid
            data_dict[tid][fid] = bbox

    return data_dict

def calculate_iou(bbox1, bbox2):
    # 计算两个矩形框的交并比
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def find_overlap(interval1, interval2):
    start_point = max(interval1[0], interval2[0])
    end_point = min(interval1[1], interval2[1])

    if start_point <= end_point:
        return start_point, end_point
    else:
        return None  # 无重叠部分

def mot_metrics(dict1, dict2, iou_threshold=0.5): # 输入是检测结果和gt
    match_tids = [['ts_tid','gt_tid', 'start_point', 'end_point', 'frame_cnt' 'avg_iou']]
    for tid1 in dict1.keys():
        avg_iou = iou_threshold
        match_tuple = []
        for tid2 in dict2.keys():
            f_list1 = list(dict1[tid1].keys())
            f_list2 = list(dict2[tid2].keys())
            overlap = find_overlap([f_list1[0],f_list1[-1]], [f_list2[0],f_list2[-1]])
            frame_cnt = 0
            if overlap:
                total_iou = 0
                start_point, end_point = overlap
                for i in range(start_point, end_point + 1):
                    if (i in f_list1) and (i in f_list2):
                        total_iou += calculate_iou(dict1[tid1][i], dict2[tid2][i])
                        frame_cnt += 1
                if (total_iou / (end_point - start_point + 1)) > avg_iou:
                    avg_iou = total_iou / frame_cnt
                    match_tuple = [tid1, tid2, start_point, end_point, frame_cnt, round(avg_iou,5)]
        # 找到平均iou最大的
        if len(match_tuple) > 0:
            if (match_tuple[1] == match_tids[-1][1]):
                if (match_tuple[-1] > match_tids[-1][-1]):
                    match_tids.pop()
                    match_tids.append(match_tuple)
            else:
                match_tids.append(match_tuple)
    # 统计评估指标
    true_positive = 0
    d1_len = 0
    d2_len = 0
    for x in match_tids[1:]:
        true_positive += int(x[4])
    for _,v in dict1.items():
        d1_len += len(v)
    for _,v in dict2.items():
        d2_len += len(v)
    false_positive = d1_len - true_positive
    false_negative = d2_len - true_positive
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (false_negative + true_positive)
    f1_score = 200 * recall * precision / (precision + recall)
    return precision, recall, f1_score

def add_zone_num(lines,zones):
    mot_list = dict()
    for line in lines:
        fid = int(lines[line]['frame'][3:]) # 
        tid = lines[line]['id']
        bbox = list(map(lambda x:int(float(x)), lines[line]['bbox']))
        if tid not in mot_list:
            mot_list[tid] = dict()
        out_dict = lines[line]
        out_dict['zone'] = zones.get_zone(bbox) # 给bbox分配了zone_num
        mot_list[tid][fid] = out_dict # 字典
    return mot_list # 字典

def gather_sequence_info(det_feat_dic): # detection_file存的是det_feat.pkl的路径，改成存有feat的字典feat_dict
    min_frame_idx = 0
    max_frame_idx = 2000    
    update_ms = None
    feature_dim = 2048 # default
    bbox_dic = {}
    feat_dic = {}
    for image_name in det_feat_dic:
        # 获取帧idx
        frame_index = int(image_name.split('_')[1])
        det_bbox = np.array(det_feat_dic[image_name]['bbox']).astype('float32')
        det_feat = det_feat_dic[image_name]['feat']
        score = det_feat_dic[image_name]['conf']
        score = np.array((score,))
        det_bbox = np.concatenate((det_bbox, score)).astype('float32')
        if frame_index not in bbox_dic:
            bbox_dic[frame_index] = [det_bbox]
            feat_dic[frame_index] = [det_feat]
        else:
            bbox_dic[frame_index].append(det_bbox)
            feat_dic[frame_index].append(det_feat)
    seq_info = {
        "detections": [bbox_dic, feat_dic],
        # "groundtruth": groundtruth,
        # "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms,
        "frame_rate": 10 # 帧率
         # "frame_rate": int(info_dict["frameRate"])
    }
    return seq_info

class ReidFeature():
    """Extract reid feature."""

    def __init__(self, gpu_id, _mcmt_cfg):
        print("init reid model")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.model, self.reid_cfg = build_reid_model(_mcmt_cfg)
        device = torch.device('cuda')
        print('device: ', device)
        self.model = self.model.to(device)
        self.model.eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.val_transforms = T.Compose([T.Resize(self.reid_cfg.INPUT.SIZE_TEST, interpolation=3),\
                              T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def extract(self, img_dict):
        """Extract image feature with given image path.
        Feature shape (2048,) float32."""

        img_batch = []
        for _, img0 in img_dict.items():
            img = Image.fromarray(img0)
            img = self.val_transforms(img)
            img = img.unsqueeze(0)
            img_batch.append(img)
        img = torch.cat(img_batch, dim=0)

        with torch.no_grad():
            img = img.to('cuda')
            flip_feats = False
            if self.reid_cfg.TEST.FLIP_FEATS == 'yes': flip_feats = True
            if flip_feats:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                        feat1 = self.model(img)
                    else:
                        feat2 = self.model(img)
                feat = feat2 + feat1
            else:
                feat = self.model(img)
        feat = feat.cpu().detach().numpy()
        return feat

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    # clip_coords(coords, img0_shape)
    return coords

# 更换视频源时，修改视频路径，配置zone分区图和roi图即可
# weights = 'yolov5s.pt' # 模型大小
# imgsz = 1280 # default是640，传入1280 [640, 1280]
# cams_ratio的长度应与摄像头个数相同, 抽帧间隔，默认连续取1，可设置为1，2，3
def run_mtmc(cams_dir = 'datasets/AIC22_Track1_MTMC_Tracking/train/S10',
             weights = 'yolov5s.pt', 
             imgsz = 1280, 
             cams_ratio = [1, 1, 1, 1, 1]):
    # 加载模型
    device = select_device('')
    det_model = attempt_load(weights, map_location=device)
    det_model.half() # to FP16

    # 重识别模型
    global ext_model
    ext_model = ReidFeature(GPU_ID, extract_cfg)

    # 加载数据
    cams = os.listdir(cams_dir)
    stride = int(det_model.stride.max()) # stride = 32
    datasets = {}
    trackers = {}
    results = {}           # key是cam，value是list
    pp_results = {}        # key是cam，value是list
    trackers_avg_feat = {} # key是cam，value是字典，该字典key是tid，value是平均特征向量
    gt_detect = {}  # 检测输出的gt
    gt_dict = {}    # 自带的gt
    frame_nums = {}

    for cam in cams:
        video_dir = os.path.join(cams_dir, cam) + '/vdo.mp4'
        gt_dir = os.path.join(cams_dir, cam) + '/gt/gt.txt'
        gt_dict[cam] = read_data_from_txt(gt_dir)
        gt_detect[cam] = {}
        frame_nums[cam] = []
        datasets[cam] = LoadImages(video_dir, img_size=imgsz, stride=stride)
        results[cam] = []
        pp_results[cam] = []
        trackers[cam] = JDETracker(min_confidence, vdo_frame_ratio)
        trackers_avg_feat[cam] = {}

    # names存的是目标检测结果的种类，检测[2,5,7](car,bus,truck)
    names = det_model.module.names if hasattr(det_model, 'module') else det_model.names
    # 创建n*m个模型的实例,初始化模型的权重,以便稍后推理
    det_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(det_model.parameters())))
    # 控制遍历帧数
    frame_cnt = 0
    # 时间偏置
    time_bias = 0.0
    # 每帧的车流密度
    zones = zone()
    # 统计时间
    total_detect_time = 0
    total_extract_time = 0
    total_sct_time = 0
    total_pp_time = 0
    total_match_time = 0
    
    while True:
        # 轮流处理每个摄像头的每一帧
        for cam_idx,cam in enumerate(cams):
            # gt文件写路径
            current_dict = dict()
            # 保存crop后的图像
            current_image_dict = dict()
            for path, img, im0s, vid_cap in datasets[cam]:
                if (getattr(datasets[cam], 'frame', 0) % cams_ratio[cam_idx] != 0):
                    continue
                # 格式化img
                img = torch.from_numpy(img).to(device)
                img = img.half()
                img /= 255.0
                # 确保维度正确
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # 开始推理
                t1 = time_synchronized()
                # pred 是一个list
                pred = det_model(img, augment=False)[0]
                # 去除检测结果中冗余的边界框
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[2, 5, 7], agnostic=True)
                # 处理推理结果
                for i, det in enumerate(pred):
                    # 传入的是图片，dataset没有frame属性，frame=0
                    # 传入的是视频，则frame是当前帧数
                    p, s, im0, frame_idx = path, '', im0s, getattr(datasets[cam], 'frame', 0)
                    p = Path(p)
                    s += '%gx%g ' % img.shape[2:]
                    # 创建一个包含图像尺寸信息的 PyTorch 张量 
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    # print("det 的长度为 %d" % len(det))
                    # print("det 的类型为 %s" % type(det)) <class 'torch.Tensor'>
                    if len(det):
                        img_det = np.copy(im0)
                        # print("rescale前的det{}".format(det))
                        # Rescale boxes from img_size to im0 size (缩放边框)
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        det_num = 0 #  局部id
                        for *xyxy, conf, cls in reversed(det):
                            x1,y1,x2,y2 = tuple(torch.tensor(xyxy).view(4).tolist())
                            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                            # shape返回 [高度, 宽度, 通道数]
                            if x1 < 0 or y1 < 0 or x2 > im0.shape[1]-1  or y2 > im0.shape[0]-1:
                                # print('clip bbox')
                                continue
                            if (y2-y1) * (x2-x1) < 1000:    # TODO: filter small bboxes
                                # print('det too small')
                                continue
                            if True:
                                det_name = "{}_{}_{:0>3d}".format(cam, frame_idx, det_num)
                                det_class = int(cls.tolist())
                                det_conf = conf.tolist()
                                current_image_dict[det_name] = img_det[y1:y2,x1:x2]
                                current_dict[det_name] = {
                                    'bbox': (x1,y1,x2,y2),
                                    'frame': frame_idx,
                                    'id': det_num,
                                    'imgname': det_name+".png",
                                    'class': det_class,
                                    'conf': det_conf
                                }
                            det_num += 1
                t2 = time_synchronized()
                total_detect_time += (t2 - t1)
                # 记录每帧的目标数
                frame_nums[cam].append([frame_idx, det_num])

                # 完成某个单视频的第n帧检测
                # current_image_dict中存有当前帧的车辆图片信息
                # current_dict中存有当前帧的bbox等信息
                break
            # 单视频特征提取
            if len(current_dict) == 0: # 未检测到车辆，跳过后续步骤
                continue
            t3 = time_synchronized()
            reid_feat_numpy = ext_model.extract(current_image_dict)
            # 用于保存提取出来的特征
            current_feat_dict = {}
            for index, ext_img in enumerate(current_image_dict.keys()):
                current_feat_dict[ext_img] = reid_feat_numpy[index]
            cur_det_feat_dict = current_dict.copy()
            for det_name, _ in current_dict.items():
                cur_det_feat_dict[det_name]['feat'] = current_feat_dict[det_name]
            t4 = time_synchronized()
            total_extract_time += (t4 - t3)

            # 单视频追踪
            t5 = time_synchronized()
            seq_info = gather_sequence_info(cur_det_feat_dict)          
            [bbox_dic, feat_dic] = seq_info['detections']
            if frame_idx not in bbox_dic:
                print(f'empty for {cam} {frame_idx}')
            detections = bbox_dic[frame_idx]
            feats = feat_dic[frame_idx]
            # Run non-maxima suppression.
            boxes = np.array([d[:4] for d in detections], dtype=float)
            scores = np.array([d[4] for d in detections], dtype=float)
            nms_keep = nms(torch.from_numpy(boxes),
                                    torch.from_numpy(scores),
                                    iou_threshold=0.99).numpy()
            detections = np.array([detections[i] for i in nms_keep], dtype=float)
            feats = np.array([feats[i] for i in nms_keep], dtype=float)

            # 更新对应的tracker (JDETracker目标追踪器)
            online_targets = trackers[cam].update(detections, feats, frame_idx)
            # online_targets = trackers[cam].update_without_embedding(detections, feats, frame_idx)
            # 更新对应的result
            for t in online_targets:
                tlwh = t.det_tlwh
                tid = t.track_id
                score = t.score
                feature = t.features[-1]
                feature = t.smooth_feat
                image_name = f'{cam}_{tid}_{frame_idx}'
                if tlwh[2] * tlwh[3] > 750:
                    results[cam].append([
                            frame_idx, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], image_name, score, feature
                        ])
            # 将results变为np.array, 并append(feat), 维度调整为2058(6 + 4 + 2048)
            t6 = time_synchronized()
            total_sct_time += (t6 - t5)
            # 准备后处理数据
            t7 = time_synchronized()
            pp_results[cam] = []
            # 按image_name排序，即按tid_fid排序
            for row in sorted(results[cam],key=lambda x: x[6]):
                [fid, tid, x, y, w, h] = row[:6]
                feat = row[-1]
                dummpy_input = np.array([fid, tid, x, y, w, h, -1, -1, -1, -1])
                dummpy_input = np.concatenate((dummpy_input, feat))
                pp_results[cam].append(dummpy_input)
            pp_results[cam] = np.array(pp_results[cam])
            # 执行后处理
            # pp_results[cam] = associate(pp_results[cam], 0.1, 10, cam)
            pp_results[cam] = track_nms(pp_results[cam], 0.65)

            cid = int(cam[-3:])
            zones.set_cam(cid)
            mot_feat_dic = {}
            for row in pp_results[cam]:
                [fid, tid, x, y, w, h] = row[:6]
                fid = int(fid)
                tid = int(tid)
                feat = np.array(row[-2048:])
                image_name = f'{cam}_{tid}_{fid}.png'
                bbox = (x, y, x+w, y+h)
                frame = f'img{int(fid):06d}'
                mot_feat_dic[image_name] = {'bbox': bbox, 'frame': frame, 'id': tid,
                                            'imgname': image_name, 'feat': feat}
            # 为bbox分配zone_num, 1 白色 2 红色 3 绿色 4 蓝色
            mot_list = add_zone_num(mot_feat_dic, zones)
            # 基于时间间隔和区域切分tracklet，切分间隔过久的和出现反向移动的轨迹
            mot_list = zones.break_mot(mot_list, cid)
            # 基于区域过滤tracklet
            # mot_list = zones.filter_mot(mot_list, cid)
            # 基于bbox过滤tracklet
            mot_list = zones.filter_bbox(mot_list, cid)
            t8 = time_synchronized()
            total_pp_time += (t8 - t7)

            t9 = time_synchronized()
            for tid in mot_list:
                tracklet = mot_list[tid]
                if (len(tracklet)) <= 1: continue
                frame_list = list(tracklet.keys())
                frame_list.sort()
                # 遍历pp_results[cam]，若某tracklet的fid满足frame_idx - tracklet_fid > 4，延迟0.4s,则认为该轨迹已离开摄像区域
                if ((frame_idx - frame_list[-1] > 4) and (tid not in trackers_avg_feat[cam])):
                    zone_list = [tracklet[f]['zone'] for f in frame_list]
                    feature_list = [tracklet[f]['feat'] for f in frame_list if (tracklet[f]['bbox'][3]-tracklet[f]['bbox'][1])*(tracklet[f]['bbox'][2]-tracklet[f]['bbox'][0])>2000]
                    if len(feature_list)<2:
                        feature_list = [tracklet[f]['feat'] for f in frame_list]
                    # 计算进出时间
                    io_time = [time_bias + frame_list[0] / 10., time_bias + frame_list[-1] / 10.]
                    # 计算轨迹的平均特征向量
                    all_feat = np.array([feat for feat in feature_list])
                    mean_feat = np.mean(all_feat, axis=0)
                    
                    start_zone = zone_list[0]
                    end_zone = zone_list[-1]
                    next_area = []
                    matched_tid = -1
                    # 与邻近摄像头中的轨迹进行跨视频匹配
                    if start_zone: # 起始区域为0，则必为新轨迹
                        matched_tid = cross_cam_match(cam, start_zone, io_time, mean_feat)
                        if end_zone:
                            next_area = next_cams_zone[cam][end_zone]

                    for next_cam, next_zone  in next_area:
                        # 可能匹配轨迹格式[out_time, g_tid, mean_feat, is_matched, tid, similarity]
                        Track_to_be_matched[next_cam][next_zone].append([io_time[1], matched_tid, mean_feat, False, -1, -1])

                    if matched_tid != -1:
                        gt_detect[cam][matched_tid] = {}
                        for i in frame_list:                  
                            gt_detect[cam][matched_tid][i] = [int(mot_list[tid][i]['bbox'][0]),int(mot_list[tid][i]['bbox'][1]),
                                                    int(mot_list[tid][i]['bbox'][2] - mot_list[tid][i]['bbox'][0]), 
                                                    int(mot_list[tid][i]['bbox'][3] - mot_list[tid][i]['bbox'][1])]

                    trackers_avg_feat[cam][tid] = {
                        'g_tid' : matched_tid,
                        'io_time': io_time,
                        'zone_list': zone_list,
                        'frame_list': frame_list,
                        'mean_feat': mean_feat,
                        # 'tracklet': tracklet
                    }
            t10 = time_synchronized()
            total_match_time += (t10 - t9)
        # 跳出while循环, 设置为2000时，可以跑完整个视频
        if frame_cnt == 2000:
            break
        frame_cnt += 1
        # end for cam in cams 
    # end while
    print('done')

    for cam in cams:
        gt_write = "town5/" + cam + "_gt_test.txt"
        detnum_write = "town5/" + cam + "_detnum.txt"
        with open(gt_write, "a") as gt_file:
            for gid, v in gt_detect[cam].items():
                for fid, bbox in v.items():
                    gt_file.write(f'{fid},{gid},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},1,3,-1,-1\n')

        with open(detnum_write, "w") as detnum_file:
            # 写每帧检测目标的数量
            for row in frame_nums[cam]:
                detnum_file.write(f'{row[0]},{row[1]}\n')
            
    with open('metrics.txt', "a") as result_file:
        for cam in cams:
            precision, recall, f1 = mot_metrics(gt_detect[cam], gt_dict[cam])
            result_file.write(f"{cam}的评估指标: IDP={round(precision*100, 3)}%, IDR={round(recall*100,3)}%, IDF1={round(f1,3)}%\n")
        result_file.write(f"目标检测总耗时:{total_detect_time},\n特征提取总耗时:{total_extract_time},\n单视频追踪总耗时:{total_sct_time},\n后处理总耗时:{total_pp_time},\n跨视频匹配总耗时:{total_match_time}")


if __name__ == '__main__':
    with torch.no_grad():
        run_mtmc()
        # pass