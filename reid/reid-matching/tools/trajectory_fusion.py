import os
from os.path import join as opj
import numpy as np
import pickle
from utils.zone_intra import zone
import sys
sys.path.append('../../../')
from config import cfg

def parse_pt(pt_file,zones):
    if not os.path.isfile(pt_file):
        return dict()
    with open(pt_file,'rb') as f:
        lines = pickle.load(f)
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

def parse_bias(timestamp_dir, scene_name):
    cid_bias = dict()
    for sname in scene_name:
        with open(opj(timestamp_dir, sname + '.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                cid = int(line[0][2:])
                bias = float(line[1])
                if cid not in cid_bias: cid_bias[cid] = bias
    return cid_bias

def out_new_mot(mot_list,mot_path):
    out_dict = dict()
    for tracklet in mot_list:
        tracklet = mot_list[tracklet]
        for f in tracklet:
            out_dict[tracklet[f]['imgname']]=tracklet[f]
    pickle.dump(out_dict,open(mot_path,'wb'))

if __name__ == '__main__':
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    scene_name = ['S06']
    data_dir = cfg.DATA_DIR
    save_dir = './exp/viz/test/S06/trajectory/'
    cid_bias = parse_bias(cfg.CID_BIAS_DIR, scene_name) # cid_bias = {41: 0.0, 42: 0.0, 43: 0.0, 44: 0.0, 45: 0.0, 46: 0.0}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    cam_paths = os.listdir(data_dir)
    cam_paths = list(filter(lambda x: 'c' in x, cam_paths))
    cam_paths.sort()
    zones = zone()

    for cam_path in cam_paths:
        print('processing {}...'.format(cam_path))
        cid = int(cam_path[-3:]) # 41
        f_w = open(opj(save_dir, '{}.pkl'.format(cam_path)), 'wb')
        cur_bias = cid_bias[cid]
        mot_path = opj(data_dir, cam_path,'{}_mot_feat.pkl'.format(cam_path)) # c041_mot_feat.pkl的路径
        new_mot_path = opj(data_dir, cam_path, '{}_mot_feat_break.pkl'.format(cam_path)) # 存放分裂过滤后的tracklet的pkl存储路径，
        print(new_mot_path)
        zones.set_cam(cid) 
        # mot_list的键是tid，值是一个字典，这个字典的键是fid
        mot_list = parse_pt(mot_path,zones) # 返回一个字典 给bbox分配zone_num
        mot_list = zones.break_mot(mot_list, cid) # 基于时间间隔和区域来分裂tracklet
        # mot_list = zones.comb_mot(mot_list, cid)
        mot_list = zones.filter_mot(mot_list, cid) # filter by zone 基于区域过滤tracklet
        mot_list = zones.filter_bbox(mot_list, cid)  # filter bbox  基于bbox过滤tracklet
        out_new_mot(mot_list, new_mot_path) # 将mot_list存到new_mot_path中

        tid_data = dict()
        for tid in mot_list:
            if cid not in [41,43,46,42,44,45]:
                break
            tracklet = mot_list[tid]
            if len(tracklet) <= 1: continue

            frame_list = list(tracklet.keys())
            frame_list.sort()
            # if tid==11 and cid==44:
            #     print(tid)
            zone_list = [tracklet[f]['zone'] for f in frame_list]
            # 在mot_feat.pkl中，bbox = (x, y, x+w, y+h)， 限制bbox大小
            feature_list = [tracklet[f]['feat'] for f in frame_list if (tracklet[f]['bbox'][3]-tracklet[f]['bbox'][1])*(tracklet[f]['bbox'][2]-tracklet[f]['bbox'][0])>2000]
            if len(feature_list)<2:
                feature_list = [tracklet[f]['feat'] for f in frame_list]
            # 进出时间 = [time1, time2]，视频的帧率为10
            io_time = [cur_bias + frame_list[0] / 10., cur_bias + frame_list[-1] / 10.]
            all_feat = np.array([feat for feat in feature_list])
            mean_feat = np.mean(all_feat, axis=0)

            tid_data[tid]={
                'cam': cid,
                'tid': tid,
                'mean_feat': mean_feat,
                'zone_list':zone_list,
                'frame_list': frame_list,
                'tracklet': tracklet,
                'io_time': io_time
            }

        pickle.dump(tid_data,f_w)
        f_w.close()
