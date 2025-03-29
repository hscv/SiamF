from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.model_load import load_pretrain

torch.set_num_threads(1)

def set_init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--video_path', default='', type=str,
                    help='videos or image path')
parser.add_argument('--save', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()


pre_foldName = 'hsiPics'
suffix_foldName = 'HSI'
imageFilter = '*.png*'

def X2Cube(img):

    B = [4, 4]
    skip = [4, 4]
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    img = out.reshape(M//4, N//4, 16)
    return img

def get_gt_txt(gt_name):
    f = open(gt_name,'r')
    gt_arr = []
    gt_res = f.readlines()
    for gt in gt_res:
        kk = gt.split('\t')[:-1]
        x = list(map(int, kk))
        gt_arr.append(x)
    return gt_arr

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or \
        video_name.endswith('mov'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, imageFilter))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            if imageFilter == '*.png*':
                frame1 = cv2.imread(img, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                frame = X2Cube(frame1)
            else:
                frame = cv2.imread(img)
            yield frame


def get_frames_falseColor(video_name):
    video_name = video_name[:-3]+'img'
    images = glob(os.path.join(video_name, '*.jp*'))
    images = sorted(images,
                    key=lambda x: int(x.split('/')[-1].split('.')[0]))
    frameArr = []
    for img in images:
        frame = cv2.imread(img)
        frameArr.append(frame)
    return frameArr


def track_once(gtArr, video_path_name):
    args.save = True
    # load config
    cfg.merge_from_file(args.config)
    set_init_seed(cfg.TRACK.SEED)
    
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    model = ModelBuilder() 
    model = load_pretrain(model, args.snapshot).cuda().eval() 
    model.MSCABModel_b1.eval()
    model.MSCABModel_b2.eval()
    model.MSCABModel_b3.eval()
    model.MSCABModel_b4.eval()

    tracker = build_tracker(model) 
    video_name = video_path_name.split('/')[-1].split('.')[0] 
    model_name_tmp_m = args.snapshot.split('/')[-1].split('_')[-1][:-4]
    save_det_path = 'demo/'+pre_foldName+'/'+model_name_tmp_m+'/' + video_name+'_det.txt'
    if not os.path.exists('demo/'+pre_foldName+'/'+model_name_tmp_m):
        os.mkdir('demo/'+pre_foldName+'/'+model_name_tmp_m)
    video_path_name = video_path_name + '/'+suffix_foldName

    det_arr = []
    f = open(save_det_path,'w')
    gt_arr = gtArr

    first_frame = True
    falseColorFrameArr = get_frames_falseColor(video_path_name.replace('test_HSI','testFalseColor'))
    cnt = -1
    for frame in get_frames(video_path_name):
        cnt += 1
        if first_frame:
            first_gt = gt_arr[0]
            det_arr.append(first_gt)
            init_rect = np.array(first_gt)
            for tmp in first_gt:
                f.write(str(tmp)+'\t')
            f.write('\n')
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame,gt_arr[cnt])
            bbox = list(map(int, outputs['bbox']))
            print (cnt,', bbox = ', bbox, ' gt_arr[cnt] = ', gt_arr[cnt])
            det_arr.append(bbox)
            for tmp in bbox:
                f.write(str(tmp)+'\t')
            f.write('\n')
    f.close()
    return det_arr


def main():
    root = args.video_path
    video_dir_arr = []
    gtArr = []
    detArr = []
    dir_arr = os.listdir(root)
    dir_arr.sort()
    for d in dir_arr:
        path = os.path.join(root, d)
        if os.path.isdir(path):
            video_dir_arr.append(path)

    for video_name in video_dir_arr:
        gt_path = video_name + '/groundtruth_rect.txt'
        gt = get_gt_txt(gt_path)
        gtArr.append(gt)

    for i in range(len(video_dir_arr)): 
        print (video_dir_arr[i])
        detRes = track_once(gtArr[i], video_dir_arr[i])

if __name__ == '__main__':
    main()
