from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from siamban.core.config import cfg
from siamban.tracker.base_tracker import SiameseTracker
from siamban.utils.bbox import corner2center
from siamban.models.abundanceModel import AbundanceExtract
from siamban.utils.single_hsi_abundance import getAbundance

from siamban.tracker.classifier.base_classifier import BaseClassifier
import torch
from PIL import Image

def iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    return iou

def normalize(score):
    score = (score - np.min(score)) / (np.max(score) - np.min(score))
    return score

def getZhiXinPic(score):
    score -= score.min()
    score =score/ score.max()
    score = (score * 255).astype(np.uint8)
    score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
    return score

class SiamBANTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamBANTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)

        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.model = model
        self.model.eval()

        ## add new template update
        self.lost_count = 0
        self.abundance_status = cfg.TRACK.ABUNDANCE_UPDATE
        cfg.TRACK.USE_CLASSIFIER = cfg.TRACK.ATOM_UPDATE

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox, endLib, endNum):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.endLib = endLib
        self.endNum = endNum

        if cfg.TRACK.USE_CLASSIFIER:
            self.temp_max = 0
            self.frame_num = 1

        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop, _ = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.z_crop = z_crop
        with torch.no_grad():
            self.model.template(z_crop)
        print ('==========self.abundance============', self.abundance_status)
        if cfg.TRACK.USE_CLASSIFIER:
            if self.abundance_status:
                self.abundanceModel = AbundanceExtract(self.endNum)
                self.classifier = BaseClassifier(self.model, self.abundanceModel)
            else:
                self.classifier = BaseClassifier(self.model, None)

        if cfg.TRACK.USE_CLASSIFIER:
            self.z0_crop = z_crop 
            print ('=====================================================')
            print ('=================Template Update=====================')
            print ('=====================================================')
            with torch.no_grad():
                self.model.template_short_term(self.z0_crop)
            s_xx = s_z * (cfg.TRACK.INSTANCE_SIZE * 2 / cfg.TRACK.EXEMPLAR_SIZE)
            x_crop, _ = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE * 2,
                round(s_xx), self.channel_average)

            self.classifier.initialize(x_crop.type(torch.FloatTensor), bbox, self.endLib, self.endNum)

    def track(self, img, gt):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop,ori_im_patch_x = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        if self.abundance_status:
            abundance_x = getAbundance(ori_im_patch_x/ori_im_patch_x.max(), self.endLib, self.endNum)

            abundance_x = torch.from_numpy(abundance_x)
            if abundance_x.dim() != 4:
                abundance_x = abundance_x.unsqueeze(0) 
                abundance_x = abundance_x.permute(0,3,1,2) 
            if cfg.CUDA:
                abundance_x = abundance_x.cuda() 

        with torch.no_grad():
            outputs = self.model.track(x_crop) 

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)
        

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        if cfg.TRACK.USE_CLASSIFIER:
            self.frame_num += 1
            if cfg.TRACK.USE_CLASSIFIER:
                flag, s = self.classifier.track()
                if flag == 'not_found':
                    self.lost_count += 1
                else:
                    self.lost_count = 0

                confidence = Image.fromarray(s.detach().cpu().numpy())
                confidence = np.array(confidence.resize((self.score_size, self.score_size))).flatten()
                if self.abundance_status:
                    abundance_param = 0.72
                    sample_abund = abundance_x.unsqueeze(1).float()
                    score_abund = self.abundanceModel(sample_abund)
                    score_abund = score_abund.detach().cpu().numpy().squeeze() 
                    score_abund = cv2.resize(score_abund,(self.score_size, self.score_size)).flatten()
                    zxt_siamban = getZhiXinPic(pscore.reshape(25,25))
                    zxt_atom = getZhiXinPic(normalize(confidence).reshape(25,25))
                    zxt_abundance = getZhiXinPic(normalize(score_abund).reshape(25,25))
                    pscore = (1-abundance_param)*pscore+abundance_param*normalize(score_abund)
                    zxt_fuse = getZhiXinPic(pscore.reshape(25,25))
                else:
                    pscore = pscore * (1 - cfg.TRACK.COEE_CLASS) + \
                    normalize(confidence) * cfg.TRACK.COEE_CLASS
                pscore = pscore.flatten()

            # raise Exception
            if cfg.TRACK.TEMPLATE_UPDATE:
                score_st = self._convert_score(outputs['cls_st'])
                pred_bbox_st = self._convert_bbox(outputs['loc_st'], self.points)
                s_c_st = change(sz(pred_bbox_st[2, :], pred_bbox_st[3, :]) /
                                (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
                r_c_st = change((self.size[0] / self.size[1]) /
                                (pred_bbox_st[2, :] / pred_bbox_st[3, :]))
                penalty_st = np.exp(-(r_c_st * s_c_st - 1) * cfg.TRACK.PENALTY_K)
                pscore_st = penalty_st * score_st

                if self.abundance_status:
                    pscore_st = (1-abundance_param)*pscore_st+abundance_param*normalize(score_abund)
                else:
                    pscore_st = pscore_st * (1 - cfg.TRACK.COEE_CLASS) + \
                            normalize(confidence) * cfg.TRACK.COEE_CLASS
                pscore_st = pscore_st.flatten()
            pass
        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.SHORT_TERM_DRIFT and self.lost_count >= 8:
            cx, cy = bbox[0] / 4 + self.center_pos[0], bbox[1] / 4 + self.center_pos[1]
        else:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.TEMPLATE_UPDATE:
            pscore_st = pscore_st * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     self.window * cfg.TRACK.WINDOW_INFLUENCE
            best_idx_st = np.argmax(pscore_st)
            bbox_st = pred_bbox_st[:, best_idx_st] / scale_z
            lr_st = penalty_st[best_idx_st] * score_st[best_idx_st] * cfg.TRACK.LR
            if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.SHORT_TERM_DRIFT and self.lost_count >= 8:
                cx_st, cy_st = bbox_st[0] / 4 + self.center_pos[0], bbox_st[1] / 4 + self.center_pos[1]
            else:
                cx_st, cy_st = bbox_st[0] + self.center_pos[0], bbox_st[1] + self.center_pos[1]
            width_st = self.size[0] * (1 - lr_st) + bbox_st[2] * lr_st
            height_st = self.size[1] * (1 - lr_st) + bbox_st[3] * lr_st
            cx_st, cy_st, width_st, height_st = self._bbox_clip(cx_st, cy_st, width_st, height_st, img.shape[:2])
            if iou((cx_st, cy_st, width_st, height_st), (cx, cy, width, height), wh=True) >= cfg.TRACK.TAU_REGRESSION \
                and score_st[best_idx_st] - score[best_idx] >= cfg.TRACK.TAU_CLASSIFICATION:
                cx, cy, width, height, score, best_idx = cx_st, cy_st, width_st, height_st, score_st, best_idx_st


        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        if cfg.TRACK.USE_CLASSIFIER:
            if self.abundance_status:
                self.classifier.update(bbox, scale_z, flag, abundance_x)
            else:
                self.classifier.update(bbox, scale_z, flag, None)

            if cfg.TRACK.TEMPLATE_UPDATE:
                maxNum = torch.max(s).item()
                if self.abundance_status:
                    if score_abund.max() > maxNum:
                        maxNum = score_abund.max()
                if maxNum >= cfg.TRACK.TARGET_UPDATE_THRESHOLD and flag != 'hard_negative':
                    if maxNum > self.temp_max:
                        self.temp_max = maxNum
                        self.channel_average = np.mean(img, axis=(0, 1))
                        self.z_crop, _ = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)

                if (self.frame_num - 1) % cfg.TRACK.TARGET_UPDATE_SKIPPING == 0:
                    self.temp_max = 0
                    with torch.no_grad():
                        self.model.template_short_term(self.z_crop)

        if cfg.TRACK.USE_CLASSIFIER and self.abundance_status:
            return {
                    'bbox': bbox,
                    'best_score': best_score,
                    'flag': flag,
                    'zxt_siamban': zxt_siamban,
                    'zxt_atom': zxt_atom,
                    'zxt_abundance': zxt_abundance,
                    'zxt_fuse': zxt_fuse
                   }
        else:
            return {
                    'bbox': bbox,
                    'best_score': best_score
                   }
