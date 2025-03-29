from pycocotools.coco import COCO
import os
from os.path import join
import json

def test_pre():
    dataDir = './data'
    count = 0
    for dataType in ['val2017']: # , 'train2017'
        dataset = dict()
        annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
        coco = COCO(annFile)
        n_imgs = len(coco.imgs)
        for n, img_id in enumerate(coco.imgs):
            # subset: val2017 image id: 0000 / 5000
            print('subset: {} image id: {:04d} / {:04d}'.format(dataType, n, n_imgs))
            img = coco.loadImgs(img_id)[0]
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)
            video_crop_base_path = join(dataType, img['file_name'].split('/')[-1].split('.')[0])
            if len(anns) > 0:
                dataset[video_crop_base_path] = dict()        

            for trackid, ann in enumerate(anns):
                rect = ann['bbox']
                c = ann['category_id']
                bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
                if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                    count += 1
                    print(count, rect)
                    continue
                dataset[video_crop_base_path]['{:02d}'.format(trackid)] = {'000000': bbox}

        print('save json (dataset), please wait 20 seconds~')
        json.dump(dataset, open('{}.json'.format(dataType), 'w'), indent=4, sort_keys=True)
        print('done!')


def test_last():
    fold = 'test_HSI'
    rootDir = '../../../../../hsi_data_whisper/train'
    dataDir = rootDir+'/whisper_train'
    count = 0
    for dataType in ['whisper_train']:
        dataset = dict()
        video_name_arr = os.listdir(dataDir)
        video_name_arr.sort()
        for video_name in video_name_arr:
            print ('video_name = ',video_name)
            annFile = '{}/whisper_train/{}/HSI/groundtruth_rect.txt'.format(rootDir,video_name)
            gt = open(annFile)
            gtArr1 = gt.readlines()
            gtArr = []
            for dataGt in gtArr1: 
                tmpData = dataGt.split('\t')[:-1]
                tmpData = list(map(int,tmpData))
                gtArr.append(tmpData)
            n_imgs = len(gtArr)
            for trackid, ann in enumerate(gtArr):
                video_crop_base_path = 'trainHSI/%s/%s_%04d' % (video_name, video_name, trackid+1)
                dataset[video_crop_base_path] = dict()
                rect = ann
                bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
                if rect[2] <= 0 or rect[3] <= 0: 
                    count += 1
                    print(count, rect)
                    continue
                dataset[video_crop_base_path]['{:02d}'.format(0)] = {'000000': bbox}

        print ('dataType = ',dataType)
        print('save json (dataset), please wait 20 seconds~')
        json.dump(dataset, open('{}.json'.format(dataType), 'w'), indent=4, sort_keys=True)
        print('done!')


if __name__ == '__main__':
    test_last()