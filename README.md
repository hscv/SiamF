# Quick Start

## The soure code of the paper "MATERIAL-GUIDED SIAMESE FUSION NETWORK FOR HYPERSPECTRAL OBJECT TRACKING".

## 1. Environment requirements & Dataset
Please follows: https://github.com/hscv/SEE-Net.

## 2. Add SiamF to your PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

## 3. Train
(a) cd first_trained_HFF/

(b) Download pretrained model in https://pan.baidu.com/s/1xUNW1wnyN7_Fo7Gcl1GaKQ   Access code: 1234 

(c) Change the path of training data in siamese/dataset/dataset.py

(d) Run:
```python
cd experiments/siamban_r50_l234
CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml
```

## 4. Test-wo online classifier
Download testing model in https://pan.baidu.com/s/1w7VfDe0fvMCk6ArtRgXR1Q
Access code: 2025 
```python
# cd first_trained_HFF/
python tools/demo.py --config experiments/siamban_r50_l234/config.yaml --snapshot trained_model.pth --video_path /data/XXX/HOT/dataset/test/test_HSI/
```

## 5. Test-with online classifier
Download testing model in https://pan.baidu.com/s/1w7VfDe0fvMCk6ArtRgXR1Q  
Access code: 2025 
```python
# cd second_test_HFF_OSC_OMC/
python tools/demo.py --config experiments/siamban_r50_l234/config.yaml --snapshot trained_model.pth --fusion wavg --sample_num 125 --temple_update 0 --skip_atom 10 --abundance_update 1 --video_path /data/XXX/HOT/dataset/test/test_HSI/ 
```

## Citation
If these codes are helpful for you, please cite this paper:
```python
@INPROCEEDINGS{9746089,
  author={Li, Zhuanfeng and Xiong, Fengchao and Lu, Jianfeng and Zhou, Jun and Qian, Yuntao},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Material-Guided Siamese Fusion Network for Hyperspectral Object Tracking}, 
  year={2022},
  volume={},
  number={},
  pages={2809-2813},
```

## Contact
lizhuanfeng@njust.edu.cn
