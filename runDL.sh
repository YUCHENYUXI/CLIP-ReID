#!/bin/bash

# 创建名为 DL 的 screen 会话
screen -S DL -d -m

# 在 DL 会话中创建一个窗口并运行训练脚本
screen -S DL -X screen -t Training
## 根据实际情况修改
screen -S DL -p Training -X stuff "conda activate py8\n"
## 根据实际情况修改
screen -S DL -p Training -X stuff "CUDA_VISIBLE_DEVICES=0 python train_vid.py --config_file configs/person/vit_base_rgb.yml\n"

# 在 DL 会话中创建第二个窗口并运行 turnoff.sh 脚本
screen -S DL -X screen -t TF
## 根据实际情况修改
screen -S DL -p TF -X stuff "bash /root/rgbe/git/CLIP-ReID/turnoff.sh\n"

# 输出成功信息
echo "Screen session 'DL' has been started with two windows:"
echo "1. 'Training' running: conda activate py8 and python train_vid.py"
echo "2. 'TF' running: bash /root/rgbe/git/CLIP-ReID/turnoff.sh"
