#!/bin/bash

# 获取当前时间戳并转为十六进制
timestamp=$(date +%s)
hex_name=$(printf "%X" $timestamp)

# 创建名为十六进制时间戳的 screen 会话
screen -S "$hex_name" -d -m

# 在该会话中创建一个窗口并运行训练脚本
screen -S "$hex_name" -X screen -t Training
screen -S "$hex_name" -p Training -X stuff "conda activate py8\n"
screen -S "$hex_name" -p Training -X stuff "CUDA_VISIBLE_DEVICES=0 python train_vid.py --config_file configs/person/vit_base_rgb.yml\n"

# 在该会话中创建第二个窗口并运行 turnoff.sh 脚本
screen -S "$hex_name" -X screen -t TF
screen -S "$hex_name" -p TF -X stuff "bash /root/rgbe/git/CLIP-ReID/turnoff.sh\n"

# 输出成功信息
echo "Screen session '$hex_name' has been started with two windows:"
echo "1. 'Training' running: conda activate py8 and python train_vid.py"
echo "2. 'TF' running: bash /root/rgbe/git/CLIP-ReID/turnoff.sh"
