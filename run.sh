#!/bin/bash

# 获取当前时间戳并转为十六进制
timestamp=$(date +%s)
hex_name=$(printf "%X" $timestamp)

# 日志文件名前缀（可选，也可以硬编码）
log_prefix="./res/$hex_name"  # 请确保这个目录存在

# 创建 screen 会话，启用日志并指定日志文件
screen -S "$hex_name" -L -Logfile "${log_prefix}_main.log" -d -m

# 创建第一个窗口：Training，带日志
screen -S "$hex_name" -X screen -t Training
screen -S "$hex_name" -p Training -X logfile "${log_prefix}_training.log"
screen -S "$hex_name" -p Training -X log on
screen -S "$hex_name" -p Training -X stuff "conda activate py8\n"
screen -S "$hex_name" -p Training -X stuff "CUDA_VISIBLE_DEVICES=0 python train_vid.py --config_file configs/person/vit_base_rgb.yml\n"

# 创建第二个窗口：TF，带日志
screen -S "$hex_name" -X screen -t TF
screen -S "$hex_name" -p TF -X logfile "${log_prefix}_tf.log"
screen -S "$hex_name" -p TF -X log on
screen -S "$hex_name" -p TF -X stuff "bash /root/rgbe/git/CLIP-ReID/turnoff.sh\n"

# 输出成功信息
echo "Screen session '$hex_name' has been started with two windows and logging enabled:"
echo "1. 'Training' running: conda activate py8 and python train_vid.py"
echo "2. 'TF' running: bash /root/rgbe/git/CLIP-ReID/turnoff.sh"
echo "Log files stored in: ${log_prefix}_*.log"
