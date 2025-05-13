#!/bin/bash

# 检查参数
useTurnoff=$1
if [ -z "$useTurnoff" ]; then
  useTurnoff=0
fi

# 获取当前时间戳并转为十六进制
timestamp=$(date +%s)
hex_name=$(printf "%X" $timestamp)

# 配置
env_cmd="conda activate py8 && "
cfg="configs/person/vit_base_rgb.yml"
command="CUDA_VISIBLE_DEVICES=0 python train_vid.py --config_file ${cfg}"
log_prefix="./res/$hex_name"  # 请确保这个目录存在

# 确保日志目录存在
mkdir -p ./res

# 创建 screen 会话
screen -S "$hex_name" -d -m

# 创建第一个窗口：Training
screen -S "$hex_name" -X screen -t Training
screen -S "$hex_name" -p Training -X logfile "${log_prefix}_training.log"
screen -S "$hex_name" -p Training -X log on
screen -S "$hex_name" -p Training -X stuff "${env_cmd}${command}\n"

# 创建第二个窗口（可选）
if [ "$useTurnoff" -eq 1 ]; then
    screen -S "$hex_name" -X screen -t TF
    screen -S "$hex_name" -p TF -X stuff "bash /root/rgbe/git/CLIP-ReID/turnoff.sh\n"
fi

# 输出信息
echo "✅ Screen session '$hex_name' 已启动："
echo "1. 'Training' 正在运行: conda activate py8 && python train_vid.py"
if [ "$useTurnoff" -eq 1 ]; then
    echo "2. 'TF' 正在运行: bash /root/rgbe/git/CLIP-ReID/turnoff.sh"
fi
echo "📝 日志保存在: ${log_prefix}_*.log"
