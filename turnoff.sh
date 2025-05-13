#!/bin/bash

# 设置阈值
GPU_THRESHOLD=10  # GPU 使用率低于 10%
TIMEOUT=5         # 5 分钟
PID_CHECK_INTERVAL=60  # 每隔 1 分钟检查一次程序是否退出
LOG_FILE="./res/gpu_shutdown.log"  # 日志文件路径

# 检查 GPU 使用率
check_gpu_usage() {
    local usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    echo "$usage"
}

# 检查程序是否退出
check_pid_exit() {
    local pid=$1
    if ! ps -p $pid > /dev/null; then
        return 1  # 程序已经退出
    else
        return 0  # 程序仍在运行
    fi
}

# 记录日志函数
log_message() {
    local message=$1
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $message" >> $LOG_FILE
}

# 检查 GPU 使用率是否低于阈值持续 5 分钟
check_gpu_idle() {
    local idle_duration=0
    while [ $idle_duration -lt $TIMEOUT ]; do
        local usage=$(check_gpu_usage)
        if [ "$usage" -lt "$GPU_THRESHOLD" ]; then
            idle_duration=$((idle_duration + 1))
        else
            idle_duration=0  # GPU 使用恢复，重置计时器
        fi
        log_message "GPU usage: $usage%, idle duration: $idle_duration minutes"
        sleep 60
    done
    return 0  # GPU 使用率低于阈值持续 5 分钟，返回成功
}

# 执行关机操作
shutdown_system() {
    log_message "System idle for $TIMEOUT minutes. Shutting down..."
    shutdown now
}

# 主程序
main() {
    # 记录脚本启动日志
    log_message "Script started."

    # 可选：如果指定了程序PID，首先监测该程序退出
    if [ -n "$1" ]; then
        pid=$1
        log_message "Monitoring process with PID $pid for exit..."
        while true; do
            if ! check_pid_exit $pid; then
                log_message "Process $pid has exited."
                sleep 300  # 程序退出后等待 5 分钟
                shutdown_system
                break
            fi
            sleep $PID_CHECK_INTERVAL
        done
    else
        # 默认：监测 GPU 使用率
        log_message "Monitoring GPU usage..."
        check_gpu_idle
        shutdown_system
    fi

    # 记录脚本结束日志
    log_message "Script ended."
}

# 执行
main "$@"
