#!/bin/bash

PATTERN="python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config configs/deploy/kuavo_env.yaml"

PID=$(pgrep -f "$PATTERN")

if [ -z "$PID" ]; then
    echo "未找到目标进程"
else
    echo "找到进程 PID: $PID，正在杀死..."
    kill -9 $PID
    echo "进程已终止"
fi