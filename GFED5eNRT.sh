#!/bin/bash
# export PATH="/home/ychen17/miniforge3/envs/work/bin:$PATH"
START_TIME=$(date +%s)
source /home/ychen17/.bashrc
# /home/ychen17/miniforge3/envs/work/bin/python ~/GFED5eNRT/Code/GFED5eNRT.py > /dev/null 2>&1
PYTHON_BIN="/home/ychen17/miniforge3/envs/work/bin/python"
SCRIPT_PATH="/home/ychen17/GFED5eNRT/Code/GFED5eNRT.py"
LOG_PATH="/home/ychen17/GFED5eNRT/Code/python.log"
rm -f "$LOG_PATH"
"$PYTHON_BIN" "$SCRIPT_PATH" >> "$LOG_PATH" 2>&1
# /home/ychen17/miniforge3/envs/work/bin/python ~/GFED5eNRT/Code/GFED5eNRT.py > /home/ychen17/GFED5eNRT/Code/python.log 2>&1
END_TIME=$(date +%s)
RUN_TIME=$((END_TIME - START_TIME))
echo "Script GFED5eNRT.sh finished at $(date)"
echo "Running time: ${RUN_TIME} seconds"
