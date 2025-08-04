#!/bin/bash
# export PATH="/home/ychen17/anaconda3/bin:$PATH"
START_TIME=$(date +%s)
~/miniforge3/envs/work/bin/python ~/GFED5eNRT/Code/GFED5eNRT.py > /dev/null 2>&1
END_TIME=$(date +%s)
RUN_TIME=$((END_TIME - START_TIME))
echo "Script GFED5eNRT.sh finished at $(date)"
echo "Running time: ${RUN_TIME} seconds"
