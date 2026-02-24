#!/bin/bash
# export PATH="/home/ychen17/miniforge3/envs/work/bin:$PATH"
START_TIME=$(date +%s)

# --------------------------
# This part is used to get account/token information from shell env file.
# Note this may not work for a cron job. In order to run the code through a cron job, please set the env variables directly in the cron job script.
source /home/ychen17/.bashrc
# --------------------------
# /home/ychen17/miniforge3/envs/work/bin/python ~/GFED5NRT/Code/GFED5NRT.py > /dev/null 2>&1
PYTHON_BIN="/home/ychen17/miniforge3/envs/work/bin/python"
SCRIPT_PATH="/home/ychen17/GFED5NRT/Code/GFED5NRT.py"
LOG_PATH="/home/ychen17/GFED5NRT/Code/python.log"
rm -f "$LOG_PATH"
"$PYTHON_BIN" "$SCRIPT_PATH" >> "$LOG_PATH" 2>&1
# /home/ychen17/miniforge3/envs/work/bin/python ~/GFED5NRT/Code/GFED5NRT.py > /home/ychen17/GFED5NRT/Code/python.log 2>&1
END_TIME=$(date +%s)
RUN_TIME=$((END_TIME - START_TIME))
echo "Script GFED5NRT.sh finished at $(date)"
echo "Running time: ${RUN_TIME} seconds"
