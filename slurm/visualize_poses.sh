#!/bin/bash

# Job name
JOB_NAME="visualize_poses"

# Create a job-specific directory with a timestamp 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASEDIR="./slurm/Jobs/${TIMESTAMP}_${JOB_NAME}"
mkdir -p "${BASEDIR}"

# Redirect output and error to log files
exec > "${BASEDIR}/output.log" 2> "${BASEDIR}/error.log"

echo "Job name: ${JOB_NAME}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Environment setup (Conda)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultra-wide-nerf

# Main command
python dataset_utils/visualize_pose.py \
    --folder "data/" \
    --output "${BASEDIR}/poses_3d_visualization.png" \
    --no-orientations \
    --arrow-length 0.1 \
    #--show


# Log completion time and exit code
echo "End time: $(date)"
echo "Exit code: $?"