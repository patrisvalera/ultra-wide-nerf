#!/bin/bash

#SBATCH --job-name=ultra-wide-nerf_ice
#SBATCH --output=/dev/null  
#SBATCH --error=/dev/null   
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --time=9-12:00:00  
#SBATCH --cpus-per-task=1  
#SBATCH --mem=48G  
#SBATCH --nodelist=gpu_name  # Specify the node name if needed

# Create job-specific directory (uses SLURM_JOB_ID available at runtime)
EXPERIMENT="trajectory_pose_computation"

DATADIR="./data/"

# Specify which volumes to use for averaging
#VOLUMES="0,50,80,100,130,150,180,210,230,260,280,310,350"
VOLUMES="0-350"

AXIS="z"
overlap_strategy="next_to_prior"   # Strategy for handling overlapping regions

BASEDIR="./slurm/Jobs/${SLURM_JOB_ID}_${EXPERIMENT}"
mkdir -p "${BASEDIR}"

# Redirect all future output to log files in BASEDIR
exec > "${BASEDIR}/output.log" 2> "${BASEDIR}/error.log"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Environment setup
#conda init bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultra-wide-nerf

# Compute aquare pose from selected volumes
echo "Computing average pose from selected volumes..."
python dataset_utils/compute_trajectory_poses.py \
    --data_folder "${DATADIR}" \
    --axis "${AXIS}" \
    --overlap_strategy "${overlap_strategy}" \
    --output_path "${BASEDIR}/poses" \
    --volumes "${VOLUMES}" \


echo "End time: $(date)"
echo "Exit code: $?"

# Remove any accidentally created files
rm -f slurm-${SLURM_JOB_ID}.out slurm-${SLURM_JOB_ID}.err