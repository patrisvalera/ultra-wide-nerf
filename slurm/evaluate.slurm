#!/bin/bash

#SBATCH --job-name=ultra-wide-nerf_ice
#SBATCH --output=/dev/null  # Temporary output to null
#SBATCH --error=/dev/null   # Temporary error to null
#SBATCH --gres=gpu:1        # Number of GPUs
#SBATCH --time=9-12:00:00   # Max run time
#SBATCH --cpus-per-task=1   # Number of CPUs 
#SBATCH --mem=48G           # Memory limit
#SBATCH --nodelist=gpu_name  # Specify the node name if needed

# ============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES
# ============================================================================

# Configuration
FT_PATH="./slurm/Jobs/"
CHKP="job/x.tar"
CHECKPOINT="${FT_PATH}${CHKP}"  


# Choose one of the following for evaluation:
## For full resolution subvolume evaluation
VOLS_LABEL="0"
POSES="./data/volume${VOLS_LABEL}/poses.npy"
TARGET_PATH="./data/volume${VOLS_LABEL}/images.npy"
## For no target comparison, e.g. trajectory, panorama
POSES="./slurm/Jobs/1_trajectory_pose_computation/poses.npy"
TARGET_PATH=""

# Config file
CONFIG="configs/config_convex.txt"

SUBSAMPLE_RATE=1  # Render every x slice

INTERP_METHOD="nearest"  
USE_SLICE_BY_SLICE=true
# if USE_SLICE_BY_SLICE, set flag to use use_slice_by_slice interpolation
if [ -z "${USE_SLICE_BY_SLICE}" ]; then
    USE_SLICE_BY_SLICE_FLAG="--use_slice_by_slice"

else
    USE_SLICE_BY_SLICE_FLAG=""
fi

#### Experiment naming ####
MODE="3D"
#MODE="2D"

#MIP_MODE="nomip"
MIP_MODE="mip"

#LOSS_TYPE="l2"
LOSS_TYPE="ssim"

N_FREQ="10"
OUTPUT_CH="3"
REGULARIZATION="reg"

EXP_NR="1"

EXPERIMENT="${MODE}_${MIP_MODE}_${LOSS_TYPE}_${N_FREQ}freq_${OUTPUT_CH}outputch_${REGULARIZATION}_exp${EXP_NR}_${INTERP_METHOD}_${SUBSAMPLE_RATE}x_volume${VOLS_LABEL}"
############################

# Training parameters
N_ITERS=50000000000
I_PRINT=1
CHUNK=900000

OUTPUT_CHANNELS=3

# ============================================================================
# SETUP
# ============================================================================

# Create job-specific directory
BASEDIR="./slurm/Jobs/${SLURM_JOB_ID}_${EXPERIMENT}"
mkdir -p "${BASEDIR}"

# Redirect all future output to log files
exec > "${BASEDIR}/output.log" 2> "${BASEDIR}/error.log"

# Create output directory
EVAL_DIR="${BASEDIR}/eval_subsampled_x${SUBSAMPLE_RATE}"
RECON_DIR="${BASEDIR}/reconstructed_${INTERP_METHOD}"

echo "=================================================="
echo "UltraNeRF Evaluation with 3D Interpolation"
echo "=================================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Poses: ${POSES}"
echo "Subsample rate: ${SUBSAMPLE_RATE}x"
echo "Interpolation: ${INTERP_METHOD}"
echo "=================================================="
echo ""

# Environment setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultra-wide-nerf

# Save this script for reproducibility
cp "$0" "${BASEDIR}/evaluate.slurm"

# Step 1: Evaluate with subsampling
echo "Step 1: Evaluating model with depth subsampling..."
CMD="python evaluate_ultranerf.py \
    --ft_path ${CHECKPOINT} \
    --poses_path ${POSES} \
    --config ${CONFIG} \
    --use_convex_mode \
    --full_volume_mode \
    --n_iters ${N_ITERS} \
    --i_print ${I_PRINT} \
    --chunk ${CHUNK} \
    --output_ch ${OUTPUT_CHANNELS} \
    --subsample_depth_every_n ${SUBSAMPLE_RATE} \
    --save_volume \
    --save_subsampling_info \
    --output_dir ${EVAL_DIR} \
    --visualize_slices 3"

# Add target path if specified
if [ -n "${TARGET_PATH}" ]; then
    CMD+=" --target_path ${TARGET_PATH}"
fi

# Print command for reproducibility
echo "Command:"
echo "${CMD}"
echo ""

# Execute evaluation
eval ${CMD}

EXIT_CODE=$?

echo ""
echo "Step 1 complete! Subsampled volumes saved to: ${EVAL_DIR}"
echo ""

# If SUBSAMPLE_RATE is 1, skip reconstruction
if [ "${SUBSAMPLE_RATE}" -eq 1 ]; then
    echo "Subsample rate is 1, skipping reconstruction."
    exit ${EXIT_CODE}

    echo "=================================================="
    echo "WORKFLOW COMPLETE"
    echo "=================================================="

    # Load and display compression info
    export EVAL_DIR

    # Remove any accidentally created files
    rm -f slurm-${SLURM_JOB_ID}.out slurm-${SLURM_JOB_ID}.err

    exit ${EXIT_CODE}

fi

# Step 2: Reconstruct full resolution
echo "Step 2: Reconstructing full-resolution volumes..."
CMD="python reconstruct_from_subsampled.py \
    --eval_dir ${EVAL_DIR} \
    --method ${INTERP_METHOD} \
    --output_dir ${RECON_DIR} \
    --visualize \
    ${USE_SLICE_BY_SLICE_FLAG}"

# Print command for reproducibility
echo "Command:"
echo "${CMD}"
echo ""

# Execute reconstruction
eval ${CMD}

EXIT_CODE=$?

echo ""
echo "Step 2 complete! Reconstructed volumes saved to: ${RECON_DIR}"
echo ""

# Step 3: Print summary
echo "=================================================="
echo "WORKFLOW COMPLETE"
echo "=================================================="

# Load and display compression info
export EVAL_DIR
export RECON_DIR

python << 'END'
import numpy as np
import os

eval_dir = os.environ.get('EVAL_DIR', '')
recon_dir = os.environ.get('RECON_DIR', '')

info_path = os.path.join(eval_dir, 'subsampling_info.npz')
if os.path.exists(info_path):
    info = np.load(info_path)
    original_depth = int(info['original_depth'])
    subsampled_depth = int(info['subsampled_depth'])
    compression = subsampled_depth / original_depth
    
    print(f"Original depth: {original_depth} slices")
    print(f"Subsampled depth: {subsampled_depth} slices")
    print(f"Compression ratio: {compression:.1%}")
    print(f"Rendering speedup: ~{1/compression:.1f}x faster")
    print(f"")
    print(f"Subsampled volumes: {eval_dir}/renderings/")
    print(f"Reconstructed volumes: {recon_dir}/")
    print(f"Quality visualizations: {recon_dir}/viz_*.png")
else:
    print(f"Could not find subsampling info at: {info_path}")
END

echo "=================================================="
echo ""
echo "Next steps:"
echo "1. View quality comparisons: ${RECON_DIR}/viz_*.png"
echo "2. Load volumes for analysis: ${RECON_DIR}/reconstructed_*.npy"
echo "3. Compare with full-resolution rendering (set --subsample_depth_every_n 1)"
echo ""

# Remove any accidentally created files
rm -f slurm-${SLURM_JOB_ID}.out slurm-${SLURM_JOB_ID}.err

exit ${EXIT_CODE}
