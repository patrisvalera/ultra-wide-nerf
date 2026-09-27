#!/bin/bash

#SBATCH --job-name=ultra-wide-nerf_ice
#SBATCH --output=/dev/null  # Temporary output to null
#SBATCH --error=/dev/null   # Temporary error to null
#SBATCH --gres=gpu:1        # Number of GPUs
#SBATCH --time=4-12:00:00   # Max run time
#SBATCH --cpus-per-task=1   # Number of CPUs 
#SBATCH --mem=48G           # Memory limit
#SBATCH --nodelist=gpu_name  # Specify the node name if needed

# ============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES
# ============================================================================

# Config
CONFIG="configs/config_convex.txt"

# Data configuration
## Root data directory
DATADIR="./data/"
VOLUME_PREFIX="volume"                   # Volume directory prefix

# Volume selection (use actual volume numbers from directory names)
# Examples:
#   - "0,20,110" for specific volumes (volume0, volume20, volume110)
#   - "0-5" for index range (first 6 discovered volumes)
#   - "even" or "odd" for alternating volumes
TRAIN_VOLUMES="0,70,100,110"
VAL_VOLUMES="50,80"
TEST_VOLUMES="50"

#### Experiment naming ####
MODE="2D"
MODE="3D"

MIP_MODE="nomip"
MIP_MODE="mip"

#LOSS_TYPE="l2"
LOSS_TYPE="ssim"

N_FREQ="10"
OUTPUT_CH="3"
REGULARIZATION="reg"

EXP_NR="1"

VOLS_LABEL="vols_${TRAIN_VOLUMES//,/_}"  # Creates label like "vols_0_20_110"

EXPERIMENT="${MODE}_${MIP_MODE}_${LOSS_TYPE}_${N_FREQ}freq_${OUTPUT_CH}outputch_${REGULARIZATION}_${VOLS_LABEL}_exp${EXP_NR}"
############################

# Training parameters
N_ITERS=10000000
I_PRINT=10000
CHUNK=40000
OUTPUT_CHANNELS=3

# Checkpointing
# Save model every N iterations
i_weights=10000

# Optional fine-tuning checkpoint
FT_PATH=""  # Leave empty for training from scratch
# FT_PATH="./slurm/Jobs/previous_job/ice/100000.tar"

# ============================================================================
# SETUP
# ============================================================================

# Create job-specific directory
BASEDIR="./slurm/Jobs/${SLURM_JOB_ID}_${EXPERIMENT}"
mkdir -p "${BASEDIR}"
mkdir -p "${BASEDIR}/visualizations"

# Redirect all future output to log files
exec > "${BASEDIR}/output.log" 2> "${BASEDIR}/error.log"

echo "============================================================================"
echo "ICE-NERF MULTI-VOLUME TRAINING"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Experiment: ${EXPERIMENT}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo ""
echo "Data Configuration:"
echo "  Data directory: ${DATADIR}"
echo "  Train volumes: ${TRAIN_VOLUMES}"
echo "  Val volumes: ${VAL_VOLUMES}"
echo "  Test volumes: ${TEST_VOLUMES}"
echo "============================================================================"
echo ""

# Environment setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultra-wide-nerf

# Save this script for reproducibility
cp "$0" "${BASEDIR}/train_ultranerf_multivol.slurm"
# ============================================================================
# PRE-TRAINING VISUALIZATION
# ============================================================================

echo "Generating pose visualizations..."
echo ""

# Visualize all selected volumes
echo "1. Visualizing all training data..."
python dataset_utils/visualize_pose.py \
    --folder "${DATADIR}" \
    --volumes "${TRAIN_VOLUMES},${VAL_VOLUMES},${TEST_VOLUMES}" \
    --volume-prefix "${VOLUME_PREFIX}" \
    --output "${BASEDIR}/visualizations/all_volumes.png" \
    --arrow-length 0.01

# Visualize training volumes only
echo "2. Visualizing training volumes..."
python dataset_utils/visualize_pose.py \
    --folder "${DATADIR}" \
    --volumes "${TRAIN_VOLUMES}" \
    --volume-prefix "${VOLUME_PREFIX}" \
    --output "${BASEDIR}/visualizations/train_volumes.png" \
    --arrow-length 0.01

# Visualize validation volumes
if [ -n "${VAL_VOLUMES}" ]; then
    echo "3. Visualizing validation volumes..."
    python dataset_utils/visualize_pose.py \
        --folder "${DATADIR}" \
        --volumes "${VAL_VOLUMES}" \
        --volume-prefix "${VOLUME_PREFIX}" \
        --output "${BASEDIR}/visualizations/val_volumes.png" \
        --arrow-length 0.01
fi

# Visualize test volumes
if [ -n "${TEST_VOLUMES}" ]; then
    echo "4. Visualizing test volumes..."
    python dataset_utils/visualize_pose.py \
        --folder "${DATADIR}" \
        --volumes "${TEST_VOLUMES}" \
        --volume-prefix "${VOLUME_PREFIX}" \
        --output "${BASEDIR}/visualizations/test_volumes.png" \
        --arrow-length 0.01
fi

echo ""
echo "Visualizations saved to: ${BASEDIR}/visualizations/"
echo ""

# ============================================================================
# TRAINING
# ============================================================================

echo "============================================================================"
echo "STARTING TRAINING"
echo "============================================================================"
echo ""


# Build training command
CMD="python run_ultranerf.py \
    --config ${CONFIG} \
    --datadir ${DATADIR} \
    --use_multi_volume_data_loading \
    --train_volumes ${TRAIN_VOLUMES} \
    --val_volumes ${VAL_VOLUMES} \
    --test_volumes ${TEST_VOLUMES} \
    --volume_prefix ${VOLUME_PREFIX} \
    --n_iters ${N_ITERS} \
    --i_print ${I_PRINT} \
    --basedir ${BASEDIR} \
    --chunk ${CHUNK} \
    --output_ch ${OUTPUT_CHANNELS} \
    --i_weights ${i_weights} \
    --tensorboard \
    --debug_viz_render \
    "

# Add fine-tuning path if specified
if [ -n "${FT_PATH}" ]; then
    CMD="${CMD} --ft_path ${FT_PATH}"
    echo "Fine-tuning from: ${FT_PATH}"
fi

# Optional: Add regularization
# CMD="${CMD} --reg"

# Print command for reproducibility
echo "Command:"
echo "${CMD}"
echo ""

# Execute training
eval ${CMD}

EXIT_CODE=$?

# ============================================================================
# POST-TRAINING
# ============================================================================

echo ""
echo "============================================================================"
echo "TRAINING COMPLETE"
echo "============================================================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo ""

# Print training summary
if [ -f "${BASEDIR}/ice/split_info.json" ]; then
    echo "Training split information:"
    cat "${BASEDIR}/ice/split_info.json"
    echo ""
fi

# List checkpoints
echo "Saved checkpoints:"
ls -lh "${BASEDIR}/ice/"*.tar 2>/dev/null || echo "No checkpoints found"
echo ""

# Visualizations
echo "Pose visualizations:"
ls -lh "${BASEDIR}/visualizations/"*.png 2>/dev/null || echo "No visualizations found"
echo ""

echo "Output directory: ${BASEDIR}"
echo "TensorBoard logs: ${BASEDIR}/summaries/"
echo "============================================================================"

# Remove any accidentally created files
rm -f slurm-${SLURM_JOB_ID}.out slurm-${SLURM_JOB_ID}.err

exit ${EXIT_CODE}
