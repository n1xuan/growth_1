#!/bin/bash
# Training pipeline for dendrite 4D reconstruction.
#
# This mirrors demo_synthetic.sh but is designed for the *experimental data*
# paradigm in neural_xray: real volumes at first/last frame, sparse projections
# in between. The synthetic balls demo is for quick validation only; the actual
# research workflow follows the experimental path (Kelvin lattice, Alporas foam).
#
# Pipeline stages (same as neural_xray experimental):
#   1. Canonical forward  — NeRF for t=0 volume (from full-angle projections)
#   2. Canonical backward — NeRF for t=T volume (from full-angle projections)
#   3. Combine checkpoints
#   4. Velocity field res 6  — learn deformation from sparse projections
#   5. Velocity field res 12 — refine deformation at higher resolution
#
# For the full resolution cascade (6→9→15→27→51 + spatiotemporal mixing),
# use submit.sh as reference.
#
# Usage:
#   ./scripts/demo_dendrite.sh [data_dir] [num_steps]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Dataset configuration
DATA_DIR="${1:-$PROJECT_ROOT/data/dendrite}"
NUMSTEPS="${2:-3000}"
DSET="dendrite"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

# Auto-detect number of frames from aggregated transforms filename
NUM_FRAMES=$(python3 -c "
from pathlib import Path
agg = sorted(Path('$DATA_DIR').glob('transforms_00_to_*.json'))
if agg:
    print(int(agg[-1].stem.split('_')[-1]) + 1)
else:
    print(20)
" 2>/dev/null || echo "20")

LAST_IDX=$((NUM_FRAMES - 1))
LAST_IDX_PAD=$(printf "%02d" $LAST_IDX)

# Training parameters (matching demo_synthetic.sh)
BATCH_SIZE=2048
BATCH_SIZE_VF=256
VF_NUM_SAMPLES_PER_RAY=256
DOWNSCALE_FACTOR=2
WEIGHT_NN_WIDTH=20
EVAL_BATCH_SIZE=$((BATCH_SIZE / 2))
EVAL_BATCH_SIZE_VF=$((BATCH_SIZE_VF / 2))
BSPLINE_METHOD='matrix'

# Velocity field parameters
# With 20-frame subset, dt=0.053 ≈ original 21-frame dt=0.05
VFIELD_RES_6_LRPW=1e-3
VFIELD_RES_6_WUS=1000
VFIELD_RES_6_TIMEDELTA=0.05

VFIELD_RES_12_LRPW=1e-3
VFIELD_RES_12_WUS=200
VFIELD_RES_12_TIMEDELTA=0.05

# Data paths
DATA0="$DATA_DIR/transforms_00.json"
DATA1="$DATA_DIR/transforms_${LAST_IDX_PAD}.json"
DATAALL="$DATA_DIR/transforms_00_to_${LAST_IDX_PAD}.json"
GRID0="$DATA_DIR/dendrite_00.yaml"
GRID1="$DATA_DIR/dendrite_${LAST_IDX_PAD}.yaml"

echo "=========================================="
echo "Dendrite 4D Reconstruction Training"
echo "=========================================="
echo "Project root:    $PROJECT_ROOT"
echo "Data directory:  $DATA_DIR"
echo "Output:          $OUTPUT_DIR"
echo "Num frames:      $NUM_FRAMES"
echo "Steps per stage: $NUMSTEPS"
echo ""

# Verify required files
echo "Checking required files..."
REQUIRED_FILES=("$DATA0" "$DATA1" "$DATAALL" "$GRID0" "$GRID1")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ Missing: $file"
        echo "  Run prepare_dendrite_data.py first"
        exit 1
    fi
    echo "  ✓ $(basename "$file")"
done

# ===== Step 1: Canonical forward =====
echo ""
echo "=========================================="
echo "Step 1: Training canonical volume forward (t=0)"
echo "=========================================="
python "$PROJECT_ROOT/nerfstudio/nerfstudio/scripts/train.py" nerf_xray \
    --data "$DATA0" \
    --output_dir "$OUTPUT_DIR" \
    --logging.local-writer.max-log-size 10 \
    --pipeline.volumetric_supervision False \
    --pipeline.datamanager.volume_grid_file "$GRID0" \
    --pipeline.datamanager.train_num_rays_per_batch $BATCH_SIZE \
    --pipeline.datamanager.eval_num_rays_per_batch $EVAL_BATCH_SIZE \
    --pipeline.model.eval_num_rays_per_chunk $EVAL_BATCH_SIZE \
    --pipeline.model.flat_field_trainable False \
    --max-num-iterations $((NUMSTEPS + 1)) \
    --optimizers.fields.scheduler.lr_pre_warmup 1e-8 \
    --optimizers.fields.scheduler.lr_final 1e-4 \
    --optimizers.fields.scheduler.warmup_steps 50 \
    --optimizers.fields.scheduler.steady_steps $((NUMSTEPS - 500)) \
    --optimizers.fields.scheduler.max_steps $NUMSTEPS \
    --timestamp "canonical_F" \
    multi-camera-dataparser --downscale-factors.val $DOWNSCALE_FACTOR --downscale-factors.test $DOWNSCALE_FACTOR || exit 1

# ===== Step 2: Canonical backward =====
echo ""
echo "=========================================="
echo "Step 2: Training canonical volume backward (t=${LAST_IDX_PAD})"
echo "=========================================="
python "$PROJECT_ROOT/nerfstudio/nerfstudio/scripts/train.py" nerf_xray \
    --data "$DATA1" \
    --output_dir "$OUTPUT_DIR" \
    --logging.local-writer.max-log-size 10 \
    --pipeline.volumetric_supervision False \
    --pipeline.datamanager.volume_grid_file "$GRID1" \
    --pipeline.datamanager.train_num_rays_per_batch $BATCH_SIZE \
    --pipeline.datamanager.eval_num_rays_per_batch $EVAL_BATCH_SIZE \
    --pipeline.model.eval_num_rays_per_chunk $EVAL_BATCH_SIZE \
    --pipeline.model.flat_field_trainable False \
    --max-num-iterations $((NUMSTEPS + 1)) \
    --optimizers.fields.scheduler.lr_pre_warmup 1e-8 \
    --optimizers.fields.scheduler.lr_final 1e-4 \
    --optimizers.fields.scheduler.warmup_steps 50 \
    --optimizers.fields.scheduler.steady_steps $((NUMSTEPS - 500)) \
    --optimizers.fields.scheduler.max_steps $NUMSTEPS \
    --timestamp "canonical_B" \
    multi-camera-dataparser --downscale-factors.val $DOWNSCALE_FACTOR --downscale-factors.test $DOWNSCALE_FACTOR || exit 1

# ===== Step 3: Velocity field resolution 6 =====
echo ""
echo "=========================================="
echo "Step 3: Training velocity field (resolution 6)"
echo "=========================================="

N1=6
STEPS=$NUMSTEPS
PADSTEPS=$(printf '%09d' $STEPS)

mkdir -p "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}/nerfstudio_models"

if [ ! -f "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}/nerfstudio_models/step-$PADSTEPS.ckpt" ]; then
    echo "Combining forward and backward checkpoints..."
    python "$PROJECT_ROOT/nerfstudio-xray/nerf-xray/nerf_xray/combine_forward_backward_checkpoints.py" \
        --fwd_ckpt "$OUTPUT_DIR/$DSET/nerf_xray/canonical_F/nerfstudio_models/step-$PADSTEPS.ckpt" \
        --bwd_ckpt "$OUTPUT_DIR/$DSET/nerf_xray/canonical_B/nerfstudio_models/step-$PADSTEPS.ckpt" \
        --out_fn "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}/nerfstudio_models/step-$PADSTEPS.ckpt" || exit 1
    LOAD_OPTIMIZER=False
else
    LOAD_OPTIMIZER=True
fi

python "$PROJECT_ROOT/nerfstudio/nerfstudio/scripts/train.py" xray_vfield \
    --data "$DATAALL" \
    --output_dir "$OUTPUT_DIR" \
    --max-num-iterations $NUMSTEPS \
    --steps_per_eval_image 500 \
    --steps_per_save 250 \
    --logging.local-writer.max-log-size 10 \
    --load-checkpoint "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}/nerfstudio_models/step-$PADSTEPS.ckpt" \
    --load-optimizer $LOAD_OPTIMIZER \
    --pipeline.volumetric_supervision True \
    --pipeline.volumetric_supervision_coefficient 1e-4 \
    --pipeline.volumetric_supervision_start_step $(($NUMSTEPS+1000)) \
    --pipeline.datamanager.init_volume_grid_file "$GRID0" \
    --pipeline.datamanager.final_volume_grid_file "$GRID1" \
    --pipeline.model.deformation_field.num_control_points $N1 $N1 $N1 \
    --pipeline.model.deformation_field.weight_nn_width $WEIGHT_NN_WIDTH \
    --pipeline.model.deformation_field.weight_nn_bias True \
    --pipeline.model.deformation_field.weight_nn_gain 1.0 \
    --pipeline.model.deformation_field.timedelta $VFIELD_RES_6_TIMEDELTA \
    --pipeline.model.deformation_field.displacement_method $BSPLINE_METHOD \
    --pipeline.model.flat_field_trainable False \
    --pipeline.model.train_field_weighing False \
    --pipeline.datamanager.train_num_rays_per_batch $BATCH_SIZE_VF \
    --pipeline.datamanager.eval_num_rays_per_batch $EVAL_BATCH_SIZE_VF \
    --pipeline.model.eval_num_rays_per_chunk $EVAL_BATCH_SIZE_VF \
    --pipeline.model.num_nerf_samples_per_ray $VF_NUM_SAMPLES_PER_RAY \
    --pipeline.model.distortion_loss_mult 0.0 \
    --pipeline.model.interlevel_loss_mult 0.0 \
    --pipeline.model.disable_mixing True \
    --pipeline.density_mismatch_start_step -1 \
    --pipeline.density_mismatch_coefficient 1e-3 \
    --optimizers.fields.optimizer.lr 1e-4 \
    --optimizers.fields.optimizer.weight_decay 1e-1 \
    --optimizers.fields.scheduler.lr_pre_warmup $VFIELD_RES_6_LRPW \
    --optimizers.fields.scheduler.lr_final 1e-6 \
    --optimizers.fields.scheduler.warmup_steps $VFIELD_RES_6_WUS \
    --optimizers.fields.scheduler.steady_steps $(($NUMSTEPS - 1000)) \
    --optimizers.fields.scheduler.max_steps $NUMSTEPS \
    --timestamp "vel_${N1}" \
    --machine.seed 40 \
    multi-camera-dataparser --downscale-factors.val $DOWNSCALE_FACTOR --downscale-factors.test $DOWNSCALE_FACTOR || exit 1

# ===== Step 4: Velocity field resolution 12 =====
echo ""
echo "=========================================="
echo "Step 4: Training velocity field (resolution 12)"
echo "=========================================="

N0=6
N1=12
STEPS=$((NUMSTEPS+NUMSTEPS))
PADSTEPS=$(printf '%09d' $STEPS)

mkdir -p "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}/nerfstudio_models"

if [ ! -f "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}/nerfstudio_models/step-$PADSTEPS.ckpt" ]; then
    echo "Refining velocity field from resolution $N0 to $N1..."
    python "$PROJECT_ROOT/nerfstudio-xray/nerf-xray/nerf_xray/refine_vfield.py" \
        --load-config "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N0}/config.yml" \
        --new-resolution $N1 \
        --new-nn-width $WEIGHT_NN_WIDTH \
        --out-path "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}/nerfstudio_models/step-$PADSTEPS.ckpt" || exit 1
    LOAD_OPTIMIZER=False
else
    LOAD_OPTIMIZER=False
fi

python "$PROJECT_ROOT/nerfstudio/nerfstudio/scripts/train.py" xray_vfield \
    --data "$DATAALL" \
    --output_dir "$OUTPUT_DIR" \
    --max-num-iterations $NUMSTEPS \
    --steps_per_eval_image 500 \
    --steps_per_save 250 \
    --logging.local-writer.max-log-size 10 \
    --load-checkpoint "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}/nerfstudio_models/step-$PADSTEPS.ckpt" \
    --load-optimizer $LOAD_OPTIMIZER \
    --pipeline.volumetric_supervision True \
    --pipeline.volumetric_supervision_coefficient 1e-4 \
    --pipeline.volumetric_supervision_start_step $(($NUMSTEPS+1000)) \
    --pipeline.datamanager.init_volume_grid_file "$GRID0" \
    --pipeline.datamanager.final_volume_grid_file "$GRID1" \
    --pipeline.model.deformation_field.num_control_points $N1 $N1 $N1 \
    --pipeline.model.deformation_field.weight_nn_width $WEIGHT_NN_WIDTH \
    --pipeline.model.deformation_field.weight_nn_bias True \
    --pipeline.model.deformation_field.weight_nn_gain 1.0 \
    --pipeline.model.deformation_field.timedelta $VFIELD_RES_12_TIMEDELTA \
    --pipeline.model.deformation_field.displacement_method $BSPLINE_METHOD \
    --pipeline.model.flat_field_trainable False \
    --pipeline.model.train_field_weighing False \
    --pipeline.datamanager.train_num_rays_per_batch $BATCH_SIZE_VF \
    --pipeline.datamanager.eval_num_rays_per_batch $EVAL_BATCH_SIZE_VF \
    --pipeline.model.eval_num_rays_per_chunk $EVAL_BATCH_SIZE_VF \
    --pipeline.model.num_nerf_samples_per_ray $VF_NUM_SAMPLES_PER_RAY \
    --pipeline.model.distortion_loss_mult 0.0 \
    --pipeline.model.interlevel_loss_mult 0.0 \
    --pipeline.model.disable_mixing True \
    --pipeline.density_mismatch_start_step -1 \
    --pipeline.density_mismatch_coefficient 1e-3 \
    --optimizers.fields.optimizer.lr 1e-4 \
    --optimizers.fields.optimizer.weight_decay 1e-1 \
    --optimizers.fields.scheduler.lr_pre_warmup $VFIELD_RES_12_LRPW \
    --optimizers.fields.scheduler.lr_final 1e-6 \
    --optimizers.fields.scheduler.warmup_steps $VFIELD_RES_12_WUS \
    --optimizers.fields.scheduler.steady_steps $(($NUMSTEPS - 1000)) \
    --optimizers.fields.scheduler.max_steps $NUMSTEPS \
    --timestamp "vel_${N1}" \
    --machine.seed 40 \
    multi-camera-dataparser --downscale-factors.val $DOWNSCALE_FACTOR --downscale-factors.test $DOWNSCALE_FACTOR || exit 1

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "=========================================="
echo "Canonical models:"
echo "  - Forward: $OUTPUT_DIR/$DSET/nerf_xray/canonical_F"
echo "  - Backward: $OUTPUT_DIR/$DSET/nerf_xray/canonical_B"
echo "Velocity field models:"
echo "  - Resolution 6:  $OUTPUT_DIR/$DSET/xray_vfield/vel_6"
echo "  - Resolution 12: $OUTPUT_DIR/$DSET/xray_vfield/vel_12"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python scripts/eval_dendrite.py compute-psnr-gt ..."
echo "  2. Export:   python scripts/export_dendrite.py volume-sequence ..."
echo "  3. Full cascade: adapt submit.sh for 6→9→15→27→51 + mixing"
echo ""
