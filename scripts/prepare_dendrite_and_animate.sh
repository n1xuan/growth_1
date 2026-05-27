#!/bin/bash
# Prepare dendrite data and create verification GIFs.
#
# This is the dendrite equivalent of generate_and_animate.sh in neural_xray.
# In neural_xray's experimental workflow, process_data.ps1 handles raw XCT
# data conversion. Here we handle tiff volumes → projections → transforms.
#
# Usage:
#   ./scripts/prepare_dendrite_and_animate.sh <tiff_data_dir> [num_frames] [output_dir]
#
# Example:
#   ./scripts/prepare_dendrite_and_animate.sh /data/synchrotron/dendrite_run01 20

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -z "$1" ]; then
    echo "Usage: $0 <tiff_data_dir> [num_frames] [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  tiff_data_dir    Directory containing tiff frames (sorted = time order)"
    echo "  num_frames       Number of frames to process (default: 20)"
    echo "  output_dir       Output directory (default: data/dendrite)"
    echo ""
    echo "Example:"
    echo "  $0 /data/synchrotron/dendrite_run01 20 data/dendrite"
    exit 1
fi

TIFF_DIR="$1"
NUM_FRAMES="${2:-20}"
DATA_DIR="${3:-$PROJECT_ROOT/data/dendrite}"

ROTATION_GIF="$DATA_DIR/rotation_animation.gif"
DEFORMATION_GIF="$DATA_DIR/deformation_animation.gif"

echo "=========================================="
echo "Step 1: Preparing dendrite data"
echo "=========================================="
echo "  Tiff directory: $TIFF_DIR"
echo "  Num frames:     $NUM_FRAMES"
echo "  Output:         $DATA_DIR"
echo ""

python3 "$SCRIPT_DIR/prepare_dendrite_data.py" \
    --data-dir "$TIFF_DIR" \
    --output-dir "$DATA_DIR" \
    --num-frames "$NUM_FRAMES" \
    --num-train-angles 16 \
    --num-intermediate-angles 2

if [ $? -ne 0 ]; then
    echo "Error: Data preparation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Creating rotation animation GIF"
echo "=========================================="

if python3 -c "import PIL" 2>/dev/null; then
    python3 << 'EOF'
from pathlib import Path
from PIL import Image
import sys, os

images_dir = Path(os.environ.get("IMAGES_DIR", ""))
output_gif = Path(os.environ.get("OUTPUT_GIF", ""))

train_images = sorted(images_dir.glob("train_*.png"))
if not train_images:
    print("No train images found, skipping rotation GIF")
    sys.exit(0)

frames = [Image.open(str(p)).copy() for p in train_images]
frames[0].save(str(output_gif), save_all=True,
               append_images=frames[1:], duration=200, loop=0)
print(f"✓ Rotation GIF: {output_gif} ({len(frames)} angles)")
EOF
else
    echo "Pillow not found, skipping GIF (pip install Pillow)"
fi

echo ""
echo "=========================================="
echo "Step 3: Creating deformation animation GIF"
echo "=========================================="

if python3 -c "import PIL" 2>/dev/null; then
    IMAGES_DIR="$DATA_DIR/images_00" OUTPUT_GIF="$ROTATION_GIF" \
    python3 << EOF
import sys
from pathlib import Path
from PIL import Image

data_dir = Path("$DATA_DIR")
output_gif = Path("$DEFORMATION_GIF")
num_frames = $NUM_FRAMES

images = []
for i in range(num_frames):
    img_path = data_dir / f"images_{i:02d}" / "eval_00.png"
    if img_path.exists():
        images.append(str(img_path))

if not images:
    print("No eval images found, skipping deformation GIF")
    sys.exit(0)

frames = [Image.open(p).copy() for p in images]
frames[0].save(str(output_gif), save_all=True,
               append_images=frames[1:], duration=100, loop=0)
print(f"✓ Deformation GIF: {output_gif} ({len(frames)} timesteps)")
EOF
else
    echo "Pillow not found, skipping GIF"
fi

echo ""
echo "=========================================="
echo "Step 4: Downsampling eval images"
echo "=========================================="

RESIZE_SCRIPT="$PROJECT_ROOT/nerf_data/scripts/resize_for_eval.py"

if [ -f "$RESIZE_SCRIPT" ]; then
    LAST_IDX=$((NUM_FRAMES - 1))
    for i in $(seq 0 $LAST_IDX); do
        IMAGE_DIR="$DATA_DIR/images_$(printf "%02d" $i)"
        if [ -d "$IMAGE_DIR" ]; then
            python3 "$RESIZE_SCRIPT" --folder "$IMAGE_DIR" --downscale-factor 2
        fi
    done
    echo "✓ Downsampled eval images"
else
    echo "Resize script not found at $RESIZE_SCRIPT, skipping"
fi

echo ""
echo "=========================================="
echo "✓ Complete!"
echo "=========================================="
echo "Prepared data:     $DATA_DIR"
echo "Rotation GIF:      $ROTATION_GIF"
echo "Deformation GIF:   $DEFORMATION_GIF"
echo ""
