#!/usr/bin/env bash
# rgbd2ply — one-command environment setup
# ============================================================================
# Usage:
#   bash setup.sh                           # interactive setup
#   bash setup.sh --sam3 /path/to/sam3      # copy SAM3 from existing install
#
# This script creates the venv, installs python dependencies, and sets up
# SAM3 (the only external dependency not bundled in this repo).
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "=== rgbd2ply setup ==="
echo "  root: $ROOT"

# ---- Python virtual environment ----
if [ -d "$ROOT/venv" ]; then
    echo "  venv already exists, skipping creation"
else
    echo "  creating venv..."
    python3 -m venv "$ROOT/venv"
fi

source "$ROOT/venv/bin/activate"

# ---- Install core dependencies ----
echo "  installing python packages..."
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q numpy scipy pyyaml opencv-python-headless einops pycocotools psutil rosbags

# ---- Install rgbd2ply ----
pip install -q -e "$ROOT"

# ---- SAM3 ----
SAM3_DIR="$ROOT/deps/sam3"

# Check for --sam3 flag
if [[ "${1:-}" == "--sam3" ]]; then
    SAM3_SRC="${2:-}"
    if [ -n "$SAM3_SRC" ] && [ -d "$SAM3_SRC" ]; then
        echo "  copying SAM3 from $SAM3_SRC ..."
        cp -r "$SAM3_SRC" "$SAM3_DIR"
    fi
fi

if [ -d "$SAM3_DIR" ]; then
    echo "  SAM3 found at deps/sam3/"
else
    echo ""
    echo "  [ACTION REQUIRED] SAM3 not found."
    echo "  Obtain SAM3 by one of:"
    echo "    a) Copy from existing machine:  bash setup.sh --sam3 /path/to/sam3"
    echo "    b) Clone from GitHub:"
    echo "       git clone https://github.com/facebookresearch/sam3.git deps/sam3"
    echo "       cd deps/sam3 && pip install -e ."
    echo "       # Download checkpoint from HuggingFace:"
    echo "       # https://huggingface.co/facebook/sam3"
    echo "  Then re-run: bash setup.sh"
    exit 1
fi

# Install SAM3
pip install -q -e "$SAM3_DIR" 2>/dev/null || true
# SAM3 may downgrade numpy; restore compatible version
pip install -q "numpy>=2.0,<2.5" 2>/dev/null || true

# ---- Verify ----
echo ""
echo "=== Verification ==="
python -c "
from rgbd2ply.config import cfg
from pathlib import Path
ok = 0; total = 4
for name, path, typ in [
    ('sam3_repo', cfg.paths.sam3_repo, 'dir'),
    ('sam3_checkpoint', cfg.paths.sam3_checkpoint, 'file'),
    ('extrinsic', cfg.paths.extrinsic, 'file'),
    ('registry', cfg.paths.registry, 'file'),
]:
    exists = Path(path).is_dir() if typ == 'dir' else Path(path).is_file()
    print(f'  {\"✓\" if exists else \"✗\"} {name}: {path}')
    if exists: ok += 1
print(f'  {ok}/{total} paths valid')
if ok < total:
    print('  → Edit rgbd2ply/config.yaml to fix missing paths')
"

echo ""
echo "=== Setup complete ==="
echo "  source venv/bin/activate"
echo "  rgbd2ply --version"
echo "  rgbd2ply config"
