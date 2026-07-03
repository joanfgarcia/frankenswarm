#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# Wrapper to run sovereign school training on GPU by temporarily freeing VRAM from redpill-llm.
#
# Usage:
#   chmod +x scripts/run_school_gpu.sh
#   ./scripts/run_school_gpu.sh
# ═══════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "Stopping redpill-llm user service to free GPU VRAM..."
systemctl --user stop redpill-llm

# Trap SIGINT, SIGTERM, and EXIT to ensure we always restore the redpill-llm service
cleanup() {
    echo "Restoring redpill-llm user service..."
    systemctl --user start redpill-llm
}
trap cleanup EXIT SIGINT SIGTERM

echo "Starting school training on GPU under cgroups..."
systemd-run --user --scope -p MemoryMax=10G .venv/bin/python src/bitnet/train_sovereign_school.py --test_mock --batch_size 64

echo "Training completed or paused successfully."
