#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# SUPERSEDED (2026-07-27) → usa ./scripts/train_school.sh
#
# Este wrapper corre el currículo ENTERO en una sola invocación con
# MemoryMax=10G, dos decisiones que envejecieron mal: 10G despierta al OOM
# killer con el modelo a 896 dim, y sin trocear por épocas una interrupción
# cuesta todo lo no guardado. El sustituto trocea época a época, sube el
# límite a 16G e inhibe la suspensión (que mata el contexto CUDA).
# Ver docs/TRAINING_BIT.md. Se conserva solo por compatibilidad.
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
systemd-run --user --scope -p MemoryMax=10G .venv/bin/python src/bitnet/training/train_sovereign_school.py --batch_size 64

echo "Training completed or paused successfully."
