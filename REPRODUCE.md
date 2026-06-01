# Reproducing the Paper Results

## Prerequisites

- Python 3.13+
- PyTorch 2.x
- GPU with ≥6GB VRAM (tested on NVIDIA RTX 5070 Laptop, 8GB)
- RAM: ≥16GB

## Setup

```bash
git clone <repository-url>
cd frankenswarm
python -m venv .venv
source .venv/bin/activate
pip install torch numpy
```

## Quick Start: Full Ablation (≈45 minutes)

This runs the complete multi-seed ablation study from §5.5 of the paper:

```bash
chmod +x scripts/run_rigorous_v3.sh
./scripts/run_rigorous_v3.sh
```

This produces 15 experiments (3 configs × 5 seeds) with automatic statistical analysis including Bootstrap 95% CI.

## Step-by-Step Reproduction

### 1. Pretraining — Glyph Embeddings + Emotion + Metacognition (≈12 min)

This creates the frozen backbone used by all survival experiments:

```bash
PYTHONPATH=. python src/bitnet/train_emotional_resonance.py \
    --config configs/experiments/EXP_036_piaget.json
```

**Expected output:** 100% concept accuracy, convergence gap ≈0.116

### 2. Survival — Action Head (≈3 min)

```bash
PYTHONPATH=. python src/bitnet/train_survival.py \
    --config configs/experiments/EXP_039_survival_full.json
```

**Expected output:** Best ≈115 ticks (seed 42)

### 3. Neurogenesis — Pain-Driven Growth (≈3 min)

```bash
PYTHONPATH=. python src/bitnet/train_neurogenesis.py \
    --config configs/experiments/EXP_040_ablation_small.json
```

**Expected output:** 4-6 growth events, final width ≈648-1458, best ≈163 ticks (seed 42)

### 4. Multi-Seed Ablation (≈45 min total)

Three configurations, each run with 5 seeds (42, 123, 456, 789, 1337):

| Config | Script | Config Files | What It Tests |
|---|---|---|---|
| A (Small, fixed) | `train_survival.py` | `RIGOR_v3_A_s*.json` | 128-wide head, no growth |
| B (Big, fixed) | `train_survival.py` | `RIGOR_v3_B_s*.json` | 648-wide head, no growth |
| C (Grown) | `train_neurogenesis.py` | `RIGOR_v3_C_s*.json` | 128→648 via pain-driven growth |

```bash
# Individual run example:
PYTHONPATH=. python src/bitnet/train_survival.py \
    --config configs/experiments/RIGOR_v3_A_s42.json

PYTHONPATH=. python src/bitnet/train_neurogenesis.py \
    --config configs/experiments/RIGOR_v3_C_s42.json
```

> **IMPORTANT:** Config C uses `train_neurogenesis.py` (which contains the Net2WiderNet growth loop), NOT `train_survival.py`.

### 5. Neuroplasticity — Frozen vs Unfrozen (≈30 min)

Requires `train_dual.py` which supports the `unfreeze_backbone` parameter:

```bash
# Frozen backbone
PYTHONPATH=. python src/bitnet/train_dual.py \
    --config configs/experiments/RIGOR_frozen_s42.json

# Unfrozen backbone (attention layers only)
PYTHONPATH=. python src/bitnet/train_dual.py \
    --config configs/experiments/RIGOR_unfrozen_s42.json
```

### 6. Communication Experiments (≈40 min per config)

```bash
PYTHONPATH=. python src/bitnet/train_arena.py \
    --config configs/experiments/EXP_047_arena_graduated.json
```

## Reading Results

All experiments write telemetry to `storage/experiments/<experiment_id>/telemetry.jsonl`. Each line is a JSON object:

```bash
# Last line = final metrics
tail -1 storage/experiments/RIGOR_v3_A_s42/telemetry.jsonl | python -m json.tool

# Extract best survival per experiment
python3 -c "
import json
path = 'storage/experiments/RIGOR_v3_A_s42/telemetry.jsonl'
data = [json.loads(l)['fitness'][0] for l in open(path) if '\"epoch\"' in l]
print(f'Best: {max(data)}, Avg(last 100): {sum(data[-100:])/100:.1f}')
"
```

## Hardware Notes

- All experiments were developed and tested on a single NVIDIA RTX 5070 Laptop GPU (8GB VRAM)
- The ternary model (2.4M parameters) occupies ≈586KB when compressed to 2-bit
- Training uses ≈2-3GB VRAM; inference uses <1GB
- No multi-GPU support is needed or implemented

## Determinism

All experiments use fixed seeds for numpy, torch, and Python's random module. Results should be reproducible within floating-point tolerance. If you observe different results, check:
1. PyTorch version (we used 2.x)
2. CUDA version and GPU architecture
3. That `PYTHONPATH=.` is set correctly
