#!/bin/bash
# ═══════════════════════════════════════════════════════════
# RIGOR v3 — Reproducibility Suite for Paper
# 
# Runs the complete ablation study:
#   A: Small action head (128), no growth  → train_survival.py
#   B: Big action head (648), no growth    → train_survival.py
#   C: Small→Big (128→648), growth enabled → train_neurogenesis.py
#
# Each config × 5 seeds = 15 runs total
# Expected time: ~45 minutes on consumer GPU (<8GB VRAM)
#
# Usage:
#   chmod +x scripts/run_rigorous_v3.sh
#   ./scripts/run_rigorous_v3.sh
# ═══════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

SEEDS=(42 123 456 789 1337)

echo "═══════════════════════════════════════════════════════"
echo "🔬 RIGOR v3 SUITE — $(date)"
echo "   Seeds: ${SEEDS[*]}"
echo "   GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo "═══════════════════════════════════════════════════════"

# ── PHASE 1: Config A — Small head, no growth ──
echo ""
echo "═══ PHASE 1: Config A (width=128, no growth) ═══"

for SEED in "${SEEDS[@]}"; do
    CONFIG="configs/experiments/RIGOR_v3_A_s${SEED}.json"
    echo "  [A] Seed ${SEED}..."
    .venv/bin/python src/bitnet/train_survival.py --config "$CONFIG" 2>&1 | tail -3
done

# ── PHASE 2: Config B — Big head, no growth ──
echo ""
echo "═══ PHASE 2: Config B (width=648, no growth) ═══"

for SEED in "${SEEDS[@]}"; do
    CONFIG="configs/experiments/RIGOR_v3_B_s${SEED}.json"
    echo "  [B] Seed ${SEED}..."
    .venv/bin/python src/bitnet/train_survival.py --config "$CONFIG" 2>&1 | tail -3
done

# ── PHASE 3: Config C — Small head, growth enabled ──
# IMPORTANT: Uses train_neurogenesis.py which has the Net2WiderNet growth loop
echo ""
echo "═══ PHASE 3: Config C (width=128→648, growth) ═══"

for SEED in "${SEEDS[@]}"; do
    CONFIG="configs/experiments/RIGOR_v3_C_s${SEED}.json"
    echo "  [C] Seed ${SEED}..."
    .venv/bin/python src/bitnet/train_neurogenesis.py --config "$CONFIG" 2>&1 | tail -5
done

# ── PHASE 4: Statistical Analysis ──
echo ""
echo "═══ PHASE 4: ANALYSIS ═══"

.venv/bin/python3 -c "
import json, numpy as np

np.random.seed(42)
n_boot = 10000

results = {}
for cfg in ['A', 'B', 'C']:
    bests, avgs = [], []
    for s in [42, 123, 456, 789, 1337]:
        path = f'storage/experiments/RIGOR_v3_{cfg}_s{s}/telemetry.jsonl'
        data = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d.get('type') == 'epoch':
                    data.append(d['fitness'][0])
        bests.append(max(data))
        avgs.append(np.mean(data[-100:]))
    results[cfg] = {'best': np.array(bests), 'avg': np.array(avgs)}

print('╔═══════════════════════════════════════════════════════════════╗')
print('║                  ABLATION RESULTS (5 seeds)                  ║')
print('╠═══════════╦══════════╦═══════════════════╦═══════════════════╣')
print('║ Config    ║ Params   ║ Best (mean±sd)    ║ Avg100 (mean±sd)  ║')
print('╠═══════════╬══════════╬═══════════════════╬═══════════════════╣')
for cfg, label, params in [('A','Small 128','33,670'), ('B','Big 648','170,430'), ('C','Grown 128→648','33K→170K')]:
    b = results[cfg]['best']
    a = results[cfg]['avg']
    print(f'║ {label:9s} ║ {params:8s} ║ {np.mean(b):5.1f} ± {np.std(b):4.1f}    ║ {np.mean(a):5.1f} ± {np.std(a):4.1f}    ║')
print('╚═══════════╩══════════╩═══════════════════╩═══════════════════╝')

# Bootstrap CI for C-B difference
c_best = results['C']['best']
b_best = results['B']['best']
diff_boot = []
for _ in range(n_boot):
    ci = np.random.choice(range(len(c_best)), size=len(c_best), replace=True)
    bi = np.random.choice(range(len(b_best)), size=len(b_best), replace=True)
    diff_boot.append(np.mean(c_best[ci]) - np.mean(b_best[bi]))
ci_diff = np.percentile(diff_boot, [2.5, 97.5])
print(f'')
print(f'Bootstrap 95% CI (C-B): [{ci_diff[0]:.1f}, {ci_diff[1]:.1f}]')
print(f'CI includes 0: {ci_diff[0] <= 0 <= ci_diff[1]}')
print(f'')
print(f'A vs B (capacity alone): Best diff = {np.mean(results[\"A\"][\"best\"]) - np.mean(results[\"B\"][\"best\"]):+.1f}')
print(f'C vs B (growth vs born big): Best diff = {np.mean(results[\"C\"][\"best\"]) - np.mean(results[\"B\"][\"best\"]):+.1f}')
"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "🏁 RIGOR v3 COMPLETE — $(date)"
echo "═══════════════════════════════════════════════════════"
