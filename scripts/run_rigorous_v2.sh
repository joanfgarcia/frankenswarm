#!/bin/bash
# ═══════════════════════════════════════════════════════════
# RIGOR SUITE v2 — Fixed ablation + random baseline
# Bug fixes:
#   1. Use train_neurogenesis.py (not train_survival.py)
#   2. Use correct config fields (growth.init_width, growth.max_width, growth.patience)
#   3. Random baseline through MinimalWorld with full mechanics
# ═══════════════════════════════════════════════════════════

set -e
cd /home/joan/Documents/IA/frankenswarm
export PYTHONPATH=.

SEEDS=(42 123 456 789 1337)
RESULTS_DIR="storage/experiments/RIGOR_v2"
mkdir -p "$RESULTS_DIR"

echo "═══════════════════════════════════════════════════════"
echo "🔬 RIGOR SUITE v2 — $(date)"
echo "   Seeds: ${SEEDS[*]}"
echo "   Output: $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════"

# ── 1. ABLATION: A (small), B (big), C (growth) — 5 seeds each ──
echo ""
echo "═══ PHASE 1: ABLATION via train_neurogenesis.py (15 runs) ═══"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed $SEED ---"
    
    # Config A: Small (128), NO growth (patience=99999, max_width=128)
    echo "  [A] Small, no growth..."
    cat > /tmp/rigor_A_${SEED}.json << ENDOFCFG
{
    "experiment_id": "RIGOR_v2_A_s${SEED}",
    "seed": $SEED,
    "pretrained_from": "storage/experiments/EXP_036_pretrained/best_agent.pt",
    "model": {"hidden_dim": 256, "num_layers": 3, "use_pos_embedding": true},
    "resonance": {"max_resonance_steps": 5, "n_think": 2, "n_verify": 2},
    "emotion": {"mode": "first_only", "dim": 64},
    "metacognition": {"enabled": true, "threshold": 0.85},
    "world": {"max_ticks": 200},
    "training": {"episodes": 500, "lr": 0.0005, "gamma": 0.97, "grad_clip": 1.0, "entropy_bonus": 0.05, "update_interval": 16},
    "growth": {"convergence_threshold": 0.80, "patience": 99999, "factor": 1.5, "max_width": 128}
}
ENDOFCFG
    .venv/bin/python src/bitnet/train_neurogenesis.py --config /tmp/rigor_A_${SEED}.json 2>&1 | tail -10 | tee -a "$RESULTS_DIR/ablation_A.log"
    
    # Config B: Big (648), NO growth (init_width=648, patience=99999, max_width=648)
    echo "  [B] Big, no growth..."
    cat > /tmp/rigor_B_${SEED}.json << ENDOFCFG
{
    "experiment_id": "RIGOR_v2_B_s${SEED}",
    "seed": $SEED,
    "pretrained_from": "storage/experiments/EXP_036_pretrained/best_agent.pt",
    "model": {"hidden_dim": 256, "num_layers": 3, "use_pos_embedding": true},
    "resonance": {"max_resonance_steps": 5, "n_think": 2, "n_verify": 2},
    "emotion": {"mode": "first_only", "dim": 64},
    "metacognition": {"enabled": true, "threshold": 0.85},
    "world": {"max_ticks": 200},
    "training": {"episodes": 500, "lr": 0.0005, "gamma": 0.97, "grad_clip": 1.0, "entropy_bonus": 0.05, "update_interval": 16},
    "growth": {"convergence_threshold": 0.80, "patience": 99999, "factor": 1.5, "max_width": 648, "init_width": 648}
}
ENDOFCFG
    .venv/bin/python src/bitnet/train_neurogenesis.py --config /tmp/rigor_B_${SEED}.json 2>&1 | tail -10 | tee -a "$RESULTS_DIR/ablation_B.log"
    
    # Config C: Small→Big, GROWTH ENABLED
    echo "  [C] Growth..."
    cat > /tmp/rigor_C_${SEED}.json << ENDOFCFG
{
    "experiment_id": "RIGOR_v2_C_s${SEED}",
    "seed": $SEED,
    "pretrained_from": "storage/experiments/EXP_036_pretrained/best_agent.pt",
    "model": {"hidden_dim": 256, "num_layers": 3, "use_pos_embedding": true},
    "resonance": {"max_resonance_steps": 5, "n_think": 2, "n_verify": 2},
    "emotion": {"mode": "first_only", "dim": 64},
    "metacognition": {"enabled": true, "threshold": 0.85},
    "world": {"max_ticks": 200},
    "training": {"episodes": 500, "lr": 0.0005, "gamma": 0.97, "grad_clip": 1.0, "entropy_bonus": 0.05, "update_interval": 16},
    "growth": {"convergence_threshold": 0.80, "patience": 50, "factor": 1.5, "max_width": 648}
}
ENDOFCFG
    .venv/bin/python src/bitnet/train_neurogenesis.py --config /tmp/rigor_C_${SEED}.json 2>&1 | tail -10 | tee -a "$RESULTS_DIR/ablation_C.log"
done

# ── 2. RANDOM BASELINE via train_neurogenesis infrastructure ──
echo ""
echo "═══ PHASE 2: RANDOM BASELINE (5 seeds, proper MinimalWorld) ═══"

.venv/bin/python - << 'PYEOF'
import numpy as np
from src.bitnet.minimal_world import MinimalWorld, ACTIONS

seeds = [42, 123, 456, 789, 1337]
for seed in seeds:
    np.random.seed(seed)
    results = []
    for ep in range(500):
        world = MinimalWorld(seed=seed + ep)
        state = world.reset()
        tick = 0
        while state.alive and tick < 200:
            tick += 1
            action = np.random.choice(ACTIONS)
            result = world.act(action)
        results.append(state.tick)
    best = max(results)
    avg100 = np.mean(results[-100:])
    avg_all = np.mean(results)
    print(f"RANDOM seed={seed} | Best: {best} | Avg100: {avg100:.1f} | AvgAll: {avg_all:.1f}")
PYEOF

echo ""
echo "═══════════════════════════════════════════════════════"
echo "🏁 RIGOR SUITE v2 COMPLETE — $(date)"
echo "   Results in: $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════"
