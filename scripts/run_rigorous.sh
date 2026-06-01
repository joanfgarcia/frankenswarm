#!/bin/bash
# ═══════════════════════════════════════════════════════════
# RIGOR SUITE — Multi-seed experiments for paper
# Run overnight, analyze in the morning
# ═══════════════════════════════════════════════════════════

set -e
cd /home/joan/Documents/IA/frankenswarm
export PYTHONPATH=.

SEEDS=(42 123 456 789 1337)
RESULTS_DIR="storage/experiments/RIGOR"
mkdir -p "$RESULTS_DIR"

echo "═══════════════════════════════════════════════════════"
echo "🔬 RIGOR SUITE — $(date)"
echo "   Seeds: ${SEEDS[*]}"
echo "   Output: $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════"

# ── 1. ABLATION: A (small), B (big), C (growth) — 5 seeds each ──
echo ""
echo "═══ PHASE 1: ABLATION (15 runs) ═══"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed $SEED ---"
    
    # Config A: Small (128), no growth
    echo "  [A] Small, no growth..."
    .venv/bin/python -c "
import json
cfg = {
    'experiment_id': 'RIGOR_ablation_A_s${SEED}',
    'seed': $SEED,
    'model': {'hidden_dim': 256, 'num_layers': 3, 'use_pos_embedding': True},
    'resonance': {'max_resonance_steps': 5, 'n_think': 2, 'n_verify': 2},
    'emotion': {'mode': 'first_only', 'dim': 64},
    'metacognition': {'enabled': True, 'threshold': 0.85},
    'world': {'max_ticks': 200},
    'training': {'episodes': 500, 'lr': 5e-4, 'gamma': 0.97, 'grad_clip': 1.0, 'entropy_bonus': 0.05, 'update_interval': 16},
    'growth': {'enabled': False},
    'action_head': {'initial_width': 128},
}
json.dump(cfg, open('/tmp/rigor_A_${SEED}.json', 'w'))
"
    .venv/bin/python src/bitnet/train_survival.py --config /tmp/rigor_A_${SEED}.json 2>&1 | tail -5 | tee -a "$RESULTS_DIR/ablation_A.log"
    
    # Config B: Big (648), no growth
    echo "  [B] Big, no growth..."
    .venv/bin/python -c "
import json
cfg = {
    'experiment_id': 'RIGOR_ablation_B_s${SEED}',
    'seed': $SEED,
    'model': {'hidden_dim': 256, 'num_layers': 3, 'use_pos_embedding': True},
    'resonance': {'max_resonance_steps': 5, 'n_think': 2, 'n_verify': 2},
    'emotion': {'mode': 'first_only', 'dim': 64},
    'metacognition': {'enabled': True, 'threshold': 0.85},
    'world': {'max_ticks': 200},
    'training': {'episodes': 500, 'lr': 5e-4, 'gamma': 0.97, 'grad_clip': 1.0, 'entropy_bonus': 0.05, 'update_interval': 16},
    'growth': {'enabled': False},
    'action_head': {'initial_width': 648},
}
json.dump(cfg, open('/tmp/rigor_B_${SEED}.json', 'w'))
"
    .venv/bin/python src/bitnet/train_survival.py --config /tmp/rigor_B_${SEED}.json 2>&1 | tail -5 | tee -a "$RESULTS_DIR/ablation_B.log"
    
    # Config C: Small→Big, growth enabled
    echo "  [C] Growth..."
    .venv/bin/python -c "
import json
cfg = {
    'experiment_id': 'RIGOR_ablation_C_s${SEED}',
    'seed': $SEED,
    'model': {'hidden_dim': 256, 'num_layers': 3, 'use_pos_embedding': True},
    'resonance': {'max_resonance_steps': 5, 'n_think': 2, 'n_verify': 2},
    'emotion': {'mode': 'first_only', 'dim': 64},
    'metacognition': {'enabled': True, 'threshold': 0.85},
    'world': {'max_ticks': 200},
    'training': {'episodes': 500, 'lr': 5e-4, 'gamma': 0.97, 'grad_clip': 1.0, 'entropy_bonus': 0.05, 'update_interval': 16},
    'growth': {'convergence_threshold': 0.80, 'patience': 50, 'factor': 1.5, 'max_width': 648},
}
json.dump(cfg, open('/tmp/rigor_C_${SEED}.json', 'w'))
"
    .venv/bin/python src/bitnet/train_survival.py --config /tmp/rigor_C_${SEED}.json 2>&1 | tail -5 | tee -a "$RESULTS_DIR/ablation_C.log"
done

# ── 2. RANDOM BASELINE — 5 seeds ──
echo ""
echo "═══ PHASE 2: RANDOM BASELINE (5 runs) ═══"

for SEED in "${SEEDS[@]}"; do
    echo "  Random agent, seed $SEED..."
    .venv/bin/python -c "
import random
from src.bitnet.minimal_world import MinimalWorld, ACTIONS

random.seed($SEED)
world = MinimalWorld(seed=$SEED)
results = []
for ep in range(500):
    state = world.reset()
    tick = 0
    while state.alive and tick < 200:
        tick += 1
        action = random.choice(ACTIONS)
        world.act(action)
    results.append(tick)
best = max(results)
avg100 = sum(results[-100:]) / 100
print(f'RANDOM seed=$SEED | Best: {best} | Avg100: {avg100:.1f}')
" 2>&1 | tee -a "$RESULTS_DIR/random_baseline.log"
done

# ── 3. NEUROPLASTICITY — frozen vs unfrozen, 5 seeds (dual) ──
echo ""
echo "═══ PHASE 3: NEUROPLASTICITY (10 runs) ═══"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed $SEED ---"
    
    # Frozen
    echo "  [F] Frozen backbone..."
    .venv/bin/python -c "
import json
cfg = {
    'experiment_id': 'RIGOR_frozen_s${SEED}',
    'seed': $SEED,
    'model': {'hidden_dim': 256, 'num_layers': 3, 'use_pos_embedding': True},
    'resonance': {'max_resonance_steps': 5, 'n_think': 2, 'n_verify': 2},
    'emotion': {'mode': 'first_only', 'dim': 64},
    'metacognition': {'enabled': True, 'threshold': 0.85},
    'world': {'max_ticks': 200},
    'training': {'episodes': 500, 'lr': 5e-4, 'gamma': 0.97, 'grad_clip': 1.0, 'entropy_bonus': 0.05, 'update_interval': 16},
    'growth': {'convergence_threshold': 0.80, 'patience': 50, 'factor': 1.5, 'max_width': 512},
    'communication': {'enabled': False},
}
json.dump(cfg, open('/tmp/rigor_frozen_${SEED}.json', 'w'))
"
    .venv/bin/python src/bitnet/train_dual.py --config /tmp/rigor_frozen_${SEED}.json 2>&1 | tail -5 | tee -a "$RESULTS_DIR/neuro_frozen.log"
    
    # Unfrozen
    echo "  [U] Unfrozen backbone..."
    .venv/bin/python -c "
import json
cfg = {
    'experiment_id': 'RIGOR_unfrozen_s${SEED}',
    'seed': $SEED,
    'unfreeze_backbone': True,
    'model': {'hidden_dim': 256, 'num_layers': 3, 'use_pos_embedding': True},
    'resonance': {'max_resonance_steps': 5, 'n_think': 2, 'n_verify': 2},
    'emotion': {'mode': 'first_only', 'dim': 64},
    'metacognition': {'enabled': True, 'threshold': 0.85},
    'world': {'max_ticks': 200},
    'training': {'episodes': 500, 'lr': 3e-4, 'gamma': 0.97, 'grad_clip': 1.0, 'entropy_bonus': 0.05, 'update_interval': 16},
    'growth': {'convergence_threshold': 0.80, 'patience': 50, 'factor': 1.5, 'max_width': 512},
    'communication': {'enabled': False},
}
json.dump(cfg, open('/tmp/rigor_unfrozen_${SEED}.json', 'w'))
"
    .venv/bin/python src/bitnet/train_dual.py --config /tmp/rigor_unfrozen_${SEED}.json 2>&1 | tail -5 | tee -a "$RESULTS_DIR/neuro_unfrozen.log"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "🏁 RIGOR SUITE COMPLETE — $(date)"
echo "   Results in: $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════"
