"""
EXP_034 Grid Runner — Glifos Ternarios.

Ejecuta las variantes del experimento de glifos ternarios.
Comparamos glifos vs EXP_033 para validar que la composicionalidad
no pierde el gap emocional.

Variantes:
  A_baseline      — sin resonancia, sin emoción (glifos)
  B_resonance     — resonancia pura (glifos, sin emoción)
  D_first_only    — resonancia + emoción first_only (glifos) ← ganador EXP_033
  D_additive      — resonancia + emoción additive (glifos)
"""

import json
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = os.path.join(BASE_DIR, "configs", "experiments")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "src", "bitnet", "train_glyph_resonance.py")

# Template base
TEMPLATE = {
	"seed": 42,
	"pop_size": 4,
	"model": {"hidden_dim": 256, "num_layers": 3, "use_pos_embedding": True},
	"resonance": {"max_resonance_steps": 5, "n_steps": 2, "pos_mode": "clock",
	              "loss_mode": "weighted", "intermediate_loss_weight": 0.2},
	"emotion": {"mode": "first_only", "dim": 64},
	"training": {"epochs": 200, "steps_per_epoch": 200, "batch_size": 32,
	             "lr": 3e-4, "weight_decay": 0.05, "grad_clip": 1.0},
	"curriculum": {"tau_start": 1.0, "tau_min": 0.3, "nursery_end": 30,
	               "transition_end": 80, "tf_min": 0.1},
	"early_stopping": {"patience": 30},
	"svd_interval": 3, "svd_phases": ["recreo", "autonomia"],
	"fear_amplifier": 3.0, "max_chain_depth": 2,
}

VARIANTS = [
	{"name": "A_baseline",   "condition": "A"},
	{"name": "B_resonance",  "condition": "B"},
	{"name": "D_first_only", "condition": "D", "emotion_mode": "first_only"},
	{"name": "D_additive",   "condition": "D", "emotion_mode": "additive"},
]


def generate_configs(tasting: bool = False):
	"""Genera los JSON de configuración."""
	prefix = "TASTING_034" if tasting else "EXP_034"
	configs = []
	for v in VARIANTS:
		cfg = json.loads(json.dumps(TEMPLATE))  # deep copy
		cfg["experiment_id"] = f"{prefix}_{v['name']}"
		cfg["condition"] = v["condition"]
		if "emotion_mode" in v:
			cfg["emotion"]["mode"] = v["emotion_mode"]
		if tasting:
			cfg["training"]["epochs"] = 40
			cfg["training"]["steps_per_epoch"] = 100
			cfg["curriculum"]["nursery_end"] = 10
			cfg["curriculum"]["transition_end"] = 25
		path = os.path.join(CONFIGS_DIR, f"{prefix}_{v['name']}.json")
		with open(path, "w", encoding="utf-8") as f:
			json.dump(cfg, f, ensure_ascii=False, indent=4)
		configs.append(path)
		print(f"  📝 {prefix}_{v['name']}")
	return configs


def run_grid(config_paths: list[str]):
	"""Ejecuta los experimentos en secuencia."""
	results = []
	for i, path in enumerate(config_paths):
		name = os.path.basename(path).replace(".json", "")
		print(f"\n{'═'*60}")
		print(f"  [{i+1}/{len(config_paths)}] {name}")
		print(f"{'═'*60}")

		t0 = time.time()
		try:
			result = subprocess.run(
				[sys.executable, TRAIN_SCRIPT, "--config", path],
				check=True, cwd=BASE_DIR,
			)
			elapsed = time.time() - t0
			results.append((name, "✅", elapsed))
			print(f"  ✅ {name} — {elapsed:.1f}s")
		except subprocess.CalledProcessError as e:
			elapsed = time.time() - t0
			results.append((name, "❌", elapsed))
			print(f"  ❌ {name} — falló después de {elapsed:.1f}s")

	print(f"\n{'═'*60}")
	print(f"  RESUMEN")
	print(f"{'═'*60}")
	for name, status, elapsed in results:
		print(f"  {status} {name} — {elapsed:.1f}s")


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--tasting", action="store_true", help="Run quick tasting (40 epochs)")
	args = parser.parse_args()

	print(f"🔤 EXP_034 Grid — Glifos Ternarios {'(TASTING)' if args.tasting else ''}")
	configs = generate_configs(tasting=args.tasting)
	run_grid(configs)
