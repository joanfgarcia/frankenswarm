"""
EXP_033 — Grid Runner: Resonancia Emocional (9 variantes).

Ejecuta las 9 condiciones experimentales secuencialmente:
    A: baseline | B: resonancia | C: resonancia + fear
    D: resonancia + emoción (additive/gated/first_only)
    E: resonancia + emoción + fear (additive/gated/first_only)

Uso:
    python -m lab.experiments.run_grid_033
    python -m lab.experiments.run_grid_033 --tasting    # Modo cata (40ep, 50 steps)
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time


def main():
	parser = argparse.ArgumentParser(description="EXP_033 Grid Runner")
	parser.add_argument("--tasting", action="store_true", help="Modo cata: 40 epochs, 50 steps")
	args = parser.parse_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_dir = os.path.join(base_dir, "configs", "experiments")

	# Encontrar todas las configs EXP_033
	configs = sorted(glob.glob(os.path.join(config_dir, "EXP_033_*.json")))
	print("🧠 EXP_033 — Resonancia Emocional")
	print(f"📦 {len(configs)} variantes encontradas")

	if args.tasting:
		print("🍷 MODO CATA: 40 epochs × 50 steps")

	print("=" * 60)

	results = {}
	start_total = time.time()

	for i, config_path in enumerate(configs):
		config_name = os.path.basename(config_path).replace(".json", "")

		# En modo tasting, modificar epochs y steps
		if args.tasting:
			with open(config_path) as f:
				cfg = json.load(f)
			cfg["training"]["epochs"] = 40
			cfg["training"]["steps_per_epoch"] = 50
			cfg["curriculum"]["nursery_end"] = 8
			cfg["curriculum"]["transition_end"] = 20
			cfg["experiment_id"] = f"TASTING_033_{config_name.replace('EXP_033_', '')}"
			tasting_cfg_path = config_path.replace("EXP_033_", "TASTING_033_")
			with open(tasting_cfg_path, "w") as f:
				json.dump(cfg, f, indent=4)
			config_path = tasting_cfg_path

		print(f"\n{'=' * 60}")
		print(f"  [{i + 1}/{len(configs)}] {config_name}")
		print(f"  Config: {config_path}")
		print("  OOM Shield: 10G")
		print(f"{'=' * 60}\n")

		start_variant = time.time()

		try:
			result = subprocess.run(
				[
					"systemd-run", "--user", "--scope", "-p", "MemoryMax=10G",
					sys.executable, "-m", "lab.experiments.train_emotional_resonance",
					"--config", config_path,
				],
				cwd=base_dir,
				timeout=7200,
			)

			elapsed = time.time() - start_variant
			status = "✅" if result.returncode == 0 else "❌"
			results[config_name] = {"status": status, "time": elapsed, "returncode": result.returncode}
			print(f"\n  {status} {config_name} — {elapsed:.1f}s")

		except subprocess.TimeoutExpired:
			results[config_name] = {"status": "⏰ TIMEOUT", "time": 7200, "returncode": -1}
			print(f"\n  ⏰ {config_name} — TIMEOUT")

		except Exception as e:
			results[config_name] = {"status": f"💥 {e}", "time": 0, "returncode": -1}
			print(f"\n  💥 {config_name} — {e}")

	# Resumen final
	total_time = time.time() - start_total
	print(f"\n{'=' * 60}")
	print("  📊 RESUMEN EXP_033")
	print(f"  ⏱️ Tiempo total: {total_time / 60:.1f} min ({total_time / 3600:.1f}h)")
	print(f"{'=' * 60}\n")

	for name, r in results.items():
		print(f"  {r['status']} {name} — {r['time']:.1f}s")


if __name__ == "__main__":
	main()
