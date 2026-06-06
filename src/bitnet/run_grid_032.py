"""
EXP_032: Grid Runner — Ejecuta las 27 variantes de la malla factorial 3³.

Genera las configs de cada variante a partir del template,
las ejecuta secuencialmente, y produce un resumen al final.

Uso:
    python -m src.bitnet.run_grid_032
    python -m src.bitnet.run_grid_032 --dry-run          # Solo genera configs, no entrena
    python -m src.bitnet.run_grid_032 --variants clock_3_final none_ramp_weighted  # Solo esas
    python -m src.bitnet.run_grid_032 --resume            # Salta variantes ya completadas
    python -m src.bitnet.run_grid_032 --tasting           # Cata rápida (~15 min, 27 variantes reducidas)
"""

import argparse
import contextlib
import itertools
import json
import os
import subprocess
import sys
import time

# ═══════════════════════════════════════════
# MALLA FACTORIAL 3³
# ═══════════════════════════════════════════

GRID_AXES = {
	"pos_mode": ["none", "entry", "clock"],
	"n_steps": ["2", "3", "ramp"],
	"loss_mode": ["final", "every", "weighted"],
}

# Template base (se sobrescribe con los ejes del grid)
BASE_CONFIG = {
	"seed": 42,
	"pop_size": 4,
	"max_chain_depth": 3,
	"fear_amplifier": 3.0,
	"svd_interval": 3,
	"svd_phases": ["recreo", "autonomia"],
	"use_logit_mask": True,
	"micro_vocab_words": [
		"gato", "perro", "casa", "árbol", "agua", "fuego",
		"tierra", "aire", "sol", "luna", "peligro", "seguridad", "implica",
	],

	"model": {
		"hidden_dim": 256,
		"num_layers": 3,
		"num_heads": 4,
		"mlp_ratio": 4,
		"use_pos_embedding": True,
	},

	"training": {
		"epochs": 200,
		"steps_per_epoch": 200,
		"batch_size": 32,
		"lr": 3e-4,
		"grad_clip": 1.0,
		"weight_decay": 0.05,
	},

	"curriculum": {
		"nursery_end": 30,
		"transition_end": 80,
		"tf_min": 0.1,
		"tau_start": 1.0,
		"tau_min": 0.3,
	},

	"depth_ramp": {
		"nursery_steps": 1,
		"transition_steps_start": 2,
		"transition_steps_end": 3,
		"autonomy_steps_start": 3,
		"autonomy_steps_end": 5,
	},
}

# Overrides para modo cata: suficiente para ver tendencias
TASTING_OVERRIDES = {
	"training": {
		"epochs": 40,
		"steps_per_epoch": 50,
		"batch_size": 32,
		"lr": 3e-4,
		"grad_clip": 1.0,
		"weight_decay": 0.05,
	},
	"curriculum": {
		"nursery_end": 10,
		"transition_end": 25,
		"tf_min": 0.1,
		"tau_start": 1.0,
		"tau_min": 0.3,
	},
	"resonance_extra": {
		"watcher_interval_epochs": 5,
	},
	"pop_size": 2,
	"svd_interval": 5,
}


def generate_variant_config(pos_mode: str, n_steps: str, loss_mode: str, tasting: bool = False) -> dict:
	"""Genera la config completa para una variante del grid."""
	config = json.loads(json.dumps(BASE_CONFIG))  # Deep copy

	prefix = "TASTING_032" if tasting else "EXP_032"
	variant_id = f"{prefix}_{pos_mode}_{n_steps}_{loss_mode}"
	config["experiment_id"] = variant_id
	config["title"] = f"Resonancia Continua — pos_{pos_mode}, depth_{n_steps}, loss_{loss_mode}"

	config["resonance"] = {
		"pos_mode": pos_mode,
		"n_steps": n_steps,
		"loss_mode": loss_mode,
		"max_resonance_steps": 5,
		"collect_watcher": True,
		"watcher_interval_epochs": 10,
		"intermediate_loss_weight": 0.2,
	}

	if tasting:
		config["training"] = TASTING_OVERRIDES["training"]
		config["curriculum"] = TASTING_OVERRIDES["curriculum"]
		config["pop_size"] = TASTING_OVERRIDES["pop_size"]
		config["svd_interval"] = TASTING_OVERRIDES["svd_interval"]
		config["resonance"]["watcher_interval_epochs"] = TASTING_OVERRIDES["resonance_extra"]["watcher_interval_epochs"]

	return config


def generate_all_variants(tasting: bool = False) -> list[tuple[str, dict]]:
	"""Genera las 27 combinaciones del grid."""
	variants = []
	prefix = "TASTING_032" if tasting else "EXP_032"
	for pos, depth, loss in itertools.product(*GRID_AXES.values()):
		variant_id = f"{prefix}_{pos}_{depth}_{loss}"
		config = generate_variant_config(pos, depth, loss, tasting=tasting)
		variants.append((variant_id, config))
	return variants


def is_variant_completed(base_dir: str, variant_id: str) -> bool:
	"""Comprueba si una variante ya se ejecutó mirando si existe telemetry.jsonl."""
	telemetry_path = os.path.join(base_dir, "storage", "experiments", variant_id, "telemetry.jsonl")
	if not os.path.exists(telemetry_path):
		return False
	# Verificar que tiene al menos 10 líneas (no es un run abortado)
	with open(telemetry_path) as f:
		lines = sum(1 for _ in f)
	return lines >= 10


def run_variant(base_dir: str, variant_id: str, config: dict, python_exe: str, mem_limit: str = "10G") -> dict:
	"""Ejecuta una variante con OOM Shield."""
	# Guardar config
	config_dir = os.path.join(base_dir, "configs", "experiments")
	config_path = os.path.join(config_dir, f"{variant_id}.json")
	os.makedirs(config_dir, exist_ok=True)
	with open(config_path, "w", encoding="utf-8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)

	# Comando con OOM Shield
	cmd = [
		"systemd-run", "--user", "--scope",
		"-p", f"MemoryMax={mem_limit}",
		python_exe, "-m", "src.bitnet.train_resonance",
		"--config", config_path,
	]

	print(f"\n{'='*60}")
	print(f"  🚀 Ejecutando: {variant_id}")
	print(f"  Config: {config_path}")
	print(f"  OOM Shield: {mem_limit}")
	print(f"{'='*60}\n")

	start_time = time.time()
	try:
		result = subprocess.run(
			cmd,
			cwd=base_dir,
			capture_output=True,
			text=True,
			timeout=3600,  # 1h máximo por variante
		)
		elapsed = time.time() - start_time

		# Extraer última línea de accuracy del output
		acc_joint = 0.0
		for line in result.stdout.split("\n"):
			if "Listener (joint):" in line:
				with contextlib.suppress(ValueError, IndexError):
					acc_joint = float(line.split("Listener (joint):")[1].strip().rstrip("%"))

		return {
			"variant_id": variant_id,
			"status": "OK" if result.returncode == 0 else "FAIL",
			"returncode": result.returncode,
			"elapsed_seconds": elapsed,
			"acc_joint_last": acc_joint,
			"stderr_tail": result.stderr[-500:] if result.stderr else "",
		}
	except subprocess.TimeoutExpired:
		return {
			"variant_id": variant_id,
			"status": "TIMEOUT",
			"elapsed_seconds": 3600,
			"acc_joint_last": 0.0,
		}
	except Exception as e:
		return {
			"variant_id": variant_id,
			"status": "ERROR",
			"error": str(e),
			"elapsed_seconds": time.time() - start_time,
			"acc_joint_last": 0.0,
		}


def print_summary(results: list[dict]):
	"""Imprime tabla resumen de todas las variantes ejecutadas."""
	print(f"\n{'='*80}")
	print(f"  EXP_032 — GRID RUNNER: RESUMEN ({len(results)} variantes)")
	print(f"{'='*80}\n")

	print(f"{'Variante':<35}  {'Status':>8}  {'Tiempo':>8}  {'acc_joint':>10}")
	print("-" * 70)

	for r in sorted(results, key=lambda x: -x.get("acc_joint_last", 0)):
		elapsed = f"{r['elapsed_seconds']:.0f}s" if r.get("elapsed_seconds") else "?"
		acc = f"{r.get('acc_joint_last', 0):.2f}%" if r.get("acc_joint_last") else "?"
		status_icon = "✅" if r["status"] == "OK" else "❌" if r["status"] == "FAIL" else "⏰"
		print(f"{r['variant_id']:<35}  {status_icon} {r['status']:>5}  {elapsed:>8}  {acc:>10}")

	# Top 3
	ok_results = [r for r in results if r["status"] == "OK"]
	if ok_results:
		top3 = sorted(ok_results, key=lambda x: -x.get("acc_joint_last", 0))[:3]
		print("\n🏆 Top 3:")
		for i, r in enumerate(top3):
			print(f"   {i+1}. {r['variant_id']} — {r.get('acc_joint_last', 0):.2f}%")

	total_time = sum(r.get("elapsed_seconds", 0) for r in results)
	print(f"\n⏱️  Tiempo total: {total_time/60:.1f} min ({total_time/3600:.1f}h)")
	print(f"   OK: {sum(1 for r in results if r['status'] == 'OK')}")
	print(f"   FAIL: {sum(1 for r in results if r['status'] == 'FAIL')}")
	print(f"   TIMEOUT: {sum(1 for r in results if r['status'] == 'TIMEOUT')}")


def main():
	parser = argparse.ArgumentParser(description="EXP_032: Grid Runner (27 variantes)")
	parser.add_argument("--dry-run", action="store_true", help="Solo genera configs, no entrena")
	parser.add_argument("--resume", action="store_true", help="Salta variantes ya completadas")
	parser.add_argument("--tasting", action="store_true", help="Cata rápida: 40ep×50steps (~15 min total)")
	parser.add_argument("--variants", nargs="*", help="Solo ejecutar estas variantes (ej: clock_3_final none_ramp_weighted)")
	parser.add_argument("--mem-limit", default="10G", help="Límite de memoria OOM Shield (default: 10G)")
	parser.add_argument("--python", default=None, help="Path al intérprete Python (default: auto-detect)")
	args = parser.parse_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

	# Auto-detect Python
	python_exe = args.python
	if python_exe is None:
		venv_python = os.path.join(base_dir, ".venv", "bin", "python")
		python_exe = venv_python if os.path.exists(venv_python) else sys.executable

	# Generar variantes
	all_variants = generate_all_variants(tasting=args.tasting)
	mode = "🍷 CATA RÁPIDA (40ep×50steps)" if args.tasting else "🔬 FULL (200ep×200steps)"
	print(f"📊 Grid factorial 3³ = {len(all_variants)} variantes [{mode}]")

	# Filtrar si se piden variantes específicas
	if args.variants:
		filter_ids = {f"EXP_032_{v}" for v in args.variants}
		all_variants = [(vid, cfg) for vid, cfg in all_variants if vid in filter_ids]
		print(f"   Filtrado a {len(all_variants)} variantes")

	if args.dry_run:
		print("\n📝 DRY RUN — Generando configs sin ejecutar:\n")
		for variant_id, config in all_variants:
			config_path = os.path.join(base_dir, "configs", "experiments", f"{variant_id}.json")
			os.makedirs(os.path.dirname(config_path), exist_ok=True)
			with open(config_path, "w", encoding="utf-8") as f:
				json.dump(config, f, ensure_ascii=False, indent=4)
			r = config["resonance"]
			print(f"  ✅ {variant_id}: pos={r['pos_mode']} depth={r['n_steps']} loss={r['loss_mode']}")
		print(f"\n💾 {len(all_variants)} configs guardadas en configs/experiments/")
		return

	# Ejecutar
	results = []
	skipped = 0

	for i, (variant_id, config) in enumerate(all_variants):
		print(f"\n[{i+1}/{len(all_variants)}] ", end="")

		if args.resume and is_variant_completed(base_dir, variant_id):
			print(f"⏭️  {variant_id} — Ya completado, saltando")
			skipped += 1
			continue

		result = run_variant(base_dir, variant_id, config, python_exe, args.mem_limit)
		results.append(result)

		# Guardar progreso incremental
		progress_path = os.path.join(base_dir, "storage", "experiments", "EXP_032_grid_progress.json")
		with open(progress_path, "w") as f:
			json.dump(results, f, indent=2, default=str)

	if skipped:
		print(f"\n⏭️  {skipped} variantes saltadas (ya completadas)")

	if results:
		print_summary(results)

		# Guardar resumen final
		summary_path = os.path.join(base_dir, "storage", "experiments", "EXP_032_grid_summary.json")
		with open(summary_path, "w") as f:
			json.dump(results, f, indent=2, default=str)
		print(f"\n💾 Resumen guardado en {summary_path}")


if __name__ == "__main__":
	main()
