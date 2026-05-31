"""
EXP_032: Análisis comparativo de la malla factorial 3³.

Lee los 27 telemetry.jsonl, extrae métricas clave, y genera:
1. Tabla resumen markdown (27 variantes × métricas)
2. Heatmaps ASCII de acc_joint por combinación de ejes
3. Ranking de top/bottom variantes
4. Comparación vs baseline EXP_029 (70% acc_joint)
5. CSV exportable para visualización externa

Uso:
    python scripts/analyze_grid_032.py
    python scripts/analyze_grid_032.py --output-dir results/EXP_032/
"""

import argparse
import csv
import json
import os
from pathlib import Path


BASELINE_EXP029_ACC_JOINT = 70.0  # EXP_029 autonomía: ~70% acc_joint

GRID_AXES = {
	"pos_mode": ["none", "entry", "clock"],
	"n_steps": ["2", "3", "ramp"],
	"loss_mode": ["final", "every", "weighted"],
}


def load_telemetry(exp_dir: str) -> list[dict]:
	"""Carga telemetry.jsonl y devuelve las entradas de tipo 'epoch'."""
	telemetry_path = os.path.join(exp_dir, "telemetry.jsonl")
	if not os.path.exists(telemetry_path):
		return []
	epochs = []
	with open(telemetry_path, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				entry = json.loads(line)
				if entry.get("type") == "epoch":
					epochs.append(entry)
			except json.JSONDecodeError:
				continue
	return epochs


def load_stability_metrics(exp_dir: str) -> list[dict]:
	"""Carga stability_metrics.jsonl."""
	path = os.path.join(exp_dir, "stability_metrics.jsonl")
	if not os.path.exists(path):
		return []
	entries = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				entries.append(json.loads(line))
			except json.JSONDecodeError:
				continue
	return entries


def load_watcher_log(exp_dir: str) -> list[dict]:
	"""Carga watcher_log.jsonl."""
	path = os.path.join(exp_dir, "watcher_log.jsonl")
	if not os.path.exists(path):
		return []
	entries = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				entries.append(json.loads(line))
			except json.JSONDecodeError:
				continue
	return entries


def extract_metrics(variant_id: str, epochs: list[dict], stability: list[dict], watcher: list[dict]) -> dict:
	"""Extrae métricas clave de una variante."""
	if not epochs:
		return {"variant_id": variant_id, "status": "NO DATA"}

	# Parsear ejes del variant_id: EXP_032_{pos}_{depth}_{loss}
	parts = variant_id.replace("EXP_032_", "").split("_")
	if len(parts) >= 3:
		pos_mode, depth, loss_mode = parts[0], parts[1], parts[2]
	else:
		pos_mode, depth, loss_mode = "?", "?", "?"

	# Métricas finales (últimas 10 epochs = autonomía estable)
	last_10 = epochs[-10:] if len(epochs) >= 10 else epochs
	last_20 = epochs[-20:] if len(epochs) >= 20 else epochs

	acc_joint_final = sum(e.get("acc_joint", 0) for e in last_10) / len(last_10)
	acc_speaker_final = sum(e.get("acc_concept", 0) for e in last_10) / len(last_10)
	loss_final = sum(e.get("loss_avg", 0) for e in last_10) / len(last_10)

	# Mejor epoch
	best_epoch = max(epochs, key=lambda e: e.get("acc_joint", 0))
	acc_joint_peak = best_epoch.get("acc_joint", 0)
	peak_epoch = best_epoch.get("epoch", 0)

	# Velocidad de convergencia: primer epoch con acc_joint > 50%
	convergence_epoch = None
	for e in epochs:
		if e.get("acc_joint", 0) > 50:
			convergence_epoch = e.get("epoch", 0)
			break

	# Estabilidad en autonomía (desviación estándar de acc_joint en últimas 20 epochs)
	if len(last_20) > 1:
		mean_acc = sum(e.get("acc_joint", 0) for e in last_20) / len(last_20)
		variance = sum((e.get("acc_joint", 0) - mean_acc) ** 2 for e in last_20) / len(last_20)
		stability_std = variance ** 0.5
	else:
		stability_std = 0.0

	# Watcher: norm_ratio y cosine del último watcher log
	norm_ratio_final = None
	cosine_final = None
	if watcher:
		last_w = watcher[-1]
		norm_ratio_final = last_w.get("norm_ratio")
		cos_conv = last_w.get("cosine_convergence", [])
		cosine_final = cos_conv[-1] if cos_conv else None

	# vs baseline
	delta_vs_029 = acc_joint_final - BASELINE_EXP029_ACC_JOINT

	return {
		"variant_id": variant_id,
		"pos_mode": pos_mode,
		"depth": depth,
		"loss_mode": loss_mode,
		"status": "OK",
		"total_epochs": len(epochs),
		# Accuracy
		"acc_joint_final": round(acc_joint_final, 2),
		"acc_joint_peak": round(acc_joint_peak, 2),
		"acc_speaker_final": round(acc_speaker_final, 2),
		"peak_epoch": peak_epoch,
		"convergence_epoch": convergence_epoch,
		# Loss
		"loss_final": round(loss_final, 4),
		# Estabilidad
		"stability_std": round(stability_std, 2),
		"norm_ratio": round(norm_ratio_final, 4) if norm_ratio_final else None,
		"cosine_convergence": round(cosine_final, 4) if cosine_final else None,
		# vs baseline
		"delta_vs_029": round(delta_vs_029, 2),
	}


def generate_markdown_report(metrics: list[dict], output_path: str):
	"""Genera informe markdown completo."""
	ok = [m for m in metrics if m["status"] == "OK"]
	no_data = [m for m in metrics if m["status"] != "OK"]

	ok_sorted = sorted(ok, key=lambda m: -m["acc_joint_final"])

	with open(output_path, "w", encoding="utf-8") as f:
		f.write("# EXP_032 — Análisis de la Malla Factorial 3³\n\n")
		f.write(f"**Variantes ejecutadas**: {len(ok)} / 27\n")
		f.write(f"**Variantes sin datos**: {len(no_data)}\n")
		f.write(f"**Baseline (EXP_029)**: {BASELINE_EXP029_ACC_JOINT}% acc_joint\n\n")

		# ═══════════════════════════════════════════
		# Tabla principal (ranking)
		# ═══════════════════════════════════════════
		f.write("## 🏆 Ranking por acc_joint (últimas 10 epochs)\n\n")
		f.write("| # | Variante | pos | depth | loss | acc_joint | peak | Δ vs 029 | loss | stability |\n")
		f.write("|---|---|---|---|---|---|---|---|---|---|\n")

		for i, m in enumerate(ok_sorted):
			delta = f"+{m['delta_vs_029']:.1f}" if m["delta_vs_029"] >= 0 else f"{m['delta_vs_029']:.1f}"
			medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}"
			f.write(
				f"| {medal} | `{m['variant_id'].replace('EXP_032_', '')}` | "
				f"{m['pos_mode']} | {m['depth']} | {m['loss_mode']} | "
				f"**{m['acc_joint_final']:.1f}%** | {m['acc_joint_peak']:.1f}% | "
				f"{delta}% | {m['loss_final']:.3f} | σ={m['stability_std']:.1f} |\n"
			)

		# ═══════════════════════════════════════════
		# Heatmaps por eje
		# ═══════════════════════════════════════════
		f.write("\n---\n\n## 📊 Heatmaps (acc_joint promedio por eje)\n\n")

		# Heatmap: pos_mode × depth (promediando sobre loss_mode)
		f.write("### pos_mode × depth (promedio sobre loss_mode)\n\n")
		f.write("| | depth_2 | depth_3 | depth_ramp |\n")
		f.write("|---|---|---|---|\n")

		for pos in GRID_AXES["pos_mode"]:
			row = f"| **{pos}** |"
			for depth in GRID_AXES["n_steps"]:
				matching = [m for m in ok if m["pos_mode"] == pos and m["depth"] == depth]
				if matching:
					avg = sum(m["acc_joint_final"] for m in matching) / len(matching)
					# Color coding: >=80 bold, >=70 normal, <70 dim
					if avg >= 80:
						row += f" **{avg:.1f}%** |"
					elif avg >= BASELINE_EXP029_ACC_JOINT:
						row += f" {avg:.1f}% |"
					else:
						row += f" _{avg:.1f}%_ |"
				else:
					row += " — |"
			f.write(row + "\n")

		# Heatmap: pos_mode × loss_mode (promediando sobre depth)
		f.write("\n### pos_mode × loss_mode (promedio sobre depth)\n\n")
		f.write("| | loss_final | loss_every | loss_weighted |\n")
		f.write("|---|---|---|---|\n")

		for pos in GRID_AXES["pos_mode"]:
			row = f"| **{pos}** |"
			for loss in GRID_AXES["loss_mode"]:
				matching = [m for m in ok if m["pos_mode"] == pos and m["loss_mode"] == loss]
				if matching:
					avg = sum(m["acc_joint_final"] for m in matching) / len(matching)
					if avg >= 80:
						row += f" **{avg:.1f}%** |"
					elif avg >= BASELINE_EXP029_ACC_JOINT:
						row += f" {avg:.1f}% |"
					else:
						row += f" _{avg:.1f}%_ |"
				else:
					row += " — |"
			f.write(row + "\n")

		# Heatmap: depth × loss_mode (promediando sobre pos)
		f.write("\n### depth × loss_mode (promedio sobre pos_mode)\n\n")
		f.write("| | loss_final | loss_every | loss_weighted |\n")
		f.write("|---|---|---|---|\n")

		for depth in GRID_AXES["n_steps"]:
			row = f"| **depth_{depth}** |"
			for loss in GRID_AXES["loss_mode"]:
				matching = [m for m in ok if m["depth"] == depth and m["loss_mode"] == loss]
				if matching:
					avg = sum(m["acc_joint_final"] for m in matching) / len(matching)
					if avg >= 80:
						row += f" **{avg:.1f}%** |"
					elif avg >= BASELINE_EXP029_ACC_JOINT:
						row += f" {avg:.1f}% |"
					else:
						row += f" _{avg:.1f}%_ |"
				else:
					row += " — |"
			f.write(row + "\n")

		# ═══════════════════════════════════════════
		# Análisis por eje (efecto marginal)
		# ═══════════════════════════════════════════
		f.write("\n---\n\n## 🔬 Efecto marginal de cada eje\n\n")

		for axis_name, axis_values in GRID_AXES.items():
			f.write(f"### Eje: {axis_name}\n\n")
			f.write(f"| {axis_name} | acc_joint μ | acc_joint σ | n |\n")
			f.write("|---|---|---|---|\n")

			for val in axis_values:
				matching = [m for m in ok if m.get(axis_name, m.get("depth")) == val]
				# Map axis to field
				if axis_name == "n_steps":
					matching = [m for m in ok if m["depth"] == val]
				if matching:
					avg = sum(m["acc_joint_final"] for m in matching) / len(matching)
					if len(matching) > 1:
						var = sum((m["acc_joint_final"] - avg) ** 2 for m in matching) / len(matching)
						std = var ** 0.5
					else:
						std = 0.0
					f.write(f"| **{val}** | {avg:.1f}% | {std:.1f} | {len(matching)} |\n")
				else:
					f.write(f"| **{val}** | — | — | 0 |\n")
			f.write("\n")

		# ═══════════════════════════════════════════
		# Watcher analysis (top 3)
		# ═══════════════════════════════════════════
		f.write("---\n\n## 🔭 Watcher — Estabilidad de los top 3\n\n")
		for i, m in enumerate(ok_sorted[:3]):
			f.write(f"### #{i+1}: `{m['variant_id'].replace('EXP_032_', '')}`\n\n")
			if m.get("norm_ratio"):
				f.write(f"- **norm_ratio**: {m['norm_ratio']:.4f}\n")
			if m.get("cosine_convergence"):
				f.write(f"- **cosine_convergence**: {m['cosine_convergence']:.4f}\n")
			f.write(f"- **acc_joint_final**: {m['acc_joint_final']:.1f}%\n")
			f.write(f"- **peak**: {m['acc_joint_peak']:.1f}% (epoch {m['peak_epoch']})\n")
			f.write(f"- **convergence_epoch** (>50%): {m['convergence_epoch']}\n\n")

		# ═══════════════════════════════════════════
		# Veredicto
		# ═══════════════════════════════════════════
		f.write("---\n\n## 📋 Veredicto\n\n")

		beats_baseline = [m for m in ok if m["delta_vs_029"] > 0]
		f.write(f"- **Variantes que superan EXP_029 (>{BASELINE_EXP029_ACC_JOINT}%)**: {len(beats_baseline)}/27\n")

		if ok_sorted:
			best = ok_sorted[0]
			f.write(f"- **Mejor variante**: `{best['variant_id']}` — **{best['acc_joint_final']:.1f}%** ")
			f.write(f"(Δ {best['delta_vs_029']:+.1f}% vs EXP_029)\n")

			worst = ok_sorted[-1]
			f.write(f"- **Peor variante**: `{worst['variant_id']}` — {worst['acc_joint_final']:.1f}%\n")

		if no_data:
			f.write(f"\n> [!WARNING]\n> {len(no_data)} variantes sin datos: ")
			f.write(", ".join(m["variant_id"] for m in no_data))
			f.write("\n")


def generate_csv(metrics: list[dict], output_path: str):
	"""Exporta métricas a CSV para visualización externa."""
	ok = [m for m in metrics if m["status"] == "OK"]
	if not ok:
		return

	fieldnames = [
		"variant_id", "pos_mode", "depth", "loss_mode",
		"acc_joint_final", "acc_joint_peak", "acc_speaker_final",
		"peak_epoch", "convergence_epoch", "loss_final",
		"stability_std", "norm_ratio", "cosine_convergence", "delta_vs_029",
	]

	with open(output_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
		writer.writeheader()
		for m in sorted(ok, key=lambda x: -x["acc_joint_final"]):
			writer.writerow(m)


def main():
	parser = argparse.ArgumentParser(description="EXP_032: Grid Analysis")
	parser.add_argument("--output-dir", default=None, help="Output directory (default: storage/experiments/)")
	parser.add_argument("--tasting", action="store_true", help="Analizar la cata rápida (TASTING_032_ prefix)")
	args = parser.parse_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	experiments_dir = os.path.join(base_dir, "storage", "experiments")
	output_dir = args.output_dir or experiments_dir

	prefix = "TASTING_032" if args.tasting else "EXP_032"
	label = "Cata Rápida" if args.tasting else "Malla Factorial 3³"
	print(f"📊 {prefix} — Análisis de la {label}\n")

	# Descubrir variantes
	all_metrics = []
	for pos in GRID_AXES["pos_mode"]:
		for depth in GRID_AXES["n_steps"]:
			for loss in GRID_AXES["loss_mode"]:
				variant_id = f"{prefix}_{pos}_{depth}_{loss}"
				exp_dir = os.path.join(experiments_dir, variant_id)

				epochs = load_telemetry(exp_dir)
				stability = load_stability_metrics(exp_dir)
				watcher = load_watcher_log(exp_dir)

				metrics = extract_metrics(variant_id, epochs, stability, watcher)
				all_metrics.append(metrics)

				status_icon = "✅" if metrics["status"] == "OK" else "⬜"
				acc = f"{metrics.get('acc_joint_final', 0):.1f}%" if metrics["status"] == "OK" else "—"
				print(f"  {status_icon} {variant_id}: {acc}")

	ok_count = sum(1 for m in all_metrics if m["status"] == "OK")
	print(f"\n📦 {ok_count}/27 variantes con datos\n")

	if ok_count == 0:
		print("❌ No hay datos. Ejecuta primero: python -m src.bitnet.run_grid_032")
		return

	# Generar informe markdown
	md_path = os.path.join(output_dir, f"{prefix}_GRID_ANALYSIS.md")
	generate_markdown_report(all_metrics, md_path)
	print(f"📝 Informe markdown: {md_path}")

	# Generar CSV
	csv_path = os.path.join(output_dir, f"{prefix}_grid_results.csv")
	generate_csv(all_metrics, csv_path)
	print(f"📊 CSV exportado: {csv_path}")

	# Imprimir top 5 en consola
	ok_sorted = sorted([m for m in all_metrics if m["status"] == "OK"], key=lambda m: -m["acc_joint_final"])
	print(f"\n🏆 Top 5:")
	for i, m in enumerate(ok_sorted[:5]):
		delta = f"+{m['delta_vs_029']:.1f}" if m["delta_vs_029"] >= 0 else f"{m['delta_vs_029']:.1f}"
		print(f"   {i+1}. {m['variant_id'].replace('EXP_032_', ''):<25} {m['acc_joint_final']:.1f}% (Δ{delta}% vs EXP_029)")

	# Efecto marginal rápido
	print(f"\n📐 Efecto marginal (media acc_joint):")
	for axis_name, axis_values in GRID_AXES.items():
		print(f"   {axis_name}:")
		for val in axis_values:
			if axis_name == "n_steps":
				matching = [m for m in ok_sorted if m["depth"] == val]
			else:
				matching = [m for m in ok_sorted if m.get(axis_name) == val]
			if matching:
				avg = sum(m["acc_joint_final"] for m in matching) / len(matching)
				print(f"     {val:>8}: {avg:.1f}%")


if __name__ == "__main__":
	main()
