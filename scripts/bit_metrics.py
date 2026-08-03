#!/usr/bin/env python3
"""Dashboard de métricas: Bit v2 (Formación Nativa K-65P) — instrumentos DL-004.

Muestra el estado real de la escuela (épocas, etapa, dim, exámenes, neurogénesis)
sin comparaciones inválidas: la cross-entropy de Bit v1 (vocab 6.400, texto natural)
y la de Bit v2 (vocab 164, lenguaje formal) NO son comparables entre sí y este
dashboard no las enfrenta (DL-004; la tabla del 2-ago fue retirada).

Uso:
	python scripts/bit_metrics.py [--state_dir DIR]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

DEFAULT_STATE_DIR = base_dir / "storage" / "checkpoints" / "sovereign_school_k65p"
LOG_DIR = base_dir / "storage" / "logs"

STAGE_NAMES = [
	"0-1 años (Preschool I)",
	"1-2 años (Preschool II)",
	"2-3 años (Preschool III)",
	"3-4 años (Preschool IV)",
	"4-5 años (Primary I)",
	"5-6 años (Primary II)",
	"6-7 años (Secondary I)",
	"7-8 años (Secondary II)",
]


def count_checkpoint_params(ckpt_path: Path) -> int | None:
	"""Cuenta parámetros REALES del checkpoint (no estimaciones)."""
	try:
		import torch
		ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
		sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
		return sum(v.numel() for v in sd.values())
	except Exception:
		return None


def get_latest_log_metrics() -> tuple[float | None, float | None, float | None]:
	logs = sorted(LOG_DIR.glob("school_k65p_*.log"), key=os.path.getmtime, reverse=True)
	if not logs:
		return None, None, None

	last_train, last_val, last_acc = None, None, None
	pattern = re.compile(r"Train Loss: ([\d\.]+|inf|nan) \| Val Loss: ([\d\.]+|inf|nan)(?: \| Val Acc \(sin pads\): ([\d\.]+))?")
	with open(logs[0], encoding="utf-8", errors="ignore") as f:
		for line in f:
			m = pattern.search(line)
			if m:
				last_train = float(m.group(1))
				last_val = float(m.group(2))
				if m.group(3):
					last_acc = float(m.group(3))
	return last_train, last_val, last_acc


def main():
	parser = argparse.ArgumentParser(description="Dashboard Bit v2 K-65P")
	parser.add_argument("--state_dir", type=str, default=str(DEFAULT_STATE_DIR))
	args = parser.parse_args()

	state_dir = Path(args.state_dir)
	state_file = state_dir / "school_state_k65p.json"

	print("═" * 80)
	print("        📊 TELEMETRÍA BIT V2 (K-65P NATIVO) — instrumentos DL-004")
	print("═" * 80 + "\n")

	if not state_file.exists():
		print("⚠️ Estado de Bit v2 no encontrado todavía en disco.")
		print(f"Buscado en: {state_file}")
		return

	state = json.loads(state_file.read_text(encoding="utf-8"))
	current_epoch = state.get("current_epoch", 0)
	current_stage_idx = state.get("current_stage_idx", 0)
	hidden_dim = state.get("hidden_dim", 128)
	milestones = state.get("milestones_achieved", [])
	neuro_history = state.get("neurogenesis_history", [])
	exam_history = state.get("exam_history", [])
	exam_failures = state.get("exam_failures", {})
	retention = state.get("retention_epochs", 0)
	best_val = state.get("best_val_loss")

	train_loss, val_loss, val_acc = get_latest_log_metrics()
	n_params = count_checkpoint_params(state_dir / "model_current_k65p.pt")

	total_epochs = 1408 + retention
	pct = min(1.0, current_epoch / max(1, total_epochs))
	bar = "█" * int(round(30 * pct)) + "░" * (30 - int(round(30 * pct)))

	print(" 🎓 Estado de la Escuela Bit v2:")
	print(f"    • Época               : {current_epoch} / {total_epochs}" + (f" (incluye {retention} ép. de repetición)" if retention else ""))
	print(f"    • Progreso            : [{bar}] {pct*100:.1f}%")
	print(f"    • Etapa               : {current_stage_idx + 1}/8 — {STAGE_NAMES[min(current_stage_idx, 7)]}")
	print(f"    • Hitos APROBADOS     : {', '.join(milestones) if milestones else 'ninguno todavía (los hitos se examinan, no se regalan)'}")
	if exam_failures:
		print(f"    • Exámenes suspendidos: {json.dumps(exam_failures, ensure_ascii=False)}")
	dim_str = f"{hidden_dim}d"
	if n_params:
		dim_str += f" | {n_params/1e6:.2f}M params reales (contados del checkpoint)"
	print(f"    • Dimensión oculta    : {dim_str}")

	print("\n 📉 Métricas (pads EXCLUIDOS de loss y precisión):")
	if train_loss is not None:
		print(f"    • Train Loss (último) : {train_loss:.4f}")
	if val_loss is not None:
		print(f"    • Val Loss (último)   : {val_loss:.4f}")
	if val_acc is not None:
		print(f"    • Val Acc (último)    : {val_acc*100:.2f}%")
	if best_val is not None and best_val != float("inf"):
		print(f"    • Mejor Val Loss      : {best_val:.4f}")
	print("    • ⚠️ Estas cifras NO son comparables con Bit v1 (vocabularios y tareas distintas).")

	print("\n 📝 Actas de examen de hito:")
	if not exam_history:
		print("    • Sin exámenes todavía.")
	else:
		for exam in exam_history:
			verdict = "✅ APROBADO" if exam.get("passed") else "❌ SUSPENDIDO"
			bigram = f" | loss={exam.get('val_loss'):.3f} vs bigrama={exam.get('bigram_loss')}" if exam.get("bigram_loss") else ""
			print(f"    • [{exam.get('milestone')}] ép. {exam.get('epoch')}: gen_valid={exam.get('gen_valid_rate')}{bigram} | acc={exam.get('token_acc_no_pad')} → {verdict}")

	print("\n 🧬 Neurogénesis Net2WiderNet:")
	if not neuro_history:
		print("    • Ningún evento (anchura base 128d).")
	else:
		for event in neuro_history:
			print(f"    • Época {event.get('epoch'):4d}: {event.get('old_dim')}d ➔ {event.get('new_dim')}d | val_loss al disparo: {event.get('val_loss')}")

	print("\n💡 Para la evaluación adversarial (OOD real, baselines): scripts/run_adversarial_suite.py\n")


if __name__ == "__main__":
	main()
