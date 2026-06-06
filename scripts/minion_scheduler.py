import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime

import yaml

# Añadir base dir al path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)



def load_queue(queue_path: str) -> dict:
	with open(queue_path, encoding="utf-8") as f:
		return yaml.safe_load(f)


def save_queue(queue_path: str, data: dict) -> None:
	with open(queue_path, "w", encoding="utf-8") as f:
		yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def run_next_experiment():
	queue_path = os.path.join(base_dir, "lab", "experiment_queue.yaml")
	inbox_dir = os.path.join(base_dir, "lab", ".inbox")
	os.makedirs(inbox_dir, exist_ok=True)

	if not os.path.exists(queue_path):
		print(f"❌ No se encuentra la cola de experimentos en: {queue_path}")
		sys.exit(1)

	queue_data = load_queue(queue_path)
	completed_ids = {item["id"] for item in queue_data.get("completed", []) if "id" in item}

	# Obtener lista de tareas pendientes
	pending_tasks = [task for task in queue_data.get("queue", []) if task.get("id") not in completed_ids]

	if not pending_tasks:
		print("🟢 No hay experimentos pendientes en la cola.")
		return

	task = pending_tasks[0]
	task_id = task["id"]
	task_name = task.get("name", f"Experimento {task_id}")

	print(f"🚀 Iniciando {task_name} (ID: {task_id})...")

	# Actualizar estado de experimento actual en el YAML
	queue_data["current_experiment"] = task_id
	queue_data["last_updated"] = datetime.now(UTC).astimezone().isoformat()
	queue_data["last_run_by"] = "minion_scheduler"
	save_queue(queue_path, queue_data)

	exp_dir = os.path.join(base_dir, "storage", "experiments", f"EXP_{task_id}")
	os.makedirs(exp_dir, exist_ok=True)
	log_path = os.path.join(exp_dir, "train.log")

	# Comando con OOM Shield de systemd
	config_path = f"configs/experiments/EXP_{task_id}.json"
	script_name = task.get("script", "src.bitnet.train_generic")
	script_file = "src/bitnet/train_generic.py"

	if script_name != "src.bitnet.train_generic":
		# Mapear script module a path de archivo
		script_file = script_name.replace(".", "/") + ".py"

	cmd = ["systemd-run", "--user", "--scope", "-p", "MemoryMax=10G", "env", "PYTHONPATH=.", ".venv/bin/python", script_file, "--config", config_path]

	print(f"📋 Ejecutando: {' '.join(cmd)}")
	print(f"📝 Redirigiendo salida a: {log_path}")

	start_time = time.monotonic()

	with open(log_path, "w", encoding="utf-8") as log_file:
		# Ejecutar el entrenamiento de forma síncrona
		process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=base_dir)
		exit_code = process.wait()

	duration_s = time.monotonic() - start_time
	duration_min = duration_s / 60.0

	print(f"🏁 Proceso finalizado con código de salida: {exit_code} (Duración: {duration_min:.2f} min)")

	# Leer telemetría final para resumir en el reporte
	telemetry_path = os.path.join(exp_dir, "telemetry.jsonl")
	last_epoch = 0
	last_loss = "N/A"
	last_acc_global = "N/A"

	if os.path.exists(telemetry_path):
		try:
			with open(telemetry_path, encoding="utf-8") as tf:
				for line in tf:
					line_data = json.loads(line)
					if line_data.get("type") == "epoch":
						last_epoch = line_data.get("epoch", last_epoch)
						last_loss = f"{line_data.get('loss_avg', 0.0):.4f}"
						last_acc_global = f"{line_data.get('acc_homeostasis', 0.0):.2f}%"

						# Intentar leer accuracies específicas de operadores desde la config del experimento
						exp_config_path = os.path.join(exp_dir, "config.json")
						if os.path.exists(exp_config_path):
							with open(exp_config_path, encoding="utf-8") as cf:
								exp_config = json.load(cf)
								list(exp_config.get("operators", {}).keys())
								# Para simular las claves logged_accs
								# (buscamos en el objeto si tiene keys correspondientes a acc_suma, acc_resta, etc.)
								# Nota: telemetry escribe acc_homeostasis, acc_concept, etc.
								# En train_generic registramos estas en el log_epoch del logger.
		except Exception as e:
			print(f"⚠️ Error al leer telemetría para reporte: {e}")

	# Crear reporte Markdown para el buzón
	result_str = "PASS" if exit_code == 0 else "FAIL"
	report_filename = f"EXP_{task_id}_report.md"
	report_path = os.path.join(inbox_dir, report_filename)

	report_md = f"""# 📨 Reporte de Experimento: EXP_{task_id}
*   **Nombre**: {task_name}
*   **Fecha/Hora**: {datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")}
*   **Resultado**: {result_str} (Código de salida: {exit_code})
*   **Duración**: {duration_min:.2f} minutos
*   **Épocas completadas**: {last_epoch}
*   **Última pérdida promedio**: {last_loss}
*   **Última precisión global en test**: {last_acc_global}

> [!NOTE]
> Reporte autogenerado por el Minion Scheduler. El log completo se encuentra en `storage/experiments/EXP_{task_id}/train.log`.
"""

	with open(report_path, "w", encoding="utf-8") as rf:
		rf.write(report_md)

	print(f"📩 Reporte guardado en el buzón: {report_path}")

	# Actualizar cola en experiment_queue.yaml
	queue_data = load_queue(queue_path)
	completed_list = queue_data.setdefault("completed", [])

	# Comprobar si ya existe una entrada para este ID para evitar duplicados
	existing_idx = next((i for i, item in enumerate(completed_list) if item.get("id") == task_id), None)

	completed_entry = {
		"id": task_id,
		"result": result_str,
		"note": f"Pérdida: {last_loss} | Test Acc: {last_acc_global} | Duración: {duration_min:.1f} min",
	}

	if existing_idx is not None:
		completed_list[existing_idx] = completed_entry
	else:
		completed_list.append(completed_entry)

	queue_data["current_experiment"] = None
	queue_data["last_updated"] = datetime.now(UTC).astimezone().isoformat()
	save_queue(queue_path, queue_data)
	print("✅ Cola de experimentos actualizada. Siguiente tarea pendiente disponible.")


if __name__ == "__main__":
	run_next_experiment()
