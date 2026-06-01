import json
import os
import subprocess
import sys
import time

import torch
import yaml

# Añadir base dir al path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import microscope

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.translator import SovereignTranslator

# Códigos ANSI para colores
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"


def clear_screen():
	os.system("cls" if os.name == "nt" else "clear")


def find_specimens():
	"""Busca todos los archivos .pt de especímenes en curriculum, archive y experimentos."""
	storage_dir = os.path.join(base_dir, "storage", "curriculum")
	archive_dir = os.path.join(storage_dir, "archive")
	experiments_dir = os.path.join(base_dir, "storage", "experiments")

	specimens = []

	# Buscar en storage/curriculum/
	if os.path.exists(storage_dir):
		for f in os.listdir(storage_dir):
			if f.endswith(".pt") or f.endswith(".bin"):
				specimens.append(os.path.join(storage_dir, f))

	# Buscar en storage/curriculum/archive/
	if os.path.exists(archive_dir):
		for f in os.listdir(archive_dir):
			if f.endswith(".pt") or f.endswith(".bin"):
				specimens.append(os.path.join(archive_dir, f))

	# Buscar en storage/experiments/*/best_agent.pt
	if os.path.exists(experiments_dir):
		for exp_id in os.listdir(experiments_dir):
			best_agent = os.path.join(experiments_dir, exp_id, "best_agent.pt")
			if os.path.exists(best_agent):
				specimens.append(best_agent)

	specimens.sort()
	return specimens


def select_specimen(specimens):
	"""Muestra el menú de selección de espécimen."""
	while True:
		clear_screen()
		print(f"{C_BOLD}{C_CYAN}================================================================================")
		print("🔬          CENTRO DE REGISTRO Y SELECCIÓN DE ESPECÍMENES DEL BÚNKER            ")
		print(f"================================================================================{C_RESET}")
		print(f"{C_BOLD}Selecciona el espécimen de silicio sobre el que deseas enfocar el microscopio:{C_RESET}\n")

		print(f"  [{C_GREEN}0{C_RESET}] Sujeto Virgen de Control (Inicialización Aleatoria)")

		for idx, path in enumerate(specimens, 1):
			name = os.path.basename(path)
			if "best_agent.pt" in path:
				parent = os.path.basename(os.path.dirname(path))
				display_name = f"{parent}/best_agent.pt"
				folder = "Arena de Experimento"
			else:
				display_name = name
				folder = "Archivo Frío" if "archive" in path else "Arena Activa"
			size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
			print(f"  [{C_GREEN}{idx}{C_RESET}] {C_BOLD}{display_name}{C_RESET} ({folder}) | Tamaño: {size_kb:.1f} KB")

		print(f"\n  [{C_RED}B{C_RESET}] Volver al menú principal")

		choice = input(f"\n{C_YELLOW}Introduce número o letra de tu elección: {C_RESET}").strip()

		if choice.upper() == "B":
			return None, None

		try:
			idx_choice = int(choice)
			if idx_choice == 0:
				return "random", "Sujeto Virgen de Control"
			elif 1 <= idx_choice <= len(specimens):
				path = specimens[idx_choice - 1]
				name = os.path.basename(path)
				if "best_agent.pt" in path:
					name = f"{os.path.basename(os.path.dirname(path))}/best_agent"
				return path, name
		except ValueError:
			pass

		input(f"\n{C_RED}Selección inválida. Presiona Enter para volver a intentar...{C_RESET}")


def run_microscope_menu(specimen_path, specimen_name, model, translator):
	"""Muestra el menú de lentes de diagnóstico para el modelo cargado."""
	while True:
		clear_screen()
		print(f"{C_BOLD}{C_CYAN}================================================================================")
		print(f"🔬            MICROSCOPIO COGNITIVO - ENFOQUE: {C_YELLOW}{specimen_name.upper()}{C_CYAN}             ")
		print(f"================================================================================{C_RESET}")
		print(f"{C_BOLD}Selecciona la lente o el test clínico a realizar:{C_RESET}\n")

		print(f"  [{C_GREEN}1{C_RESET}] Lente 1: Estructura Sináptica Ternaria (Conteo de pesos)")
		print(f"  [{C_GREEN}2{C_RESET}] Lente 2: Dinámica Homeostática de Silicio (Simulación de estrés)")
		print(f"  [{C_GREEN}3{C_RESET}] Lente 3: Sonda Lingüística de Capa 1 (Test de señalización y empatía)")
		print(f"  [{C_GREEN}4{C_RESET}] Lente 4: Espectro SVD (Valores singulares de atención)")
		print(f"  [{C_GREEN}5{C_RESET}] {C_BOLD}Ejecutar Diagnóstico Completo{C_RESET} (Todas las lentes secuenciales)")
		print(f"  [{C_GREEN}6{C_RESET}] {C_MAGENTA}Cambiar de Espécimen / Volver al listado{C_RESET}")

		choice = input(f"\n{C_YELLOW}Selecciona una opción: {C_RESET}").strip()

		if choice == "6" or choice.upper() == "Q":
			return

		clear_screen()

		if choice == "1":
			microscope.inspect_synapses(model)
		elif choice == "2":
			microscope.inspect_homeostasis(model)
		elif choice == "3":
			microscope.inspect_linguistics(model, translator)
		elif choice == "4":
			microscope.inspect_svd(model)
		elif choice == "5":
			print(f"Ejecutando suite completa para: {C_BOLD}{specimen_name}{C_RESET}")
			microscope.inspect_synapses(model)
			microscope.inspect_homeostasis(model)
			microscope.inspect_linguistics(model, translator)
			microscope.inspect_svd(model)
		else:
			print(f"{C_RED}Opción inválida.{C_RESET}")

		input(f"\n{C_YELLOW}Presiona Enter para volver al menú de lentes... {C_RESET}")


def inspect_specimens(translator, vocab_embeddings):
	while True:
		specimens = find_specimens()
		path, name = select_specimen(specimens)
		if path is None:
			break

		# Cargar el modelo
		clear_screen()
		print(f"{C_CYAN}Cargando espécimen: {C_BOLD}{name}...{C_RESET}")

		# Determinamos dimensiones y capas según nombre/especificaciones del modelo (por defecto 256/4)
		hidden_dim = 256
		num_layers = 4

		# Si es un experimento reciente, podemos leer su config.json para cargar las dimensiones exactas
		if "best_agent" in name:
			exp_id = name.split("/")[0]
			config_path = os.path.join(base_dir, "storage", "experiments", exp_id, "config.json")
			if os.path.exists(config_path):
				try:
					with open(config_path, encoding="utf-8") as cf:
						cdata = json.load(cf)
						hidden_dim = cdata.get("hidden_dim", hidden_dim)
						num_layers = cdata.get("num_layers", num_layers)
						print(f"📐 Config de modelo detectada: hidden_dim={hidden_dim}, num_layers={num_layers}")
				except Exception:
					pass

		model = BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=hidden_dim, num_layers=num_layers)
		if path != "random" and os.path.exists(path):
			try:
				model.load_state_dict(torch.load(path, map_location="cpu"))
			except Exception as e:
				print(f"{C_RED}Error al cargar pesos del espécimen: {e}. Usando inicialización de control.{C_RESET}")
				input("\nPresiona Enter para continuar...")

		run_microscope_menu(path, name, model, translator)


def is_scheduler_running():
	"""Comprueba si el script minion_scheduler.py está ejecutándose."""
	try:
		res = subprocess.run(["pgrep", "-f", "minion_scheduler.py"], capture_output=True, text=True)
		return res.returncode == 0
	except Exception:
		return False


def plot_ascii_loss(losses: list[float], width: int = 40, height: int = 8) -> str:
	if not losses:
		return "No hay suficientes datos de pérdida para graficar."

	min_l = min(losses)
	max_l = max(losses)
	span = max_l - min_l if max_l != min_l else 1.0

	n = len(losses)
	if n > width:
		indices = [int(i) for i in range(0, n - 1, max(1, (n - 1) // width))]
		sampled_losses = [losses[i] for i in indices[:width]]
	else:
		sampled_losses = losses

	grid = [[" " for _ in range(len(sampled_losses))] for _ in range(height)]
	for col, val in enumerate(sampled_losses):
		row = int((val - min_l) / span * (height - 1))
		grid[height - 1 - row][col] = "*"

	lines = []
	for r in range(height):
		val_at_row = max_l - r / (height - 1) * span
		row_str = "".join(grid[r])
		lines.append(f"{val_at_row:5.2f} | {row_str}")
	lines.append("      └" + "─" * len(sampled_losses))
	return "\n".join(lines)


def show_orchestrator():
	queue_path = os.path.join(base_dir, "lab", "experiment_queue.yaml")

	while True:
		clear_screen()
		print(f"{C_BOLD}{C_CYAN}================================================================================")
		print("🎛️                  CENTRO DE MANDO Y ORQUESTADOR DE MINIONES                   ")
		print(f"================================================================================{C_RESET}")

		if not os.path.exists(queue_path):
			print(f"{C_RED}No se encuentra el archivo lab/experiment_queue.yaml{C_RESET}")
			input("\nPresiona Enter para volver...")
			return

		with open(queue_path, encoding="utf-8") as f:
			queue_data = yaml.safe_load(f)

		running = is_scheduler_running()
		curr_id = queue_data.get("current_experiment")
		status_str = f"{C_GREEN}RUNNING (EXP_{curr_id}){C_RESET}" if running else f"{C_YELLOW}IDLE{C_RESET}"

		print(f"Estado del Minion Runner: {status_str} | Última Actualización: {queue_data.get('last_updated', 'N/A')}\n")

		completed_list = queue_data.get("completed", [])
		completed_ids = {item["id"] for item in completed_list if "id" in item}
		queue_list = queue_data.get("queue", [])

		print(f"{C_BOLD}Cola de Experimentos:{C_RESET}")
		print(f"{C_BOLD}{'-' * 80}{C_RESET}")
		print(f"{C_BOLD}{'ID':6} | {'Nombre':35} | {'Estado':12} | {'Nota/Resultado':20}{C_RESET}")
		print(f"{'-' * 80}")

		for task in queue_list:
			tid = task.get("id")
			name = task.get("name", "N/A")[:35]

			if tid == curr_id and running:
				status = f"{C_GREEN}RUNNING{C_RESET}"
				note = "Entrenando..."
			elif tid in completed_ids:
				# Obtener nota
				centry = next((x for x in completed_list if x.get("id") == tid), None)
				result = centry.get("result", "SUCCESS") if centry else "SUCCESS"
				status = f"{C_GREEN}SUCCESS{C_RESET}" if result == "PASS" else f"{C_RED}FAIL{C_RESET}"
				note = centry.get("note", "")[:20] if centry else ""
			else:
				status = f"{C_YELLOW}PENDING{C_RESET}"
				note = "En cola"

			print(f"{tid:6} | {name:35} | {status:12} | {note:20}")
		print(f"{'-' * 80}\n")

		# Si hay un experimento corriendo, mostramos telemetría interactiva
		if running and curr_id:
			telemetry_path = os.path.join(base_dir, "storage", "experiments", f"EXP_{curr_id}", "telemetry.jsonl")
			if os.path.exists(telemetry_path):
				try:
					losses = []
					last_acc = 0.0
					with open(telemetry_path, encoding="utf-8") as tf:
						for line in tf:
							ldata = json.loads(line)
							if ldata.get("type") == "epoch":
								losses.append(ldata.get("loss_avg", 0.0))
								last_acc = ldata.get("acc_homeostasis", last_acc)

					print(f"{C_BOLD}Monitoreo de Telemetría (EXP_{curr_id}): Test Acc: {last_acc:.2f}%{C_RESET}")
					print(plot_ascii_loss(losses, width=50, height=6))
					print()
				except Exception as e:
					print(f"Leyendo telemetría... ({e})")

		print(f"Opciones:  [{C_GREEN}R{C_RESET}] Refrescar  [{C_GREEN}L{C_RESET}] Lanzar Minion Runner en background  [{C_RED}B{C_RESET}] Volver")

		opt = input(f"\n{C_YELLOW}Introduce opción: {C_RESET}").strip().upper()

		if opt == "B":
			break
		elif opt == "L":
			if running:
				print(f"\n{C_RED}El Minion Runner ya está ejecutándose en segundo plano.{C_RESET}")
				time.sleep(1.5)
			else:
				print(f"\n{C_GREEN}Iniciando Minion Scheduler en background...{C_RESET}")
				# Lanzar de forma completamente desacoplada
				log_file = os.path.join(base_dir, "storage", "minion_scheduler.log")
				with open(log_file, "a", encoding="utf-8") as lf:
					subprocess.Popen(
						[".venv/bin/python", "scripts/minion_scheduler.py"], stdout=lf, stderr=subprocess.STDOUT, cwd=base_dir, start_new_session=True
					)
				time.sleep(1.5)


def show_inbox():
	inbox_dir = os.path.join(base_dir, "lab", ".inbox")

	while True:
		clear_screen()
		print(f"{C_BOLD}{C_CYAN}================================================================================")
		print("📨                     BUZÓN DE REPORTES DE MINIONES                            ")
		print(f"================================================================================{C_RESET}")

		if not os.path.exists(inbox_dir):
			os.makedirs(inbox_dir, exist_ok=True)

		files = [f for f in os.listdir(inbox_dir) if f.endswith(".md") or f.endswith(".json")]
		files.sort()

		if not files:
			print(f"{C_YELLOW}El buzón de entrada está vacío. No hay reportes nuevos.{C_RESET}")
			print(f"\n  [{C_RED}B{C_RESET}] Volver")
			opt = input(f"\n{C_YELLOW}Introduce opción: {C_RESET}").strip().upper()
			if opt == "B":
				break
			continue

		print(f"Se han encontrado {len(files)} reportes sin archivar:\n")
		for idx, f in enumerate(files, 1):
			print(f"  [{C_GREEN}{idx}{C_RESET}] {C_BOLD}{f}{C_RESET}")

		print(f"\n  [{C_GREEN}A{C_RESET}] Archivar todos los reportes leídos")
		print(f"  [{C_RED}B{C_RESET}] Volver")

		choice = input(f"\n{C_YELLOW}Introduce opción: {C_RESET}").strip()

		if choice.upper() == "B":
			break
		elif choice.upper() == "A":
			# Archivar todos
			for f in files:
				src_path = os.path.join(inbox_dir, f)
				# Extraer exp_id, ej: EXP_016
				parts = f.split("_")
				if len(parts) >= 2:
					exp_id = f"{parts[0]}_{parts[1]}"
					dest_dir = os.path.join(base_dir, "storage", "experiments", exp_id)
					os.makedirs(dest_dir, exist_ok=True)
					dest_path = os.path.join(dest_dir, "report.md")
					os.rename(src_path, dest_path)
					print(f"📦 Archivado {f} -> {dest_path}")
			time.sleep(1.5)
		else:
			try:
				idx = int(choice)
				if 1 <= idx <= len(files):
					f = files[idx - 1]
					filepath = os.path.join(inbox_dir, f)

					# Mostrar reporte
					clear_screen()
					print(f"{C_BOLD}{C_CYAN}================================================================================")
					print(f"📄 REPORTE: {f}                                                               ")
					print(f"================================================================================{C_RESET}\n")

					with open(filepath, encoding="utf-8") as rf:
						print(rf.read())

					print(f"\n{C_BOLD}{'-' * 80}{C_RESET}")
					print(
						f"Opciones:  [{C_RED}D{C_RESET}] Eliminar reporte  [{C_GREEN}A{C_RESET}] Archivar reporte  [{C_GREEN}Enter{C_RESET}] Volver"
					)

					sub_opt = input(f"\n{C_YELLOW}Opción: {C_RESET}").strip().upper()
					if sub_opt == "D":
						os.remove(filepath)
						print("🔥 Reporte eliminado.")
						time.sleep(1.0)
					elif sub_opt == "A":
						parts = f.split("_")
						if len(parts) >= 2:
							exp_id = f"{parts[0]}_{parts[1]}"
							dest_dir = os.path.join(base_dir, "storage", "experiments", exp_id)
							os.makedirs(dest_dir, exist_ok=True)
							dest_path = os.path.join(dest_dir, "report.md")
							os.rename(filepath, dest_path)
							print(f"📦 Reporte archivado en: {dest_path}")
							time.sleep(1.0)
			except ValueError:
				pass


def main():
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	while True:
		clear_screen()
		print(f"{C_BOLD}{C_CYAN}================================================================================")
		print("🔬                 SIETCH LABORATORY PANEL DE CONTROL (v1.1)                    ")
		print(f"================================================================================{C_RESET}")
		print(f"{C_BOLD}Selecciona el módulo operacional:{C_RESET}\n")

		print(f"  [{C_GREEN}1{C_RESET}] 🔬 Microscopio Cognitivo (Inspección clínica de especímenes)")
		print(f"  [{C_GREEN}2{C_RESET}] 🎛️  Orquestador de Experimentos (Cola y telemetría de Miniones)")
		print(f"  [{C_GREEN}3{C_RESET}] 📨 Buzón del Sietch (Reportes leídos de .inbox/)")
		print(f"  [{C_RED}Q{C_RESET}] Salir del laboratorio")

		choice = input(f"\n{C_YELLOW}Introduce opción: {C_RESET}").strip().upper()

		if choice == "Q":
			print(f"\n{C_RED}Apagando luces del laboratorio... Bitácora cerrada.{C_RESET}")
			sys.exit(0)
		elif choice == "1":
			inspect_specimens(translator, vocab_embeddings)
		elif choice == "2":
			show_orchestrator()
		elif choice == "3":
			show_inbox()


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print(f"\n\n{C_RED}Panel interrumpido bruscamente. Apagando luces del laboratorio...{C_RESET}")
		sys.exit(0)
