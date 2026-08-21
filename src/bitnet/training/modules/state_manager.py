import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

EXAM_MAX_FAILURES = 3  # suspensos del MISMO hito antes de ceder la decisión al operador
EXAM_REMEDIAL_FRACTION = 0.25  # fracción de la etapa que se repasa tras cada suspenso
EXAM_PAUSE_EXIT_CODE = 78  # contrato con la receta (pause_exit_code): el runner sella PAUSED


@dataclass
class EvalResult:
	"""Resultado de una evaluación de Samantha."""
	passed: bool
	needs_operator: bool
	model: object  # torch.nn.Module
	state: dict = field(default_factory=dict)
	checkpoint_path: str = ""
	milestone_name: str = ""
	stage_idx: int = 0
	stage_name: str = ""
	eval_age: int = 0
	attempt_num: int = 0
	next_milestone: str | None = None
	next_stage_idx: int = 0


def plan_exam_failure(state, stage_conf, target_milestone):
	"""Decide qué hacer con un suspenso: repaso dentro de la etapa o pausa del operador.

	La puerta de la etapa es el EXAMEN, no el contador de épocas: el suspenso
	retiene la etapa retrocediendo el contador un bloque de repaso, y el examen
	re-dispara solo al volver a alcanzar end_epoch. Al K-ésimo suspenso del mismo
	hito, la decisión pasa al operador. Devuelve (nuevo_estado, needs_operator).
	"""
	held = dict(state)
	failures = dict(held.get("exam_failures", {}))
	failures[target_milestone] = failures.get(target_milestone, 0) + 1
	needs_operator = failures[target_milestone] >= EXAM_MAX_FAILURES

	rewind = max(8, int(stage_conf["epochs"] * EXAM_REMEDIAL_FRACTION))
	held["current_epoch"] = stage_conf["end_epoch"] if needs_operator else max(stage_conf["start_epoch"], stage_conf["end_epoch"] - rewind + 1)
	held["current_stage_idx"] = stage_conf["stage_idx"]
	held["exam_failures"] = failures
	return held, needs_operator


def run_samantha_eval(
	model, current_checkpoint_path, target_milestone, save_dir, stage_idx, stage_name, milestones_achieved, state_path, args, device, base_dir, epoch,
	stage_conf=None,
):
	"""Evalúa el modelo con Samantha y devuelve EvalResult en lugar de sys.exit."""
	# Guardar pesos
	torch = __import__("torch")
	torch.save(model.state_dict(), current_checkpoint_path)

	# 🔀 CPU Offloading para liberar GPU para Samantha
	print("\n🔀 [CPU-OFFLOAD] Descargando modelo a CPU y limpiando VRAM CUDA...")
	model = model.cpu()
	__import__("torch").cuda.empty_cache()

	# Obtener edad numérica
	eval_age = int(target_milestone.split("_")[0])

	# Invocar evaluador
	eval_cmd = [
		sys.executable,
		"scripts/evaluate_samantha_age.py",
		"--model_path",
		current_checkpoint_path,
		"--hidden_dim",
		str(model.hidden_dim),
		"--num_layers",
		str(len(model.core_layers)),
		"--target_age",
		str(eval_age),
		"--device",
		"cpu",
		# Máscara de gateo REAL de la etapa (DL-009): Samantha permite exactamente
		# lo que el entrenamiento dejó producir. El evaluador cae al gateo legado
		# si el fichero no existe (checkpoints antiguos).
		"--stage_masks",
		os.path.join(save_dir, "stage_gate_masks.json"),
		"--stage_idx",
		str(stage_idx),
	]
	if args.test_mock:
		eval_cmd.append("--test_mock")

	print(f"🚀 Iniciando proceso síncrono del evaluador Samantha (Hito target: {eval_age} años)")
	env = dict(os.environ)
	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
	env["PYTHONPATH"] = project_root + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
	eval_res = subprocess.run(eval_cmd, env=env)

	# 🔀 GPU Reload
	print("🔀 [GPU-RELOAD] Retornando modelo a GPU CUDA...")
	model = model.to(device)

	# Comprobar si aprobó el hito
	if eval_res.returncode == 0:
		print(f"\n🏆 ¡HITO DE EDAD ALCANZADO! El alumno ha superado el hito de {eval_age} años.")
		if target_milestone not in milestones_achieved:
			milestones_achieved.append(target_milestone)

		# Avanzar target_milestone y actualizar stage
		next_milestone = None
		next_stage_idx = stage_idx
		if target_milestone == "2_years":
			next_milestone = "3_years"
			next_stage_idx = 2
		elif target_milestone == "3_years":
			next_milestone = "4_years"
			next_stage_idx = 3
		elif target_milestone == "4_years":
			next_milestone = "5_years"
			next_stage_idx = 4
		elif target_milestone == "5_years":
			next_milestone = "6_years"
			next_stage_idx = 5
		elif target_milestone == "6_years":
			next_milestone = "7_years"
			next_stage_idx = 6
		elif target_milestone == "7_years":
			next_milestone = "8_years"
			next_stage_idx = 7
		elif target_milestone == "8_years":
			next_milestone = "completed"
			next_stage_idx = 8

		# Escribir actualización de estado
		with open(state_path, encoding="utf-8") as sf:
			existing_state = json.load(sf)
		existing_state.update({
			"current_epoch": epoch + 1,
			"current_stage_idx": next_stage_idx,
			"hidden_dim": model.hidden_dim,
			"num_layers": len(model.core_layers),
			"exam_failures": existing_state.get("exam_failures", {}),
			"target_milestone": next_milestone,
			"milestones_achieved": milestones_achieved,
		})
		with open(state_path, "w", encoding="utf-8") as sf:
			json.dump(existing_state, sf, indent=4)

		# Guardar checkpoints fijos
		checkpoint_milestone_path = os.path.join(save_dir, f"model_milestone_{target_milestone}.pt")
		torch.save(model.state_dict(), checkpoint_milestone_path)
		torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pt"))
		torch.save(model.state_dict(), os.path.join(save_dir, f"model_{stage_name}.pt"))

		# Escribir archivo de señal JSON para avisar al usuario
		milestone_achieved_path = os.path.join(base_dir, "storage", "checkpoints", "milestone_achieved.json")
		with open(milestone_achieved_path, "w", encoding="utf-8") as mf:
			json.dump(
				{
					"milestone": target_milestone,
					"next_milestone": next_milestone,
					"hidden_dim": model.hidden_dim,
					"num_layers": len(model.core_layers),
					"stage": stage_name,
					"status": "achieved",
					"model_path": checkpoint_milestone_path,
				},
				mf,
				indent=4,
			)

		print(f"\n🛑 [PAUSA DE DESARROLLO] Estado guardado en: {state_path}")
		print(f"🎒 Señal de hito guardada en: {milestone_achieved_path}")
		print("💬 Por favor, abre el playground interactivo para chatear con el modelo:")
		print("   PYTHONPATH=. .venv/bin/python playground/chat_school_agent.py\n")
		print("👋 Pausando el bucle de entrenamiento. ¡Buen trabajo, profesora! Entrenador detenido.")

		return EvalResult(
			passed=True,
			needs_operator=False,
			model=model,
			state=existing_state,
			checkpoint_path=current_checkpoint_path,
			milestone_name=target_milestone,
			stage_idx=next_stage_idx,
			stage_name=stage_name,
			eval_age=eval_age,
			next_milestone=next_milestone,
			next_stage_idx=next_stage_idx,
		)
	else:
		# Suspenso = repaso, no fallo del sistema
		with open(state_path, encoding="utf-8") as sf:
			held_state = json.load(sf)
		held_state, needs_operator = plan_exam_failure(held_state, stage_conf, target_milestone)
		with open(state_path, "w", encoding="utf-8") as sf:
			json.dump(held_state, sf, indent=4)

		attempt_num = held_state["exam_failures"][target_milestone]
		milestone_failed_path = os.path.join(base_dir, "storage", "checkpoints", "milestone_failed.json")
		with open(milestone_failed_path, "w", encoding="utf-8") as mf:
			json.dump(
				{
					"milestone": target_milestone,
					"attempt": attempt_num,
					"max_attempts": EXAM_MAX_FAILURES,
					"status": "needs_operator" if needs_operator else "remedial",
					"resume_epoch": held_state["current_epoch"],
					"stage": stage_name,
				},
				mf,
				indent=4,
			)

		print(f"\n❌ [EXAMEN SUSPENDIDO] El alumno ha suspendido el hito de {eval_age} años (intento {attempt_num}/{EXAM_MAX_FAILURES}).")
		print(f"📋 Señal de suspenso guardada en: {milestone_failed_path}")
		if needs_operator:
			print("🛑 [PAUSA DEL OPERADOR] Suspensos agotados para este hito: revisa las calificaciones de Samantha y reanuda con `red-pill job resume`.")
		else:
			print(f"📚 [REPASO] La etapa se retiene: vuelta a la época {held_state['current_epoch']} y re-examen al alcanzar de nuevo la {stage_conf['end_epoch']}.")

		return EvalResult(
			passed=False,
			needs_operator=needs_operator,
			model=model,
			state=held_state,
			checkpoint_path=current_checkpoint_path,
			milestone_name=target_milestone,
			stage_idx=stage_idx,
			stage_name=stage_name,
			eval_age=eval_age,
			attempt_num=attempt_num,
		)


def trigger_neurogenesis(model, optimizer, new_dim, glyphs, device, current_checkpoint_path, state_path, epoch, milestones_achieved, target_milestone, strategy=None):
	"""Amplía la dimensión oculta del modelo vía Net2WiderNet.

	Args:
		strategy: TrainingStrategy para crear el nuevo optimizer. Si es None, usa AdamW estándar.
	"""
	import gc

	import torch

	from src.bitnet.growth.net2net import net2wider_model
	from src.bitnet.model.modeling_bitnet import BitNet4LayerModel

	print(f"\n🧬 [NEUROGÉNESIS EN CALIENTE] Época {epoch}: Ampliando dimensión oculta del Core: {model.hidden_dim} ➔ {new_dim}...")

	torch.save(model.state_dict(), current_checkpoint_path)

	# 1. CPU Offloading of old model and optimizer to free GPU VRAM
	model = model.cpu()
	for state_opt in optimizer.state.values():
		for k, v in state_opt.items():
			if isinstance(v, torch.Tensor):
				state_opt[k] = v.cpu()
	torch.cuda.empty_cache()
	gc.collect()

	# 2. Instantiate new model on CPU — respetando el modo de embedding del alumno
	# (brazo estándar DL-006: use_glyphs=False con tabla one-hot congelada).
	if getattr(model, "use_glyphs", True):
		new_model = BitNet4LayerModel(
			use_glyphs=True,
			glyph_table=glyphs,
			hidden_dim=new_dim,
			num_layers=len(model.core_layers),
			use_pos_embedding=True,
			is_causal=True,
			max_seq_len=128,
		).cpu()
	else:
		new_model = BitNet4LayerModel(
			use_glyphs=False,
			vocab_embeddings=model.vocab_embeddings.cpu().numpy(),
			hidden_dim=new_dim,
			num_layers=len(model.core_layers),
			use_pos_embedding=True,
			is_causal=True,
			max_seq_len=128,
		).cpu()

	# 3. Instantiate new optimizer on CPU with scaled learning rate
	lr_scale = 128.0 / new_dim
	if strategy is not None:
		new_optimizer = strategy.create_optimizer(new_model, lr_scale)
	else:
		new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=4e-4 * lr_scale, weight_decay=0.05)

	# 4. Perform net2wider mapping on CPU
	model = net2wider_model(
		model,
		new_hidden_dim=new_dim,
		noise_std=0.01,
		old_optimizer=optimizer,
		new_optimizer=new_optimizer
	)

	# 5. Move new model and optimizer back to active GPU/CPU device
	model = model.to(device)
	for state_opt in new_optimizer.state.values():
		for k, v in state_opt.items():
			if isinstance(v, torch.Tensor):
				state_opt[k] = v.to(device)

	# 6. Clean up temporary variables
	del new_model
	gc.collect()
	torch.cuda.empty_cache()

	torch.save(model.state_dict(), current_checkpoint_path)

	# F1 FIX: Preserve ALL state fields after neurogenesis (was writing partial state)
	with open(state_path, encoding="utf-8") as sf:
		current_state = json.load(sf)
	current_state.update({
		"current_epoch": epoch,
		"hidden_dim": model.hidden_dim,
		"num_layers": len(model.core_layers),
		"target_milestone": target_milestone,
		"milestones_achieved": milestones_achieved,
	})
	# F3: Reset plateau monitor after neurogrowth — fresh window for new dim
	current_state["best_val_loss"] = float("inf")
	current_state["epochs_without_improvement"] = 0
	with open(state_path, "w", encoding="utf-8") as sf:
		json.dump(current_state, sf, indent=4)
	print(f"🧬 Neurogénesis completada. Nuevos parámetros: {sum(p.numel() for p in model.parameters()):,}\n")
	return model, new_optimizer
