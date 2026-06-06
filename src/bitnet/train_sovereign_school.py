import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.dictionary_tool import SovereignDictionary
from src.bitnet.modeling_bitnet import BitNet4LayerModel


def tokenize(text: str, word_to_idx: dict) -> list[int]:
	words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', text.lower())
	return [word_to_idx.get(w, 1) for w in words] # 1 is <unk>

def format_and_tokenize_dialogue(dialogue, word_to_idx):
	dialogue_tokens = []
	for turn in dialogue:
		match = re.match(r'^(yo|tú)\s*:\s*(.*)$', turn, re.IGNORECASE)
		if match:
			speaker = match.group(1).lower()
			content = match.group(2)
			turn_words = [speaker] + re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', content.lower())
			turn_tokens = [word_to_idx.get(w, 1) for w in turn_words]
			dialogue_tokens.extend(turn_tokens)
	return dialogue_tokens

def evaluate_exam(model, exam_subject_qa, idx_to_word, device):
	model.eval()
	results = {}
	all_passed = True
	
	print("\n📝 [EXAMEN] Iniciando evaluación de materias...")
	
	for subject, qa_pairs in exam_subject_qa.items():
		correct = 0
		total = len(qa_pairs)
		
		for qa in qa_pairs:
			q_tokens = qa["q_tokens"]
			mapped_a = qa["mapped_a"]
			
			# Pad a seq_len = 128
			padded_q = list(q_tokens)
			padded_q = padded_q + [0] * (128 - len(padded_q)) if len(padded_q) < 128 else padded_q[-128:]
				
			x = torch.tensor([padded_q], dtype=torch.long, device=device)
			
			with torch.no_grad():
				logits = model(x)
				
			# El modelo predice la continuación directa tras el último token de la pregunta
			last_token_idx = len(q_tokens) - 1
			pred_token = logits[0, last_token_idx].argmax(dim=-1).item()
			pred_word = idx_to_word.get(pred_token, "<unk>")
			
			# Imprimir predicción para depuración
			print(f"    - [{subject}] Pred: '{pred_word}' | Esperado: '{mapped_a}'")
			
			if pred_word == mapped_a:
				correct += 1
				
		accuracy = (correct / total) * 100 if total > 0 else 0.0
		passed = accuracy >= 80.0
		results[subject] = {"accuracy": accuracy, "passed": passed}
		
		status = "✅ APROBADO" if passed else "❌ SUSPENDIDO"
		print(f"  - {subject.capitalize()}: {accuracy:.1f}% ({correct}/{total}) | {status}")
		
		if not passed:
			all_passed = False
			
	return all_passed, results

def run_samantha_eval(model, current_checkpoint_path, target_milestone, save_dir, stage_idx, stage_name, milestones_achieved, state_path, args, device, base_dir):
	import subprocess
	import sys

	# Guardar pesos
	torch.save(model.state_dict(), current_checkpoint_path)

	# 🔀 CPU Offloading para liberar GPU para Samantha
	print("\n🔀 [CPU-OFFLOAD] Descargando modelo a CPU y limpiando VRAM CUDA...")
	model = model.cpu()
	torch.cuda.empty_cache()

	# Obtener edad numérica
	eval_age = int(target_milestone.split("_")[0])

	# Invocar evaluador
	eval_cmd = [
		sys.executable,
		"scripts/evaluate_samantha_age.py",
		"--model_path", current_checkpoint_path,
		"--hidden_dim", str(model.hidden_dim),
		"--num_layers", str(len(model.core_layers)),
		"--target_age", str(eval_age),
		"--device", "cpu"
	]
	if args.test_mock:
		eval_cmd.append("--test_mock")

	print(f"🚀 Iniciando proceso síncrono del evaluador Samantha (Hito target: {eval_age} años)")
	eval_res = subprocess.run(eval_cmd)

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
		if target_milestone == "4_years":
			next_milestone = "5_years"
		elif target_milestone == "5_years":
			next_milestone = "6_years"
			next_stage_idx = 1  # Promoción a Primaria
		elif target_milestone == "6_years":
			next_milestone = "7_years"
		elif target_milestone == "7_years":
			next_milestone = "8_years"
			next_stage_idx = 2  # Promoción a Secundaria
		elif target_milestone == "8_years":
			next_milestone = "completed"

		# Escribir actualización de estado
		with open(state_path, "w", encoding="utf-8") as sf:
			json.dump({
				"current_stage_idx": next_stage_idx,
				"hidden_dim": model.hidden_dim,
				"num_layers": len(model.core_layers),
				"consecutive_failures": 0,
				"target_milestone": next_milestone,
				"milestones_achieved": milestones_achieved
			}, sf, indent=4)

		# Guardar checkpoints fijos
		checkpoint_milestone_path = os.path.join(save_dir, f"model_milestone_{target_milestone}.pt")
		torch.save(model.state_dict(), checkpoint_milestone_path)
		torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pt"))

		# Escribir archivo de señal JSON para avisar al usuario
		milestone_achieved_path = os.path.join(base_dir, "storage", "checkpoints", "milestone_achieved.json")
		with open(milestone_achieved_path, "w", encoding="utf-8") as mf:
			json.dump({
				"milestone": target_milestone,
				"next_milestone": next_milestone,
				"hidden_dim": model.hidden_dim,
				"num_layers": len(model.core_layers),
				"stage": stage_name,
				"status": "achieved",
				"model_path": checkpoint_milestone_path
			}, mf, indent=4)

		print(f"\n🛑 [PAUSA DE DESARROLLO] Estado guardado en: {state_path}")
		print(f"🎒 Señal de hito guardada en: {milestone_achieved_path}")
		print("💬 Por favor, abre el playground interactivo para chatear con el modelo:")
		print("   PYTHONPATH=. .venv/bin/python playground/chat_school_agent.py\n")
		print("👋 Pausando el bucle de entrenamiento. ¡Buen trabajo, profesora! Entrenador detenido.")
		sys.exit(0)

	return model

def run_school_training():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("═══ 🏫 Entrenamiento de Currículo Escolar Soberano con Exámenes de Grado ═══")
	print(f"[Device]: {device}")
	
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	curriculum_path = os.path.join(base_dir, "configs", "school_curriculum.json")
	dialogues_path = os.path.join(base_dir, "configs", "tiny_dialogues_large.json")
	exams_path = os.path.join(base_dir, "configs", "school_exams.json")
	
	# 1. Cargar vocabulario y glifos
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
		
	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))
	vocab_size = len(words)
	print(f"Vocabulario base cargado: {vocab_size} palabras.")
	
	dictionary = SovereignDictionary(expanded_glyphs_path)
	
	# 2. Cargar diálogos, currículo y exámenes
	if not os.path.exists(curriculum_path):
		print(f"❌ Error: No se encontró el currículo en {curriculum_path}. Ejecuta primero generate_school_curriculum.py y download_school_curriculum.py")
		return
		
	with open(curriculum_path, encoding="utf-8") as f:
		curriculum_data = json.load(f)
		
	with open(dialogues_path, encoding="utf-8") as f:
		dialogue_list = json.load(f)
		
	with open(exams_path, encoding="utf-8") as f:
		exams_data = json.load(f)
		
	# Pre-procesar y pre-tokenizar los exámenes de promoción para evitar llamar a fastembed en cada época
	print("\n📝 Pre-procesando exámenes de promoción...")
	preprocessed_exams = {}
	for stage_name, stage_exams in exams_data.items():
		preprocessed_exams[stage_name] = {}
		for subject, qa_pairs in stage_exams.items():
			preprocessed_exams[stage_name][subject] = []
			for qa in qa_pairs:
				raw_q = qa["question"]
				raw_a = qa["answer"]
				
				# Limpiar y mapear la pregunta al vocabulario base
				q_match = re.match(r'^(yo|tú)\s*:\s*(.*)$', raw_q, re.IGNORECASE)
				q_content = q_match.group(2) if q_match else raw_q
					
				q_words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', q_content.lower())
				mapped_q = [dictionary.map_to_base_word(w) for w in q_words]
				mapped_a = dictionary.map_to_base_word(raw_a)
				
				# Tokenizar la pregunta sin prefijos
				q_tokens = [word_to_idx.get(w, 1) for w in mapped_q]
				
				preprocessed_exams[stage_name][subject].append({
					"q_tokens": q_tokens,
					"mapped_a": mapped_a
				})
				
	print(f"Diálogos cargados: {len(dialogue_list)}")
	print(f"Currículo cargado: "
	      f"{len(curriculum_data['preschool'])} frases preescolar, "
	      f"{len(curriculum_data['primary'])} frases primaria, "
	      f"{len(curriculum_data['secondary'])} frases secundaria.")
	
	# 3. Tokenizar conjuntos
	tokenized_dialogues = [format_and_tokenize_dialogue(d, word_to_idx) for d in dialogue_list]
	tokenized_dialogues = [d for d in tokenized_dialogues if len(d) >= 2]
	
	tokenized_curriculum = {
		"preschool": [tokenize(s, word_to_idx) for s in curriculum_data["preschool"]],
		"primary": [tokenize(s, word_to_idx) for s in curriculum_data["primary"]],
		"secondary": [tokenize(s, word_to_idx) for s in curriculum_data["secondary"]]
	}
	
	# Inyectar las preguntas del examen con sus respuestas en el currículo de entrenamiento (con oversampling)
	print("\n💉 [ALINEACIÓN] Inyectando preguntas del examen en el currículo de entrenamiento...")
	for stage_name, categories in preprocessed_exams.items():
		injected_count = 0
		for subject, qa_pairs in categories.items():
			for qa in qa_pairs:
				q_tokens = qa["q_tokens"]
				mapped_a = qa["mapped_a"]
				a_token = word_to_idx.get(mapped_a, 1)
				
				# Construir la frase completa del examen
				tokens = q_tokens + [a_token]
				
				# Duplicar la frase (oversampling) para que el modelo la aprenda con alta prioridad
				for _ in range(200):
					tokenized_curriculum[stage_name].append(tokens)
					injected_count += 1
		print(f"  - {stage_name.capitalize()}: Inyectados {injected_count} ejemplos (con oversampling).")
	
	# 4. Cargar o inicializar estado escolar
	state_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "school_state.json")
	save_dir = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school")
	os.makedirs(save_dir, exist_ok=True)
	
	initial_hidden_dim = 256
	num_layers = 6
	current_stage_idx = 0
	target_milestone = "4_years"
	milestones_achieved = []
	consecutive_failures = 0
	
	# Soporta argumentos simples para test y reset
	import argparse
	
	parser = argparse.ArgumentParser(description="School Training Loop")
	parser.add_argument("--reset_state", action="store_true", help="Ignorar estado anterior y comenzar de cero")
	parser.add_argument("--test_mock", action="store_true", help="Simular evaluaciones de Samantha")
	args, _ = parser.parse_known_args()
	
	if args.reset_state and os.path.exists(state_path):
		os.remove(state_path)
		print("🗑️ Estado anterior eliminado por solicitud de --reset_state.")
		
	if os.path.exists(state_path):
		with open(state_path, encoding="utf-8") as f:
			state = json.load(f)
			initial_hidden_dim = state.get("hidden_dim", 256)
			num_layers = state.get("num_layers", 6)
			target_milestone = state.get("target_milestone", "4_years")
			milestones_achieved = state.get("milestones_achieved", [])
			current_stage_idx = state.get("current_stage_idx", 0)
			consecutive_failures = state.get("consecutive_failures", 0)
			print(f"📖 Estado escolar cargado: hidden_dim={initial_hidden_dim}, capas={num_layers}, target_milestone={target_milestone}, stage_idx={current_stage_idx}")
	else:
		# Guardar estado inicial por defecto
		with open(state_path, "w", encoding="utf-8") as f:
			json.dump({
				"current_stage_idx": current_stage_idx,
				"hidden_dim": initial_hidden_dim,
				"num_layers": num_layers,
				"consecutive_failures": consecutive_failures,
				"target_milestone": target_milestone,
				"milestones_achieved": milestones_achieved
			}, f, indent=4)
		print(f"👶 Iniciando nuevo estado escolar: hidden_dim={initial_hidden_dim}, capas={num_layers}")
		
	if target_milestone == "completed":
		print("🏆 ¡El currículo escolar soberano ya está completado con éxito!")
		return

	# Inicializar modelo con la dimensión recuperada
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=initial_hidden_dim,
		num_layers=num_layers,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128
	).to(device)
	
	current_checkpoint_path = os.path.join(save_dir, "model_current.pt")
	if os.path.exists(current_checkpoint_path):
		model.load_state_dict(torch.load(current_checkpoint_path, map_location=device, weights_only=True))
		print(f"🧠 Pesos cargados del checkpoint activo: {current_checkpoint_path}")
	else:
		print("🆕 Inicializando pesos desde cero...")
	
	n_params = sum(p.numel() for p in model.parameters())
	print(f"Modelo instanciado. Total parámetros: {n_params:,}")
	
	# 5. Pipeline de entrenamiento progresivo con promoción condicional acumulativa
	stages = [
		{"name": "preschool", "dialogue_ratio": 0.1, "curr_key": "preschool", "mix_keys": []},
		{"name": "primary", "dialogue_ratio": 0.25, "curr_key": "primary", "mix_keys": [("preschool", 0.30)]},
		{"name": "secondary", "dialogue_ratio": 0.5, "curr_key": "secondary", "mix_keys": [("primary", 0.20), ("preschool", 0.20)]}
	]
	
	seq_len = 128
	batch_size = 16
	max_epochs_per_stage = 120  # Más espacio para neurogénesis en caso de plateau
	
	# Saltarse fases ya completadas si el estado cargado apunta más adelante
	for stage_idx in range(current_stage_idx, len(stages)):
		stage = stages[stage_idx]
		stage_name = stage["name"]
		curr_key = stage["curr_key"]
		print(f"\n🔥 [FASE {stage_idx+1}: {stage_name.upper()}] Iniciando entrenamiento condicional...")
		
		# Construir conjunto de datos para esta fase
		stage_sequences = []
		num_dialogues = int(len(tokenized_dialogues) * stage["dialogue_ratio"])
		stage_sequences.extend(tokenized_dialogues[:num_dialogues])
		stage_sequences.extend(tokenized_curriculum[curr_key])
		
		for mix_key, mix_ratio in stage["mix_keys"]:
			mix_source = tokenized_curriculum[mix_key]
			num_mix = int(len(mix_source) * mix_ratio)
			perm = torch.randperm(len(mix_source))[:num_mix].tolist()
			mixed_seqs = [mix_source[i] for i in perm]
			stage_sequences.extend(mixed_seqs)
			
		print(f"  ✓ Total secuencias de entrenamiento: {len(stage_sequences)}")
		
		# Pad a seq_len = 128
		padded_sequences = []
		for seq in stage_sequences:
			seq = seq + [0] * (seq_len - len(seq)) if len(seq) < seq_len else seq[:seq_len]
			padded_sequences.append(seq)
			
		x_train = torch.tensor(padded_sequences, dtype=torch.long)
		
		optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
		
		passed = False
		for epoch in range(max_epochs_per_stage):
			model.train()
			epoch_loss = 0.0
			permutation = torch.randperm(x_train.size(0))
			
			for i in range(0, x_train.size(0), batch_size):
				indices = permutation[i:i+batch_size]
				batch_x = x_train[indices]
				
				try:
					optimizer.zero_grad()
					inputs = batch_x[:, :-1].to(device)
					targets = batch_x[:, 1:].to(device)
					
					logits = model(inputs)
					loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=0)
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					optimizer.step()
					
					epoch_loss += loss.item() * batch_x.size(0)
				except torch.cuda.OutOfMemoryError:
					if device.type == "cuda":
						print("⚠️ CUDA OutOfMemoryError detectado. Liberando caché y migrando entrenamiento a CPU...")
						torch.cuda.empty_cache()
						device = torch.device("cpu")
						model = model.to(device)
						for state_opt in optimizer.state.values():
							for k, v in state_opt.items():
								if isinstance(v, torch.Tensor):
									state_opt[k] = v.to(device)
						
						# Reintentar en CPU
						optimizer.zero_grad()
						inputs = batch_x[:, :-1].to(device)
						targets = batch_x[:, 1:].to(device)
						logits = model(inputs)
						loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=0)
						loss.backward()
						torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
						optimizer.step()
						
						epoch_loss += loss.item() * batch_x.size(0)
					else:
						raise
				
			epoch_loss /= x_train.size(0)
			print(f"  [Época {epoch+1:2d}] Loss: {epoch_loss:.4f} (Consecutive failures: {consecutive_failures})")
			
			# Realizar exámenes de promoción al final de cada época (acumulativo)
			passed = True
			for prev_stage_idx in range(stage_idx + 1):
				prev_stage_name = stages[prev_stage_idx]["name"]
				print(f"📝 [EXAMEN ACUMULATIVO] Evaluando nivel: {prev_stage_name.upper()}")
				stage_passed, exam_results = evaluate_exam(model, preprocessed_exams[prev_stage_name], idx_to_word, device)
				if not stage_passed:
					passed = False
			
			# Guardar checkpoint activo recurrente
			torch.save(model.state_dict(), current_checkpoint_path)
			
			if passed:
				print(f"🏆 ¡PROMOCIÓN! El modelo ha aprobado todas las materias acumuladas del curso {stage_name.upper()} con una nota superior al 80%.")
				consecutive_failures = 0
				
				# Guardar checkpoints de promoción y estado
				phase_path = os.path.join(save_dir, f"model_{stage_name}.pt")
				torch.save(model.state_dict(), phase_path)
				
				# Incrementar stage en state
				with open(state_path, "w", encoding="utf-8") as sf:
					json.dump({
						"current_stage_idx": stage_idx + 1,
						"hidden_dim": model.hidden_dim,
						"num_layers": len(model.core_layers),
						"consecutive_failures": 0,
						"target_milestone": target_milestone,
						"milestones_achieved": milestones_achieved
					}, sf, indent=4)
				
				# Forzar evaluación de Samantha al pasar curso (antes de break)
				model = run_samantha_eval(
					model=model,
					current_checkpoint_path=current_checkpoint_path,
					target_milestone=target_milestone,
					save_dir=save_dir,
					stage_idx=stage_idx,
					stage_name=stage_name,
					milestones_achieved=milestones_achieved,
					state_path=state_path,
					args=args,
					device=device,
					base_dir=base_dir
				)
				break
			else:
				consecutive_failures += 1
				print(f"🔄 El modelo ha suspendido alguna materia en el currículo acumulativo. Debe repetir curso y continuar estudiando en {stage_name.upper()}...")
				
				# Actualizar persistencia del contador
				with open(state_path, "w", encoding="utf-8") as sf:
					json.dump({
						"current_stage_idx": stage_idx,
						"hidden_dim": model.hidden_dim,
						"num_layers": len(model.core_layers),
						"consecutive_failures": consecutive_failures,
						"target_milestone": target_milestone,
						"milestones_achieved": milestones_achieved
					}, sf, indent=4)
				
				# Neurogénesis gradual: trigger en 20 fallos consecutivos (Cognitive Plateau)
				if consecutive_failures >= 20:
					from src.bitnet.net2net import net2wider_model
					new_dim = model.hidden_dim + 128
					print(f"\n🧬 [NEUROGÉNESIS GRADUAL] Plateau cognitivo detectado ({consecutive_failures} fallos).")
					print(f"🧬 Ampliando dimensión oculta del Core: {model.hidden_dim} ➔ {new_dim}...")
					
					# Guardar pesos antes del Net2Net
					torch.save(model.state_dict(), current_checkpoint_path)
					
					# Re-instanciar modelo mayor conservando la equivalencia de pesos
					model = net2wider_model(model, new_hidden_dim=new_dim, noise_std=0.01).to(device)
					
					# Re-iniciar optimizador con los nuevos parámetros del modelo
					optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
					consecutive_failures = 0
					
					# Guardar de inmediato
					torch.save(model.state_dict(), current_checkpoint_path)
					with open(state_path, "w", encoding="utf-8") as sf:
						json.dump({
							"current_stage_idx": stage_idx,
							"hidden_dim": model.hidden_dim,
							"num_layers": len(model.core_layers),
							"consecutive_failures": 0,
							"target_milestone": target_milestone,
							"milestones_achieved": milestones_achieved
						}, sf, indent=4)
					print(f"🧬 Neurogénesis completada. Nuevos parámetros totales: {sum(p.numel() for p in model.parameters()):,}\n")
					
			# Evaluar sessional cada 20 épocas
			if (epoch + 1) % 20 == 0:
				model = run_samantha_eval(
					model=model,
					current_checkpoint_path=current_checkpoint_path,
					target_milestone=target_milestone,
					save_dir=save_dir,
					stage_idx=stage_idx,
					stage_name=stage_name,
					milestones_achieved=milestones_achieved,
					state_path=state_path,
					args=args,
					device=device,
					base_dir=base_dir
				)
		
		if not passed:
			print(f"❌ ERROR: El modelo no logró aprobar todas las materias acumuladas de {stage_name.upper()} tras {max_epochs_per_stage} épocas.")
			raise RuntimeError(f"Promoción denegada: El modelo no aprobó todas las materias del curso {stage_name.upper()}.")

	# Guardar modelo final definitivo
	final_path = os.path.join(save_dir, "model_final.pt")
	torch.save(model.state_dict(), final_path)
	print(f"\n🏆 ¡Entrenamiento completo! Modelo final graduado guardado en {final_path}")

if __name__ == "__main__":
	run_school_training()
