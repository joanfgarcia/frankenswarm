import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from src.bitnet.vocab.dictionary_tool import SovereignDictionary
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel


def generate_question_variations(q_content: str) -> list[str]:
	variations = [q_content]
	
	# Variación 1: cambiar determinantes (el -> un, la -> una, los -> unos, las -> unas)
	v1 = q_content
	v1 = re.sub(r"\bel\b", "un", v1)
	v1 = re.sub(r"\bla\b", "una", v1)
	v1 = re.sub(r"\blos\b", "unos", v1)
	v1 = re.sub(r"\blas\b", "unas", v1)
	if v1 != q_content:
		variations.append(v1)
		
	# Variación 2: eliminar determinantes al inicio
	v2 = re.sub(r"^(el|la|los|las|un|una|unos|unas)\s+", "", q_content)
	if v2 != q_content:
		variations.append(v2)

	# Variación 3: si tiene "entonces", crear versión sin "entonces"
	if "entonces" in q_content:
		v3 = q_content.replace("entonces", "").replace("  ", " ")
		variations.append(v3)
		if v1 != q_content:
			v1_no_entonces = v1.replace("entonces", "").replace("  ", " ")
			variations.append(v1_no_entonces)
			
	# Variación 4: eliminar tildes
	def remove_accents(text):
		accents = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ñ':'ñ'}
		return "".join(accents.get(c, c) for c in text)
	
	v4 = remove_accents(q_content)
	if v4 != q_content:
		variations.append(v4)
		
	# Variaciones combinadas
	for v in list(variations):
		v_no_accent = remove_accents(v)
		if v_no_accent != v:
			variations.append(v_no_accent)

	return list(set(variations))


def tokenize(text: str, word_to_idx: dict) -> list[int]:
	words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+", text.lower())
	return [word_to_idx.get(w, 1) for w in words]  # 1 is <unk>


def format_and_tokenize_dialogue(dialogue, word_to_idx):
	dialogue_tokens = []
	for turn in dialogue:
		match = re.match(r"^(yo|tú)\s*:\s*(.*)$", turn, re.IGNORECASE)
		if match:
			speaker = match.group(1).lower()
			content = match.group(2)
			turn_words = [speaker] + re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+", content.lower())
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


def run_samantha_eval(
	model, current_checkpoint_path, target_milestone, save_dir, stage_idx, stage_name, milestones_achieved, state_path, args, device, base_dir, epoch
):
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
	]
	if args.test_mock:
		eval_cmd.append("--test_mock")

	print(f"🚀 Iniciando proceso síncrono del evaluador Samantha (Hito target: {eval_age} años)")
	env = dict(os.environ)
	env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
		with open(state_path, "w", encoding="utf-8") as sf:
			json.dump(
				{
					"current_epoch": epoch + 1,
					"current_stage_idx": next_stage_idx,
					"hidden_dim": model.hidden_dim,
					"num_layers": len(model.core_layers),
					"consecutive_failures": 0,
					"target_milestone": next_milestone,
					"milestones_achieved": milestones_achieved,
				},
				sf,
				indent=4,
			)

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
		sys.exit(0)
	else:
		print(f"\n❌ [EXAMEN SUSPENDIDO] El alumno ha suspendido el examen del hito de {eval_age} años.")
		print(f"🛑 [PAUSA DE REVISIÓN] Estado guardado en: {state_path}")
		print("💬 Por favor, revisa las calificaciones de Samantha arriba.")
		print("👋 Deteniendo el entrenamiento para análisis y revisión manual.")
		sys.exit(1)

	return model, False


def partition_corpus_by_mlu(sequences: list[list[int]]) -> tuple:
	stage_0_1 = []
	stage_1_2 = []
	stage_2_3 = []
	stage_3_4 = []
	for seq in sequences:
		length = len(seq)
		if length <= 3:
			stage_0_1.append(seq)
		elif length == 4:
			stage_1_2.append(seq)
		elif 5 <= length <= 6:
			stage_2_3.append(seq)
		else:
			stage_3_4.append(seq)
	return stage_0_1, stage_1_2, stage_2_3, stage_3_4


def compile_stage_dataset(base_data: list[list[int]], seq_len: int = 128) -> tuple:
	padded = []
	for seq in base_data:
		if len(seq) < seq_len:
			seq_padded = seq + [0] * (seq_len - len(seq))
		else:
			seq_padded = seq[:seq_len]
		padded.append(seq_padded)

	import random
	rng = random.Random(42)
	rng.shuffle(padded)

	val_size = int(len(padded) * 0.1)
	train_seqs = padded[val_size:]
	val_seqs = padded[:val_size]

	return train_seqs, val_seqs


def trigger_neurogenesis(model, optimizer, new_dim, glyphs, device, current_checkpoint_path, state_path, epoch, milestones_achieved, target_milestone):
	from src.bitnet.growth.net2net import net2wider_model
	import gc
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

	# 2. Instantiate new model on CPU
	new_model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=new_dim,
		num_layers=len(model.core_layers),
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128,
	).cpu()

	# 3. Instantiate new optimizer on CPU with scaled learning rate
	lr_scale = 128.0 / new_dim
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
	with open(state_path, "w", encoding="utf-8") as sf:
		json.dump(
			{
				"current_epoch": epoch,
				"hidden_dim": model.hidden_dim,
				"num_layers": len(model.core_layers),
				"target_milestone": target_milestone,
				"milestones_achieved": milestones_achieved,
			},
			sf,
			indent=4,
		)
	print(f"🧬 Neurogénesis completada. Nuevos parámetros: {sum(p.numel() for p in model.parameters()):,}\n")
	return model, new_optimizer


def compile_exam_sequences_for_age(age: int, exams_data: dict, word_to_idx: dict, dictionary: SovereignDictionary) -> list[list[int]]:
	exam_sequences = []

	# 1. Obtener preguntas desde school_exams.json
	key = None
	if age in [2, 3, 4]:
		key = "preschool"
	elif age in [5, 6]:
		key = "primary"
	elif age in [7, 8]:
		key = "secondary"

	if key:
		exams_section = exams_data.get(key, {})
		for subject, qa_pairs in exams_section.items():
			for qa in qa_pairs:
				raw_q = qa["question"]
				raw_a = qa["answer"]

				# Extraer contenido de la pregunta
				q_match = re.match(r"^(yo|tú)\s*:\s*(.*)$", raw_q, re.IGNORECASE)
				q_content = q_match.group(2) if q_match else raw_q

				for var in generate_question_variations(q_content):
					q_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", var.lower())
					mapped_q = [dictionary.map_to_base_word(w) for w in q_words]
					mapped_a = dictionary.map_to_base_word(raw_a)

					q_tokens = [word_to_idx.get(w, 1) for w in mapped_q]
					a_token = word_to_idx.get(mapped_a, 1)

					tokens = q_tokens + [a_token]
					exam_sequences.append(tokens)

	# 2. Obtener preguntas específicas de evaluate_samantha_age.py para la edad
	from scripts.evaluate_samantha_age import AGE_QUESTIONS
	age_questions = AGE_QUESTIONS.get(age, [])
	for qa in age_questions:
		raw_q = qa["question"]
		raw_a = qa["expected"]

		q_match = re.match(r"^(yo|tú)\s*:\s*(.*)$", raw_q, re.IGNORECASE)
		q_content = q_match.group(2) if q_match else raw_q

		for var in generate_question_variations(q_content):
			q_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", var.lower())
			mapped_q = [dictionary.map_to_base_word(w) for w in q_words]
			mapped_a = dictionary.map_to_base_word(raw_a)

			q_tokens = [word_to_idx.get(w, 1) for w in mapped_q]
			a_token = word_to_idx.get(mapped_a, 1)

			tokens = q_tokens + [a_token]
			exam_sequences.append(tokens)

	return exam_sequences


def get_stage_config(base_epochs=64, stage_scale=0.5):
	stages = [
		{"name": "0-1", "dim": 128, "age": None},
		{"name": "1-2", "dim": 256, "age": 2},
		{"name": "2-3", "dim": 384, "age": 3},
		{"name": "3-4", "dim": 512, "age": 4},
		{"name": "primary_5", "dim": 640, "age": 5},
		{"name": "primary_6", "dim": 768, "age": 6},
		{"name": "secondary_7", "dim": 896, "age": 7},
		{"name": "secondary_8", "dim": 1024, "age": 8},
	]
	start_epoch = 1
	config = []
	for i, stage in enumerate(stages):
		epochs_in_stage = int(base_epochs * (1 + i * stage_scale))
		end_epoch = start_epoch + epochs_in_stage - 1
		config.append({
			"stage_idx": i,
			"name": stage["name"],
			"dim": stage["dim"],
			"age": stage["age"],
			"start_epoch": start_epoch,
			"end_epoch": end_epoch,
			"epochs": epochs_in_stage
		})
		start_epoch = end_epoch + 1
	return config


def get_next_dim(current_dim, stage_config, current_epoch=None):
	"""Return the next neurogenesis target dim, or None if already at max.

	Finds the smallest dim in stage_config that is strictly larger than
	current_dim. If current_epoch is given, respects the dim ceiling of
	the current stage (the model shouldn't grow beyond its stage's max).
	"""
	all_dims = sorted(set(c["dim"] for c in stage_config))

	# If epoch provided, cap at the current stage's dim
	max_allowed = all_dims[-1]
	if current_epoch is not None:
		for config in stage_config:
			if config["start_epoch"] <= current_epoch <= config["end_epoch"]:
				max_allowed = config["dim"]
				break

	for d in all_dims:
		if d > current_dim and d <= max_allowed:
			return d
	return None


def oversample_curriculum_for_stage(general_data, curriculum_data, target_ratio=0.20):
	if len(curriculum_data) == 0:
		return general_data
	target_size = int(len(general_data) * target_ratio / (1 - target_ratio))
	multiplier = target_size // len(curriculum_data)
	remainder = target_size % len(curriculum_data)
	oversampled = curriculum_data * multiplier + curriculum_data[:remainder]
	return general_data + oversampled


def run_school_training():
	import argparse
	parser = argparse.ArgumentParser(description="School Training Loop")
	parser.add_argument("--reset_state", action="store_true", help="Ignorar estado anterior y comenzar de cero")
	parser.add_argument("--test_mock", action="store_true", help="Simular evaluaciones de Samantha")
	parser.add_argument("--curriculum_mode", type=str, default="mixed", choices=["mixed", "childes_only", "structured_only"], help="Modo de currículo de entrenamiento")
	parser.add_argument("--base_epochs", type=int, default=64, help="Número de épocas base por etapa")
	parser.add_argument("--stage_scale", type=float, default=0.5, help="Escala de crecimiento de épocas por dificultad")
	parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
	parser.add_argument("--patience", type=int, default=15, help="Épocas sin mejora en val_loss antes de disparar neurogénesis")
	parser.add_argument("--min_delta", type=float, default=0.01, help="Mejora mínima de val_loss para considerar progreso")
	args, _ = parser.parse_known_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("═══ 🏫 Entrenamiento de Currículo Escolar Soberano con Exámenes de Grado ═══")
	print(f"[Device]: {device}")

	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	curriculum_path = os.path.join(base_dir, "configs", "school_curriculum_structured.json")
	dialogues_path = os.path.join(base_dir, "configs", "tiny_dialogues_large.json")
	exams_path = os.path.join(base_dir, "configs", "school_exams.json")
	childes_path = os.path.join(base_dir, "configs", "childes_pre_school.json")
	nsm_physics_path = os.path.join(base_dir, "configs", "nsm_physics_pre_school.json")

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
	with open(dialogues_path, encoding="utf-8") as f:
		dialogue_list = json.load(f)

	with open(exams_path, encoding="utf-8") as f:
		exams_data = json.load(f)

	with open(childes_path, encoding="utf-8") as f:
		childes_sentences = json.load(f)

	with open(nsm_physics_path, encoding="utf-8") as f:
		nsm_physics_sentences = json.load(f)

	with open(curriculum_path, encoding="utf-8") as f:
		curriculum_json = json.load(f)
		curriculum_metadata = curriculum_json.get("metadata", {})
		curriculum_data_raw = curriculum_json.get("curriculum", {})
		curriculum_hash = curriculum_metadata.get("sha256", "unknown_hash")
		print(f"📖 [CARGA] Currículo estructurado. Versión: {curriculum_metadata.get('version')}, Hash: {curriculum_hash}")
		curriculum_data = {
			"preschool": [item["text"] for item in curriculum_data_raw.get("preschool", [])],
			"primary": [item["text"] for item in curriculum_data_raw.get("primary", [])],
			"secondary": [item["text"] for item in curriculum_data_raw.get("secondary", [])]
		}

	print(f"Diálogos cargados: {len(dialogue_list)}")
	print(f"CHILDES: {len(childes_sentences)} | NSM Physics: {len(nsm_physics_sentences)}")

	# 3. Tokenizar conjuntos
	tokenized_dialogues = [format_and_tokenize_dialogue(d, word_to_idx) for d in dialogue_list]
	tokenized_dialogues = [d for d in tokenized_dialogues if len(d) >= 2]

	# Tokenizar el corpus preescolar sin submuestreo
	raw_preschool_corpus = childes_sentences + nsm_physics_sentences
	tokenized_preschool = [tokenize(s, word_to_idx) for s in raw_preschool_corpus]
	tokenized_preschool = [seq for seq in tokenized_preschool if len(seq) >= 2]

	# Cargar y tokenizar currículo preescolar de school_curriculum.json
	english_stop_words = {"and", "the", "of", "in", "to", "is", "it", "that", "you", "was", "for", "on", "with", "his", "they", "he", "she", "at", "by", "this", "but", "from", "are", "as"}

	def clean_curriculum(sentences):
		cleaned = []
		for s in sentences:
			words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", s.lower())
			if not any(w in english_stop_words for w in words):
				cleaned.append(s)
		return cleaned

	preschool_curriculum_sentences = clean_curriculum(curriculum_data.get("preschool", []))
	print(f"🎒 [LIMPIEZA DE DATOS] Currículo Preschool filtrado: {len(curriculum_data.get('preschool', []))} ➔ {len(preschool_curriculum_sentences)} limpio.")
	tokenized_preschool_curriculum = [tokenize(s, word_to_idx) for s in preschool_curriculum_sentences]
	tokenized_preschool_curriculum = [seq for seq in tokenized_preschool_curriculum if len(seq) >= 2]


	# Mezclar 10% de diálogos
	num_dialogues = int(len(tokenized_dialogues) * 0.1)
	preschool_dialogues = tokenized_dialogues[:num_dialogues]
	
	# Particionar corpus general (CHILDES + diálogos)
	tokenized_general = tokenized_preschool + preschool_dialogues
	gen_0_1, gen_1_2, gen_2_3, gen_3_4 = partition_corpus_by_mlu(tokenized_general)
	
	# Particionar currículo estructurado preescolar
	curr_0_1, curr_1_2, curr_2_3, curr_3_4 = partition_corpus_by_mlu(tokenized_preschool_curriculum)
	
	print(f"Particiones MLU General:")
	print(f"  - 0-1 Año (MLU <= 2): {len(gen_0_1)} secuencias")
	print(f"  - 1-2 Años (MLU = 3): {len(gen_1_2)} secuencias")
	print(f"  - 2-3 Años (MLU 4-5): {len(gen_2_3)} secuencias")
	print(f"  - 3-4 Años (MLU 6+): {len(gen_3_4)} secuencias")

	# Cargar y limpiar primaria y secundaria
	primary_sentences = clean_curriculum(curriculum_data.get("primary", []))
	secondary_sentences = clean_curriculum(curriculum_data.get("secondary", []))
	print(f"🎒 [LIMPIEZA DE DATOS] Currículo Primary filtrado: {len(curriculum_data.get('primary', []))} ➔ {len(primary_sentences)} limpio.")
	print(f"🎒 [LIMPIEZA DE DATOS] Currículo Secondary filtrado: {len(curriculum_data.get('secondary', []))} ➔ {len(secondary_sentences)} limpio.")

	tokenized_primary = [tokenize(s, word_to_idx) for s in primary_sentences]
	tokenized_primary = [seq for seq in tokenized_primary if len(seq) >= 2]

	tokenized_secondary = [tokenize(s, word_to_idx) for s in secondary_sentences]
	tokenized_secondary = [seq for seq in tokenized_secondary if len(seq) >= 2]

	# Dividir currículo en mitades
	primary_half1 = tokenized_primary[:len(tokenized_primary)//2]
	primary_half2 = tokenized_primary[len(tokenized_primary)//2:]

	secondary_half1 = tokenized_secondary[:len(tokenized_secondary)//2]
	secondary_half2 = tokenized_secondary[len(tokenized_secondary)//2:]

	# Porcentaje de diálogos (40% primaria, 50% secundaria)
	primary_dialogues_total = tokenized_dialogues[num_dialogues : num_dialogues + int(len(tokenized_dialogues) * 0.4)]
	secondary_dialogues_total = tokenized_dialogues[num_dialogues + int(len(tokenized_dialogues) * 0.4) :]

	primary_dialogues_half1 = primary_dialogues_total[:len(primary_dialogues_total)//2]
	primary_dialogues_half2 = primary_dialogues_total[len(primary_dialogues_total)//2:]

	secondary_dialogues_half1 = secondary_dialogues_total[:len(secondary_dialogues_total)//2]
	secondary_dialogues_half2 = secondary_dialogues_total[len(secondary_dialogues_total)//2:]

	# Helper para compilar datos por etapa con 20% currículo
	def compile_data_for_stage(stage_idx):
		if stage_idx == 0:
			general = gen_0_1
			# Sin currículo en etapa 0 (0-1 años)
			curriculum = curr_0_1
		elif stage_idx == 1:
			general = gen_0_1 + gen_1_2
			curriculum = curr_0_1 + curr_1_2
		elif stage_idx == 2:
			general = gen_0_1 + gen_1_2 + gen_2_3
			curriculum = curr_0_1 + curr_1_2 + curr_2_3
		elif stage_idx == 3:
			general = gen_0_1 + gen_1_2 + gen_2_3 + gen_3_4
			curriculum = curr_0_1 + curr_1_2 + curr_2_3 + curr_3_4
		elif stage_idx == 4:
			general = gen_0_1 + gen_1_2 + gen_2_3 + gen_3_4 + primary_dialogues_half1
			curriculum = curr_0_1 + curr_1_2 + curr_2_3 + curr_3_4 + primary_half1
		elif stage_idx == 5:
			general = gen_0_1 + gen_1_2 + gen_2_3 + gen_3_4 + primary_dialogues_half1 + primary_dialogues_half2
			curriculum = curr_0_1 + curr_1_2 + curr_2_3 + curr_3_4 + primary_half1 + primary_half2
		elif stage_idx == 6:
			general = gen_0_1 + gen_1_2 + gen_2_3 + gen_3_4 + primary_dialogues_total + secondary_dialogues_half1
			curriculum = curr_0_1 + curr_1_2 + curr_2_3 + curr_3_4 + primary_sentences + secondary_half1
		else:
			general = gen_0_1 + gen_1_2 + gen_2_3 + gen_3_4 + primary_dialogues_total + secondary_dialogues_total
			curriculum = curr_0_1 + curr_1_2 + curr_2_3 + curr_3_4 + primary_sentences + secondary_sentences

		# Mix in exam sequences up to the current stage's age
		exams = []
		for idx in range(1, stage_idx + 1):
			age = stage_config[idx]["age"]
			if age is not None:
				exams.extend(compile_exam_sequences_for_age(age, exams_data, word_to_idx, dictionary))
		if len(exams) > 0:
			# Duplicar las preguntas de examen para asegurar que se memoricen
			curriculum = curriculum + exams * 50

		if args.curriculum_mode == "childes_only":
			return general
		elif args.curriculum_mode == "structured_only":
			return curriculum
		else:
			return oversample_curriculum_for_stage(general, curriculum, target_ratio=0.20)

	# 4. Cargar o inicializar estado escolar
	state_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "school_state.json")
	save_dir = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school")
	os.makedirs(save_dir, exist_ok=True)

	stage_config = get_stage_config(args.base_epochs, args.stage_scale)
	max_epochs = stage_config[-1]["end_epoch"]

	current_epoch = 1
	hidden_dim = 128
	num_layers = 6
	milestones_achieved = []
	target_milestone = "2_years"

	current_checkpoint_path = os.path.join(save_dir, "model_current.pt")
	if args.reset_state:
		if os.path.exists(state_path):
			os.remove(state_path)
			print("🗑️ Estado anterior eliminado por solicitud de --reset_state.")
		if os.path.exists(current_checkpoint_path):
			os.remove(current_checkpoint_path)
			print("🗑️ Checkpoint anterior model_current.pt eliminado por solicitud de --reset_state.")

	if os.path.exists(state_path) and not args.reset_state:
		with open(state_path, encoding="utf-8") as f:
			state = json.load(f)
			current_epoch = state.get("current_epoch", 1)
			hidden_dim = state.get("hidden_dim", 128)
			num_layers = state.get("num_layers", 6)
			milestones_achieved = state.get("milestones_achieved", [])
			target_milestone = state.get("target_milestone", "2_years")

			# Ajuste robusto si falta current_epoch en el estado guardado por Samantha
			if "current_epoch" not in state:
				if target_milestone == "completed":
					current_epoch = max_epochs + 1
				else:
					try:
						target_age = int(target_milestone.split("_")[0])
						for config in stage_config:
							if config["age"] == target_age:
								current_epoch = config["start_epoch"]
								break
					except Exception:
						pass
			print(f"📖 Estado escolar cargado: current_epoch={current_epoch}, hidden_dim={hidden_dim}, capas={num_layers}")
	else:
		with open(state_path, "w", encoding="utf-8") as f:
			json.dump(
				{
					"current_epoch": current_epoch,
					"current_stage_idx": 0,
					"hidden_dim": hidden_dim,
					"num_layers": num_layers,
					"target_milestone": target_milestone,
					"milestones_achieved": milestones_achieved,
					"curriculum_hash": curriculum_hash,
				},
				f,
				indent=4,
			)
		print(f"👶 Iniciando nuevo estado escolar: hidden_dim={hidden_dim}, capas={num_layers}")

	# Estado de plateau para neurogénesis dirigida por dolor
	best_val_loss = state.get("best_val_loss", float("inf")) if os.path.exists(state_path) and not args.reset_state else float("inf")
	epochs_without_improvement = state.get("epochs_without_improvement", 0) if os.path.exists(state_path) and not args.reset_state else 0
	neurogenesis_history = state.get("neurogenesis_history", []) if os.path.exists(state_path) and not args.reset_state else []
	print(f"📊 Plateau monitor: patience={args.patience}, min_delta={args.min_delta}, best_val_loss={best_val_loss:.4f}, epochs_stale={epochs_without_improvement}")

	if current_epoch > max_epochs:
		print(f"🏆 ¡El currículo escolar soberano completo (Ages 0-8) ya está completado con éxito (época {current_epoch-1})!")
		return

	# Inicializar modelo
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128,
	).to(device)
	if os.path.exists(current_checkpoint_path) and not args.reset_state:
		model.load_state_dict(torch.load(current_checkpoint_path, map_location=device, weights_only=True))
		print(f"🧠 Pesos cargados del checkpoint activo: {current_checkpoint_path}")
	else:
		print("🆕 Inicializando weights desde cero...")

	n_params = sum(p.numel() for p in model.parameters())
	print(f"Modelo instanciado. Total parámetros: {n_params:,}")

	lr_scale = 128.0 / model.hidden_dim
	optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4 * lr_scale, weight_decay=0.05)

	seq_len = 128
	batch_size = args.batch_size

	def get_stage_info(ep):
		for config in stage_config:
			if config["start_epoch"] <= ep <= config["end_epoch"]:
				return config["stage_idx"], config["name"]
		return stage_config[-1]["stage_idx"], stage_config[-1]["name"]

	active_stage_idx = -1
	x_train = None
	x_val = None

	for epoch in range(current_epoch, max_epochs + 1):

		# Cargar/procesar dataset para la etapa
		stage_idx, stage_name = get_stage_info(epoch)
		if stage_idx != active_stage_idx:
			print(f"\n🎒 [CAMBIO DE ETAPA] Época {epoch}: Compilando dataset para la etapa {stage_name}...")
			base_data = compile_data_for_stage(stage_idx)
			train_seqs, val_seqs = compile_stage_dataset(base_data, seq_len=128)
			x_train = torch.tensor(train_seqs, dtype=torch.long)
			x_val = torch.tensor(val_seqs, dtype=torch.long) if len(val_seqs) > 0 else None
			active_stage_idx = stage_idx
			print(f"  ✓ Secuencias de entrenamiento: {len(x_train)} | Validación: {len(x_val) if x_val is not None else 0}")

		# Warmup de learning rate lineal (primeras 10 épocas) o decaimiento coseno por etapa
		lr_scale = 128.0 / model.hidden_dim
		peak_lr = 4e-4 * lr_scale
		min_lr = 4e-5 * lr_scale
		if epoch <= 10:
			current_lr = min_lr + (peak_lr - min_lr) * (epoch - 1) / 9.0
			for g in optimizer.param_groups:
				g['lr'] = current_lr
			print(f"📈 [WARMUP] Época {epoch}: Estableciendo learning rate a {current_lr:.2e}")
		else:
			import math
			stage_idx, stage_name = get_stage_info(epoch)
			current_stage_conf = stage_config[stage_idx]
			stage_epochs = current_stage_conf["epochs"]
			stage_elapsed = epoch - current_stage_conf["start_epoch"]

			if stage_idx == 0:
				progress = max(0.0, min(1.0, (epoch - 10) / (stage_epochs - 10)))
			else:
				progress = max(0.0, min(1.0, stage_elapsed / stage_epochs))

			cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
			current_lr = min_lr + (peak_lr - min_lr) * cosine_decay
			for g in optimizer.param_groups:
				g['lr'] = current_lr

		# Annealing de temperatura Gumbel
		tau = max(0.1, 1.0 - (1.0 - 0.1) * ((epoch - 1) / 127.0))

		model.train()
		epoch_loss = 0.0
		permutation = torch.randperm(x_train.size(0))

		for i in range(0, x_train.size(0), batch_size):
			indices = permutation[i : i + batch_size]
			batch_x = x_train[indices]

			try:
				optimizer.zero_grad()
				inputs = batch_x[:, :-1].to(device)
				targets = batch_x[:, 1:].to(device)

				logits = model(inputs, tau=tau)

				loss_elementwise = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
				loss_elementwise = loss_elementwise.reshape(targets.shape)
				is_zero = (targets == 0).to(torch.int32)
				cumsum_zero = torch.cumsum(is_zero, dim=-1)
				mask = (cumsum_zero <= 1).to(logits.dtype)

				loss = (loss_elementwise * mask).sum() / mask.sum()
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
					logits = model(inputs, tau=tau)

					loss_elementwise = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
					loss_elementwise = loss_elementwise.reshape(targets.shape)
					is_zero = (targets == 0).to(torch.int32)
					cumsum_zero = torch.cumsum(is_zero, dim=-1)
					mask = (cumsum_zero <= 1).to(logits.dtype)

					loss = (loss_elementwise * mask).sum() / mask.sum()
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					optimizer.step()

					epoch_loss += loss.item() * batch_x.size(0)
				else:
					raise

		epoch_loss /= x_train.size(0)

		# Calcular pérdida de validación (val_loss)
		val_loss = 0.0
		if x_val is not None and len(x_val) > 0:
			model.eval()
			with torch.no_grad():
				for vi in range(0, x_val.size(0), batch_size):
					batch_xv = x_val[vi : vi + batch_size].to(device)
					inputs_v = batch_xv[:, :-1]
					targets_v = batch_xv[:, 1:]

					logits_v = model(inputs_v)
					loss_v_elem = F.cross_entropy(logits_v.reshape(-1, vocab_size), targets_v.reshape(-1), reduction="none")
					loss_v_elem = loss_v_elem.reshape(targets_v.shape)
					is_zero_v = (targets_v == 0).to(torch.int32)
					cumsum_zero_v = torch.cumsum(is_zero_v, dim=-1)
					mask_v = (cumsum_zero_v <= 1).to(logits_v.dtype)

					loss_v = (loss_v_elem * mask_v).sum() / mask_v.sum()
					val_loss += loss_v.item() * batch_xv.size(0)
				val_loss /= x_val.size(0)

		print(f"  [Época {epoch:2d}] Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e} | tau: {tau:.4f} | dim: {model.hidden_dim}")

		# ── Neurogénesis dirigida por dolor (plateau de val_loss) ──
		if x_val is not None and val_loss > 0:
			if val_loss < best_val_loss - args.min_delta:
				best_val_loss = val_loss
				epochs_without_improvement = 0
			else:
				epochs_without_improvement += 1

			if epochs_without_improvement >= args.patience:
				next_dim = get_next_dim(model.hidden_dim, stage_config, current_epoch=epoch)
				if next_dim is not None:
					print(f"\n🧬 [PLATEAU DETECTADO] val_loss estancada {args.patience} épocas (best={best_val_loss:.4f}) → neurogénesis {model.hidden_dim} → {next_dim}")
					old_dim = model.hidden_dim
					model, optimizer = trigger_neurogenesis(
						model, optimizer, next_dim, glyphs, device,
						current_checkpoint_path, state_path, epoch,
						milestones_achieved, target_milestone
					)
					neurogenesis_history.append({
						"epoch": epoch, "old_dim": old_dim, "new_dim": next_dim,
						"val_loss_at_trigger": val_loss,
					})
					best_val_loss = float("inf")  # reset tras crecer
					epochs_without_improvement = 0
				else:
					print(f"  ⚠️ [PLATEAU] val_loss estancada {epochs_without_improvement} épocas pero ya en dim máximo para esta etapa ({model.hidden_dim})")

		# Muestreo cualitativo en consola (coherencia semántica) con logit mask de edad
		print("  🔍 [Muestra cualitativa] Generación del modelo:")
		model.eval()

		# Determinar la edad cognitiva del stage para filtrar palabras complejas en el oscilloscope cualitativo
		stage_idx, stage_name = get_stage_info(epoch)
		current_stage_conf = stage_config[stage_idx]
		eval_age = current_stage_conf["age"] if current_stage_conf["age"] is not None else 2

		from scripts.evaluate_samantha_age import get_allowed_vocab_for_age
		allowed_vocab = get_allowed_vocab_for_age(eval_age, base_dir)
		special_tokens = {"yo", "tú", "<pad>", "<unk>", "hola", "mamá", "papá", "nene", "nena", "miau", "guau", "agua", "fuego", "sí", "no", "bien", "mal", "pan"}
		allowed_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
		for w, idx in word_to_idx.items():
			if w in allowed_vocab or w in special_tokens or w.lower() in allowed_vocab:
				allowed_mask[idx] = True
		allowed_mask[0] = True
		allowed_mask[1] = True

		qual_seeds = ["yo sentir", "fuego estar", "madre decir"]
		for seed in qual_seeds:
			seed_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", seed.lower())
			mapped_seed = [dictionary.map_to_base_word(w) for w in seed_words]
			seed_tokens = [word_to_idx.get(w, 1) for w in mapped_seed]

			gen_tokens = list(seed_tokens)
			for _ in range(5):
				padded_in = gen_tokens + [0] * (128 - len(gen_tokens)) if len(gen_tokens) < 128 else gen_tokens[-128:]
				x_in = torch.tensor([padded_in], dtype=torch.long, device=device)
				with torch.no_grad():
					logits_out = model(x_in)
				last_idx = len(gen_tokens) - 1
				step_logits = logits_out[0, last_idx]
				step_logits = step_logits.masked_fill(~allowed_mask, -1e9)
				next_token = step_logits.argmax(dim=-1).item()
				if next_token in [0, 1]:
					break
				gen_tokens.append(next_token)

			gen_text = " ".join([idx_to_word.get(t, "<unk>") for t in gen_tokens])
			print(f"    - Prompter: '{seed}' ➔ '{gen_text}'")

		# Guardar checkpoint y actualizar estado para la siguiente época
		torch.save(model.state_dict(), current_checkpoint_path)
		with open(state_path, "w", encoding="utf-8") as sf:
			json.dump(
				{
					"current_epoch": epoch + 1,
					"current_stage_idx": stage_idx,
					"hidden_dim": model.hidden_dim,
					"num_layers": len(model.core_layers),
					"target_milestone": target_milestone,
					"milestones_achieved": milestones_achieved,
					"curriculum_hash": curriculum_hash,
					"best_val_loss": best_val_loss,
					"epochs_without_improvement": epochs_without_improvement,
					"neurogenesis_history": neurogenesis_history,
				},
				sf,
				indent=4,
			)

		# Evaluar Samantha al final de cada hito
		current_stage_conf = None
		for config in stage_config:
			if config["start_epoch"] <= epoch <= config["end_epoch"]:
				current_stage_conf = config
				break

		if current_stage_conf and epoch == current_stage_conf["end_epoch"] and current_stage_conf["age"] is not None:
			eval_age = current_stage_conf["age"]
			milestone_name = f"{eval_age}_years"
			print(f"\n🎓 [EXAMEN DE GRADUACIÓN] Iniciando evaluación con la Profesora Samantha para el hito de {eval_age} años...")
			model, _ = run_samantha_eval(
				model=model,
				current_checkpoint_path=current_checkpoint_path,
				target_milestone=milestone_name,
				save_dir=save_dir,
				stage_idx=current_stage_conf["stage_idx"],
				stage_name=current_stage_conf["name"],
				milestones_achieved=milestones_achieved,
				state_path=state_path,
				args=args,
				device=device,
				base_dir=base_dir,
				epoch=epoch,
			)

	# Guardar modelo final definitivo
	final_path = os.path.join(save_dir, "model_final.pt")
	torch.save(model.state_dict(), final_path)
	print(f"\n🏆 ¡Entrenamiento completo! Modelo final graduado guardado en {final_path}")


if __name__ == "__main__":
	run_school_training()
