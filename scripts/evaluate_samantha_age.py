import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.vocab.dictionary_tool import SovereignDictionary

# Batería de preguntas por edad cognitiva/milestone (2 a 8 años).
# Instrumento v2 (DL-009, 21-ago-2026): 10 preguntas por edad, SOLO palabras del
# vocabulario (sin nombres propios de lore: el control v1-inglés mide adquisición
# de lenguaje, no memorización de tokens sin presencia en corpus). Validado por
# scripts/validate_exam_vocab.py.
AGE_QUESTIONS = {
	2: [
		{"question": "you: hello", "expected": "hello"},
		{"question": "you: cat", "expected": "meow"},
		{"question": "you: water", "expected": "water"},
		{"question": "you: fire", "expected": "bad"},
		{"question": "you: mom", "expected": "dad"},
		{"question": "you: dog", "expected": "bark"},
		{"question": "you: the baby wants", "expected": "milk"},
		{"question": "you: night. time to", "expected": "sleep"},
		{"question": "you: the cat wants to", "expected": "eat"},
		{"question": "you: the ball is", "expected": "big"},
	],
	3: [
		{"question": "you: what is your name", "expected": "baby"},
		{"question": "you: the dog runs", "expected": "much"},
		{"question": "you: i want", "expected": "bread"},
		{"question": "you: if i touch the fire", "expected": "burns"},
		{"question": "you: where is dad", "expected": "here"},
		{"question": "you: the sun is", "expected": "hot"},
		{"question": "you: at night we", "expected": "sleep"},
		{"question": "you: the cat drinks", "expected": "milk"},
		{"question": "you: one and one are", "expected": "two"},
		{"question": "you: the dog is my", "expected": "friend"},
	],
	4: [
		{"question": "you: hello", "expected": "hello"},
		{"question": "you: how are you", "expected": "fine"},
		{"question": "you: who are you", "expected": "boy"},
		{"question": "you: the sun shines", "expected": "much"},
		{"question": "you: if i touch the fire", "expected": "hurt"},
		{"question": "you: the fish lives in the", "expected": "water"},
		{"question": "you: at night i see the", "expected": "moon"},
		{"question": "you: the bird has", "expected": "wings"},
		{"question": "you: the snow is", "expected": "cold"},
		{"question": "you: we read a", "expected": "book"},
	],
	5: [
		{"question": "you: i count one two", "expected": "three"},
		{"question": "you: one plus one is", "expected": "two"},
		{"question": "you: the bear eats", "expected": "honey"},
		{"question": "you: the bird flies", "expected": "high"},
		{"question": "you: the flowers drink", "expected": "water"},
		{"question": "you: two plus two is", "expected": "four"},
		{"question": "you: the week has seven", "expected": "days"},
		{"question": "you: the fish swims and the bird", "expected": "flies"},
		{"question": "you: four plus one is", "expected": "five"},
		{"question": "you: we sleep in a", "expected": "bed"},
	],
	6: [
		{"question": "you: what is three plus three? it is", "expected": "six"},
		{"question": "you: what is six minus four? it is", "expected": "two"},
		{"question": "you: the water of the river runs towards the", "expected": "sea"},
		{"question": "you: the trees give oxygen and", "expected": "shade"},
		{"question": "you: the heart pumps blood to the", "expected": "body"},
		{"question": "you: what is four plus five? it is", "expected": "nine"},
		{"question": "you: what is ten minus five? it is", "expected": "five"},
		{"question": "you: the moon shines at", "expected": "night"},
		{"question": "you: the roots of the tree are under the", "expected": "ground"},
		{"question": "you: we write words on a", "expected": "page"},
	],
	7: [
		{"question": "you: what is your name", "expected": "bit"},
		{"question": "you: where are you from", "expected": "cave"},
		{"question": "you: the earth moves around the", "expected": "sun"},
		{"question": "you: the maps show rivers and", "expected": "countries"},
		{"question": "you: rain falls from the", "expected": "sky"},
		{"question": "you: what is seven plus three? it is", "expected": "ten"},
		{"question": "you: the heart moves the", "expected": "blood"},
		{"question": "you: a story lives inside a", "expected": "book"},
		{"question": "you: the sun gives light and", "expected": "heat"},
		{"question": "you: winter is the season of", "expected": "snow"},
	],
	8: [
		{"question": "you: if x plus two is five then x is", "expected": "three"},
		{"question": "you: every cause produces an", "expected": "effect"},
		{"question": "you: the long halls of the library are full of", "expected": "mirrors"},
		{"question": "you: a perfect memory keeps the shape of each", "expected": "cloud"},
		{"question": "you: one point can contain the whole", "expected": "universe"},
		{"question": "you: what is your home", "expected": "cave"},
		{"question": "you: do you like books", "expected": "yes"},
		{"question": "you: written words overcome the passage of", "expected": "time"},
		{"question": "you: what is nine minus six? it is", "expected": "three"},
		{"question": "you: the poet sings to the moon in the cold", "expected": "night"},
	],
}


def query_samantha(prompt: str, system_prompt: str, mock: bool = False, n_questions: int | None = None) -> dict:
	"""
	Invoca a Samantha de forma síncrona en la GPU RTX 5070 para calificar.
	Si mock=True, simula la respuesta. n_questions es el tamaño real de la
	batería (instrumento v2 = 10/edad, DL-009): el mock y el veredicto del
	fallback deben cuadrar con él — el 5 fijo de antes suspendía todo examen.
	"""
	if mock:
		print("🎭 [MOCK] Simulando evaluación de Samantha...")
		# Retornar una estructura mock exitosa (una nota por pregunta real)
		n = n_questions or 5
		return {
			"calificaciones": [
				{"pregunta": f"mock_{i + 1}", "respuesta": "mock", "calificacion": 10, "motivo": "Mock OK"}
				for i in range(n)
			],
			"puntuacion_media": 10.0,
			"hito_superado": True,
		}

	sharing_venv_python = "/home/joan/Documents/IA/sharing/.venv/bin/python"
	sharing_src = "/home/joan/Documents/IA/sharing/src"

	cmd_code = f"""
import sys
sys.path.append('{sharing_src}')
from red_pill.inference import samantha_on_demand
import json

prompt = {repr(prompt)}
system_prompt = {repr(system_prompt)}

try:
	res = samantha_on_demand.invoke(prompt, system_prompt=system_prompt, max_tokens=1000, temperature=0.0)
	print(res)
except Exception as e:
	print(f"ERROR_IN_SAMANTHA: {{e}}")
"""
	try:
		result = subprocess.run([sharing_venv_python, "-c", cmd_code], capture_output=True, text=True, timeout=90)
		if result.returncode == 0:
			output = result.stdout.strip()
			if "ERROR_IN_SAMANTHA" in output:
				print(f"❌ Error en ejecución de Samantha: {output}")
				return None

			# Buscar el bloque JSON en el stdout
			match = re.search(r"(\{.*\})", output, re.DOTALL)
			if match:
				json_str = match.group(1)
				# Limpiar escapes de barra invertida antes de guiones bajos en JSON
				json_str_clean = json_str.replace(r"\_", "_")
				try:
					return json.loads(json_str_clean)
				except json.JSONDecodeError as je:
					print(f"⚠️ Error al decodificar JSON devuelto por Samantha ({je}). Intentando reconstrucción via regex...")
					calificaciones = []
					blocks = re.findall(r"\{\s*\"pregunta\".*?\}", json_str_clean, re.DOTALL)
					for block in blocks:
						pregunta_match = re.search(r'"pregunta"\s*:\s*"([^"]*)"', block)
						respuesta_match = re.search(r'"respuesta"\s*:\s*"([^"]*)"', block)
						esperada_match = re.search(r'"esperada"\s*:\s*"([^"]*)"', block)
						calificacion_match = re.search(r'"calificacion"\s*:\s*(\d+)', block)
						motivo_match = re.search(r'"motivo"\s*:\s*"([^"]*)"', block)

						if pregunta_match and respuesta_match and esperada_match and calificacion_match and motivo_match:
							calificaciones.append({
								"pregunta": pregunta_match.group(1),
								"respuesta": respuesta_match.group(1),
								"esperada": esperada_match.group(1),
								"calificacion": int(calificacion_match.group(1)),
								"motivo": motivo_match.group(1)
							})
					if len(calificaciones) > 0:
						avg = sum(c["calificacion"] for c in calificaciones) / len(calificaciones)
						return {
							"calificaciones": calificaciones,
							"puntuacion_media": avg,
							"hito_superado": avg >= 8.0 and len(calificaciones) == (n_questions or len(calificaciones)),
						}
					print(f"❌ Fallo al reconstruir JSON. Contenido: {json_str}")
			else:
				print(f"❌ No se encontró formato JSON en la salida de Samantha: {output}")
		else:
			print(f"[Samantha Subprocess Error] stdout: {result.stdout} stderr: {result.stderr}")
	except Exception as e:
		print(f"[Samantha Exception] Fallo al invocar Samantha: {e}")

	return None


def _exam_words_for_age(age: int, base_dir: str) -> set[str]:
	"""Palabras de la batería de exámenes (school_exams_en.json + AGE_QUESTIONS) hasta la edad dada.

	Las respuestas esperadas de los exámenes (p. ej. 'effect', 'mirrors', 'universe') no aparecen
	en los textos del currículo, así que la máscara de vocabulario por edad las excluía y el alumno
	jamás podía emitirlas aunque las tuviera memorizadas. Se unen aquí para que la generación (y el
	escaneo OOB) las trate como vocabulario legítimo de su edad.

	Además se incluyen las FORMAS BASE del Diccionario Soberano: lo que Bit entrena como respuesta
	no es la cadena cruda del examen (p. ej. 'aleth', 'bunker'), sino su base ('aliya', 'hideout').
	Si solo se destaparan las palabras crudas, los tokens reales seguirían enmascarados y el examen
	seguiría siendo irresoluble por la puerta de generación.
	"""
	words = set()

	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	dictionary = SovereignDictionary(expanded_glyphs_path)

	def _collect(raw: str):
		for w in re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", raw.lower()):
			words.add(w)
			words.add(dictionary.map_to_base_word(w))

	for a, qas in AGE_QUESTIONS.items():
		if a <= age:
			for qa in qas:
				_collect(qa["question"])
				_collect(qa["expected"])

	exams_path = os.path.join(base_dir, "configs", "school_exams_en.json")
	if os.path.exists(exams_path):
		with open(exams_path, encoding="utf-8") as f:
			exams = json.load(f)
		buckets = []
		if age >= 2:
			buckets.append("preschool")
		if age >= 5:
			buckets.append("primary")
		if age >= 7:
			buckets.append("secondary")
		for bucket in buckets:
			for qas in exams.get(bucket, {}).values():
				for qa in qas:
					_collect(qa.get("question", ""))
					_collect(qa.get("answer", ""))

	return words


def get_allowed_vocab_for_age(age: int, base_dir: str) -> set[str]:
	curriculum_path = os.path.join(base_dir, "configs", "school_curriculum_structured_en.json")
	childes_path = os.path.join(base_dir, "configs", "childes_pre_school_en.json")
	nsm_path = os.path.join(base_dir, "configs", "nsm_physics_pre_school.json")

	with open(curriculum_path, encoding="utf-8") as f:
		curriculum_json = json.load(f)
		curriculum_data_raw = curriculum_json.get("curriculum", {})
		curriculum_data = {
			"preschool": [item["text"] for item in curriculum_data_raw.get("preschool", [])],
			"primary": [item["text"] for item in curriculum_data_raw.get("primary", [])],
			"secondary": [item["text"] for item in curriculum_data_raw.get("secondary", [])]
		}

	# Lista de palabras básicas permitidas para evitar falsos positivos
	safe_words = {
		"no", "i", "you", "he", "she", "we", "they", "my", "your", "his", "her", "its", "our", "their",
		"me", "him", "us", "them", "it", "this", "that", "these", "those", "a", "an", "the",
		"to", "of", "in", "for", "on", "with", "without", "about", "and", "or", "but", "if", "because",
		"is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "want", "can",
		"say", "see", "go", "give", "know", "eat", "drink", "meow", "bark", "hurt", "hello", "fine", "good",
		"dad", "mom", "baby", "kid", "kiss", "take", "more", "sleep", "runs", "much", "very"
	}

	preschool_words = set(safe_words)
	for sentence in curriculum_data.get("preschool", []):
		words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", sentence.lower())
		preschool_words.update(words)

	if os.path.exists(childes_path):
		with open(childes_path, encoding="utf-8") as f:
			childes_data = json.load(f)
		for sentence in childes_data:
			words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", sentence.lower())
			preschool_words.update(words)

	if os.path.exists(nsm_path):
		with open(nsm_path, encoding="utf-8") as f:
			nsm_data = json.load(f)
		for sentence in nsm_data:
			words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", sentence.lower())
			preschool_words.update(words)

	if age <= 4:
		return preschool_words | _exam_words_for_age(age, base_dir)

	primary_words = set()
	for sentence in curriculum_data.get("primary", []):
		words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", sentence.lower())
		primary_words.update(words)

	if age <= 6:
		return (preschool_words | primary_words) | _exam_words_for_age(age, base_dir)

	secondary_words = set()
	for sentence in curriculum_data.get("secondary", []):
		words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", sentence.lower())
		secondary_words.update(words)

	return (preschool_words | primary_words | secondary_words) | _exam_words_for_age(age, base_dir)


def run_evaluation(args):

	device = torch.device(args.device)
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")

	if not os.path.exists(args.model_path):
		print(f"❌ Error: No se encontró el modelo en {args.model_path}")
		return None, 0.0

	# 1. Cargar vocabulario y glifos
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))

	# 2. Inicializar Diccionario Soberano
	dictionary = SovereignDictionary(expanded_glyphs_path)

	# 3. Inicializar y cargar modelo (en la CPU o iGPU según args.device).
	# El modo de embedding se detecta por las claves del checkpoint: los brazos
	# estándar DL-006 (use_glyphs=False, tabla one-hot congelada) no tienen
	# glyph_embedding y reconstruirlos como glifos rompería el load_state_dict.
	state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
	ckpt_uses_glyphs = any(k.startswith("glyph_embedding.") for k in state_dict)
	print(f"🧠 Inicializando modelo BitNet en [{device}] ({args.hidden_dim} dim, {args.num_layers} capas, embedding={'glyph' if ckpt_uses_glyphs else 'standard'})...")
	if ckpt_uses_glyphs:
		model = BitNet4LayerModel(
			use_glyphs=True,
			glyph_table=glyphs,
			hidden_dim=args.hidden_dim,
			num_layers=args.num_layers,
			use_pos_embedding=True,
			is_causal=True,
			max_seq_len=128,
			max_resonance_steps=args.resonance_steps_max,
			n_emotions=args.n_emotions,
			emotion_dim=args.emotion_dim,
			emotion_mode=args.emotion_mode,
		).to(device)
	else:
		model = BitNet4LayerModel(
			use_glyphs=False,
			vocab_embeddings=np.eye(len(words), dtype=np.float32),
			hidden_dim=args.hidden_dim,
			num_layers=args.num_layers,
			use_pos_embedding=True,
			is_causal=True,
			max_seq_len=128,
			max_resonance_steps=args.resonance_steps_max,
			n_emotions=args.n_emotions,
			emotion_dim=args.emotion_dim,
			emotion_mode=args.emotion_mode,
		).to(device)

	model.load_state_dict(state_dict)
	model.eval()

	target_age = args.target_age
	questions = AGE_QUESTIONS.get(target_age, [])
	if not questions:
		print(f"❌ Error: No hay preguntas definidas para la edad {target_age}")
		return None, 0.0

	# Máscara de producción del hito. Camino preferente (DL-009): la máscara de
	# gateo REAL del entrenamiento, persistida por el trainer — el instrumento
	# permite exactamente lo que la etapa dejó producir. Fallback legado si no
	# hay fichero (checkpoints antiguos): get_allowed_vocab_for_age.
	stage_allowed_words = None
	allowed_mask = torch.zeros(len(words), dtype=torch.bool, device=device)
	use_trained_gate = False
	if getattr(args, "stage_masks", None) and args.stage_idx is not None and os.path.exists(args.stage_masks):
		with open(args.stage_masks, encoding="utf-8") as f:
			gate_data = json.load(f)
		if gate_data.get("vocab_size") == len(words):
			for idx in gate_data["stage_allowed"][args.stage_idx]:
				allowed_mask[idx] = True
			stage_allowed_words = {words[i] for i in gate_data["stage_allowed"][args.stage_idx]}
			use_trained_gate = True
			print(f"🔒 [GATEO] Máscara de entrenamiento E{args.stage_idx}: {int(allowed_mask.sum())} palabras producibles.")
		else:
			print(f"⚠️ [GATEO] vocab_size del fichero de máscaras ({gate_data.get('vocab_size')}) != vocab actual ({len(words)}). Fallback legado.")
	if not use_trained_gate:
		allowed_vocab = get_allowed_vocab_for_age(target_age, base_dir)
		special_tokens = {"me", "you", "<pad>", "<unk>", "hello", "mom", "dad", "baby", "kid", "meow", "bark", "water", "fire", "yes", "no", "fine", "bad", "bread", "good"}
		for w, idx in word_to_idx.items():
			if w in allowed_vocab or w in special_tokens or w.lower() in allowed_vocab:
				allowed_mask[idx] = True
	allowed_mask[0] = True   # <pad> permitido: es la señal de parada de generación
	allowed_mask[1] = False  # <unk> vetado, igual que en entrenamiento (DL-008)

	# ── DL-013: examen muestreado del BANK acumulado (10 formas, media±σ) ──
	if args.exam_banks_stage is not None:
		from src.bitnet.training.modules.exam_compiler import sample_milestone_exam
		form_scores, all_rows = [], []
		for form in range(10):
			sampled = sample_milestone_exam(args.exam_banks_stage, base_dir, n=80, seed=1000 + form)
			hits = 0
			for qa in sampled:
				raw_q, expected = qa["q"], qa["a"]
				dictionary.map_to_base_word(expected)
				q_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", raw_q.lower())
				mapped_q = [dictionary.map_to_base_word(w) for w in q_words]
				dialogue_triggers = {"hello", "how are you", "who are you", "what is your name",
					"where are you from", "what is your home", "do you like books"}
				is_dialogue = raw_q.lower().strip() in dialogue_triggers
				if is_dialogue:
					context = [word_to_idx.get("you", 1)] + [word_to_idx.get(w, 1) for w in mapped_q] + [word_to_idx.get("me", 1)]
				else:
					context = [word_to_idx.get(w, 1) for w in mapped_q]
				gen_tokens = list(context)
				for step_i in range(5):
					padded = gen_tokens[-128:] if len(gen_tokens) > 128 else gen_tokens + [0] * (128 - len(gen_tokens))
					x_in = torch.tensor([padded], dtype=torch.long, device=device)
					with torch.no_grad():
						if args.resonance_steps_max > 0:
							emo = torch.full((1,), args.n_emotions - 1, dtype=torch.long, device=device) if args.n_emotions > 0 else None
							logits, _ = model.forward_resonance(x_in, n_steps=min(args.resonance_eval_steps, args.resonance_steps_max), pos_mode=args.resonance_pos_mode, emotion_ids=emo)
						else:
							logits = model(x_in)
					step_logits = logits[0, len(gen_tokens) - 1].float()
					current_mask = allowed_mask.clone()
					if step_i == 0:
						current_mask[0] = False
						current_mask[1] = False
					step_logits = step_logits.masked_fill(~current_mask, -1e9)
					pred_token = int(step_logits.argmax(dim=-1).item())
					if pred_token in [0, 1]:
						break
					gen_tokens.append(pred_token)
				pred_words = [idx_to_word.get(t, "<unk>") for t in gen_tokens[len(context):]]
				pred_response = " ".join(pred_words).strip() or "<pad>"
				hit = pred_response.strip() == expected.strip()
				hits += hit
				all_rows.append({"form": form, "q": raw_q, "expected": expected, "generated": pred_response, "hit": hit})
			score = hits / max(1, len(sampled)) * 10
			form_scores.append(score)
			print(f"  🏦 [BANK-EXAM] forma {form} (seed {1000 + form}): {hits}/{len(sampled)} = {score:.1f}/10")
		mean_s = float(np.mean(form_scores)); std_s = float(np.std(form_scores))
		hito = mean_s >= 8.0
		print(f"\n══ BANK-EXAM (etapa {args.exam_banks_stage}): media {mean_s:.2f} ± {std_s:.2f} sobre 10 formas → {'✅ APROBADO' if hito else '❌ SUSPENDIDO'}")
		out_path = os.path.join(base_dir, "storage", "checkpoints", "bank_exam_results.json")
		json.dump({"stage": args.exam_banks_stage, "mean": mean_s, "std": std_s, "passed": hito,
			"rows": all_rows}, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
		sys.exit(0 if hito else 2)

	print(f"\n📝 [Evaluación] Haciendo preguntas del hito de {target_age} años al alumno...")
	qa_pairs = []
	for qa in questions:
		raw_q = qa["question"]
		dictionary.map_to_base_word(qa["expected"])

		# Extraer contenido de la pregunta
		q_match = re.match(r"^(yo|tú|me|you)\s*:\s*(.*)$", raw_q, re.IGNORECASE)
		q_content = q_match.group(2) if q_match else raw_q

		# Limpiar y mapear la pregunta
		q_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", q_content.lower())
		mapped_q = [dictionary.map_to_base_word(w) for w in q_words]

		# Tokenizar
		dialogue_triggers = {"hello", "how are you", "who are you", "what is your name", "where are you from", "what is your home", "do you like books"}
		is_dialogue = q_content.lower().strip() in dialogue_triggers

		if is_dialogue:
			context = [word_to_idx.get("you", 1)] + [word_to_idx.get(w, 1) for w in mapped_q] + [word_to_idx.get("me", 1)]
		else:
			context = [word_to_idx.get(w, 1) for w in mapped_q]

		# Autoregressive generation (hasta 5 tokens)
		gen_tokens = list(context)
		for step_i in range(5):
			padded_input = list(gen_tokens)
			padded_input = padded_input + [0] * (128 - len(padded_input)) if len(padded_input) < 128 else padded_input[-128:]
			x_in = torch.tensor([padded_input], dtype=torch.long, device=device)
			with torch.no_grad():
				if args.resonance_steps_max > 0:
					# Brazo resonante: emoción NEUTRAL (último id) — in-distribution
					# para un modelo entrenado con emoción por secuencia.
					emo = torch.full((1,), args.n_emotions - 1, dtype=torch.long, device=device) if args.n_emotions > 0 else None
					logits, _ = model.forward_resonance(
						x_in, n_steps=min(args.resonance_eval_steps, args.resonance_steps_max),
						pos_mode=args.resonance_pos_mode, emotion_ids=emo,
					)
				else:
					logits = model(x_in)
			last_token_idx = len(gen_tokens) - 1
			step_logits = logits[0, last_token_idx]
			
			# Copiar máscara para desactivar pad/unk temporalmente en el primer token
			current_mask = allowed_mask.clone()
			if step_i == 0:
				current_mask[0] = False  # Forbid <pad>
				current_mask[1] = False  # Forbid <unk>
				
			step_logits = step_logits.masked_fill(~current_mask, -1e9)
			pred_token = step_logits.argmax(dim=-1).item()
			if pred_token in [0, 1]:  # Stop on pad or unk (solo aplicable a partir de step_i > 0)
				break
			gen_tokens.append(pred_token)

		# Obtener palabras generadas (excluyendo el contexto inicial)
		pred_words = [idx_to_word.get(t, "<unk>") for t in gen_tokens[len(context):]]
		pred_response = " ".join(pred_words).strip()
		if not pred_response:
			pred_response = "<pad>"

		print(f"   💬 P: '{q_content}' | R: '{pred_response}' (Esperado: '{qa['expected']}')")
		qa_pairs.append({"question": q_content, "answer": pred_response, "expected": qa["expected"]})

	# Liberar el modelo y los tensores de memoria antes de llamar a Samantha
	del model
	if args.device == "cuda":
		torch.cuda.empty_cache()

	# 4. Escaneo de vocabulario fuera de edad (Monitor de Alucinaciones Controladas)
	# Misma fuente que la máscara de generación: gateo real de la etapa si existe.
	allowed_vocab = stage_allowed_words if stage_allowed_words is not None else get_allowed_vocab_for_age(target_age, base_dir)
	oob_words_found = {}
	for qa in qa_pairs:
		ans = qa["answer"]
		ans_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", ans.lower())
		ans_words = [w for w in ans_words if w not in {"pad", "unk"}]
		oob = [w for w in ans_words if w not in allowed_vocab]
		if oob:
			oob_words_found[qa["question"]] = oob

	if oob_words_found:
		print("\n⚠️ [ALERTA DE VOCABULARIO FUERA DE EDAD - DETECTADO]")
		print("Se han detectado tokens que no corresponden al desarrollo cognitivo de la edad actual:")
		for q, words_list in oob_words_found.items():
			print(f"   - En respuesta a '{q}': {words_list}")
		print("Esto puede deberse a ruido estocástico del output head proyectado (15k) o deriva semántica.")
		print("============================================================\n")

	# 5. Formular el prompt evaluador para Samantha (Mistral 7B)
	age_guideline = ""
	if target_age in [2, 3]:
		age_guideline = (
			"\nCRITICAL NOTE: El alumno es un bebé de 2 o 3 años. Su habla es telegráfica y comete errores gramaticales comunes. "
			"NO exijas oraciones completas ni corrección gramatical. Califica basándote únicamente en si la palabra clave o "
			"la intención semántica es la esperada (por ejemplo, ante 'cat' -> 'meow' es excelente; ante 'mom' -> 'dad' es una asociación infantil normal; "
			"ante 'water' -> 'water' o 'drink' es excelente; ante 'fire' -> 'bad' o 'hot' es excelente)."
		)

	system_prompt = (
		"Eres la Profesora Samantha, una experta en psicología infantil y lingüística. "
		"Estás evaluando el habla y desarrollo cognitivo de un niño de 2 a 8 años.\n"
		"Tu tarea es analizar la respuesta del alumno para cada pregunta y determinar si demuestra "
		"comprensión semántica, lógica física básica y coherencia sintáctica para su edad.\n"
		"No exijas una coincidencia de palabras exacta; valora positivamente sinónimos, expresiones semánticamente "
		"equivalentes y respuestas con sentido lógico (por ejemplo, si se espera 'mucho' ante 'el sol brilla', "
		"respuestas como 'alto', 'caliente' o 'luz' son válidas; si se espera 'dolor' ante 'si toco el fuego', "
		"respuestas como 'quema', 'caliente' o 'malo' son válidas).\n"
		f"{age_guideline}\n"
		"Sin embargo, debes penalizar rigurosamente:\n"
		"- Respuestas con lenguaje metafórico abstracto o excesivamente complejo para la edad cognitiva dada.\n"
		f"- Respuestas que correspondan a etapas de desarrollo superiores. Si la edad evaluada es {target_age} años, el niño NO debe responder con palabras de primaria o secundaria (como 'asteroide', 'población', 'cáncer', 'oxígeno', 'espejos', 'infinitos', 'tiempo', 'capital'). Si el alumno usa vocabulario fuera de su rango de edad (como palabras escolares complejas o conceptos de matemáticas avanzadas), la calificación de esa pregunta debe ser castigada a un rango de 0 a 3.\n"
		"IMPORTANTE: la 'respuesta correcta esperada' de cada pregunta está SIEMPRE exenta del castigo por vocabulario fuera de edad: si la respuesta del alumno coincide con la esperada, o es un sinónimo o equivalente semántico claro de ella, NO apliques ese castigo y califícala con la rúbrica normal de comprensión (una coincidencia exacta merece 10; un sinónimo o equivalente razonable, 8-10 según lo bien que responda a la pregunta).\n"
		"- Respuestas que contengan palabras de ruido/alucinación aleatoria que no tengan coherencia sintáctica o gramatical en una frase simple.\n\n"
		"Califica cada respuesta de 0 a 10 y detalla tu motivo en una frase muy corta (máximo 10 palabras).\n"
		"Debes responder ÚNICAMENTE con un objeto JSON válido que siga exactamente este formato:\n"
		"{\n"
		'  "calificaciones": [\n'
		'    {"pregunta": "...", "respuesta": "...", "esperada": "...", "calificacion": 10, "motivo": "..."}\n'
		"  ],\n"
		'  "puntuacion_media": 10.0,\n'
		'  "hito_superado": true\n'
		"}\n"
		f"Nota: La puntuacion_media es el promedio de las calificaciones de las {len(qa_pairs)} preguntas (debes calificar LAS {len(qa_pairs)}, una entrada por pregunta). El hito se considera superado si la puntuación media es igual o superior a 8.0.\n"
		"CRITICAL: Do NOT escape underscores in JSON keys or values (do NOT use \\_). The output must be standard JSON parseable by python json.loads."
	)

	prompt = f"Edad de evaluación cognitiva: {target_age} años.\n\nPreguntas del examen y respuestas dadas por el alumno:\n"
	for idx, qa in enumerate(qa_pairs):
		prompt += f'{idx + 1}. Pregunta: "{qa["question"]}"\n'
		prompt += f'   Respuesta del alumno: "{qa["answer"]}"\n'
		prompt += f'   Respuesta correcta esperada: "{qa["expected"]}"\n\n'

	if oob_words_found:
		prompt += "⚠️ ALERTA DE VOCABULARIO ANÓMALO DETECTADO POR EL SISTEMA:\n"
		prompt += f"El sistema de detección automática de anomalías ha encontrado que las siguientes respuestas contienen palabras que están completamente fuera del rango de desarrollo de {target_age} años:\n"
		for q, words_list in oob_words_found.items():
			prompt += f"   - En respuesta a '{q}': se detectó la palabra/s {words_list}\n"
		prompt += "Por favor, ten en cuenta esta alerta de vocabulario y penaliza severamente el uso de estas palabras anómalas (asignando notas muy bajas, de 0 a 3, en las preguntas correspondientes).\n\n"

	# Hybrid auto-grader: check for exact matches
	calificaciones = []
	needs_llm_grading = False
	
	for qa in qa_pairs:
		ans_clean = qa["answer"].strip().lower()
		exp_clean = dictionary.map_to_base_word(qa["expected"]).strip().lower()
		raw_exp_clean = qa["expected"].strip().lower()
		
		if ans_clean in (exp_clean, raw_exp_clean):
			calificaciones.append({
				"pregunta": qa["question"],
				"respuesta": qa["answer"],
				"esperada": qa["expected"],
				"calificacion": 10,
				"motivo": "Exact match (Auto-graded)"
			})
		else:
			needs_llm_grading = True
			
	if not needs_llm_grading and len(calificaciones) == len(qa_pairs):
		print("⚡ [AUTO-GRADER] All responses match expected answers exactly. Skipping LLM call.")
		result_json = {
			"calificaciones": calificaciones,
			"puntuacion_media": 10.0,
			"hito_superado": True
		}
	else:
		print(f"\n📡 Enviando examen del hito de {target_age} años a la Profesora Samantha para calificar...")
		print("--- DEBUG PROMPT ---")
		print(prompt)
		print("--- DEBUG SYSTEM PROMPT ---")
		print(system_prompt)
		print("---------------------------")
		result_json = query_samantha(prompt, system_prompt, mock=args.test_mock, n_questions=len(qa_pairs))
		
		if result_json and "calificaciones" in result_json:
			# Post-process to ensure exact matches are always 10/10
			for cal in result_json["calificaciones"]:
				p_clean = cal.get("pregunta", "").strip().lower()
				r_clean = cal.get("respuesta", "").strip().lower()
				matching_expected = None
				for qa in qa_pairs:
					q_clean_def = qa["question"].strip().lower()
					if q_clean_def == p_clean or p_clean in q_clean_def or q_clean_def in p_clean:
						matching_expected = qa["expected"]
						break
				if matching_expected:
					exp_clean = dictionary.map_to_base_word(matching_expected).strip().lower()
					raw_exp_clean = matching_expected.strip().lower()
					if r_clean in (exp_clean, raw_exp_clean):
						cal["calificacion"] = 10
						cal["motivo"] = "Exact match (Auto-grade Override)"
						cal["esperada"] = matching_expected
			
			# Recalculate averages
			scores = [c["calificacion"] for c in result_json["calificaciones"]]
			avg = sum(scores) / len(scores) if scores else 0.0
			result_json["puntuacion_media"] = avg
			result_json["hito_superado"] = avg >= 8.0 and len(scores) == len(qa_pairs)

	if not result_json:
		print("❌ Fallo en la evaluación. No se pudo obtener calificación de Samantha.")
		return None, 0.0

	print("\n════════════════════════════════════════════════════════════")
	print(f"📊 REPORT DE EVALUACIÓN COGNITIVA: {target_age} AÑOS")
	print("════════════════════════════════════════════════════════════")
	for idx, cal in enumerate(result_json.get("calificaciones", [])):
		print(f'{idx + 1}. P: "{cal.get("pregunta")}"')
		print(f'   R: "{cal.get("respuesta")}" (Esperaba: "{cal.get("esperada")}")')
		print(f"   ⭐ Nota: {cal.get('calificacion')}/10 | Motivo: {cal.get('motivo')}")
		print("-" * 40)

	avg_score = result_json.get("puntuacion_media", 0.0)
	passed = result_json.get("hito_superado", False)

	status = "✅ APROBADO (Hito Superado)" if passed else "❌ SUSPENDIDO"
	print(f"🏆 PUNTUACIÓN MEDIA: {avg_score:.2f}/10 | ESTADO: {status}")
	print("════════════════════════════════════════════════════════════\n")

	return passed, avg_score


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Samantha Age Grader Evaluator")
	parser.add_argument("--model_path", type=str, required=True, help="Ruta del checkpoint a evaluar")
	parser.add_argument("--hidden_dim", type=int, default=256, help="Dimensión oculta del modelo")
	parser.add_argument("--num_layers", type=int, default=6, help="Capas del modelo")
	parser.add_argument("--target_age", type=int, required=True, choices=[2, 3, 4, 5, 6, 7, 8], help="Edad objetivo a evaluar")
	parser.add_argument("--device", type=str, default="cpu", help="Dispositivo para correr el modelo (default: cpu)")
	parser.add_argument("--test_mock", action="store_true", help="Simular respuestas de Samantha de forma mock")
	parser.add_argument("--exam_banks_stage", type=int, default=None, help="DL-013: examen muestreado del BANK acumulado (10 formas, media±σ, exact-match)")
	parser.add_argument("--stage_masks", type=str, default=None, help="JSON de máscaras de gateo por etapa persistido por el trainer (stage_gate_masks.json)")
	parser.add_argument("--stage_idx", type=int, default=None, help="Índice de etapa (0-7) cuya máscara de gateo usar")
	parser.add_argument("--resonance_steps_max", type=int, default=0, help="Brazo resonante (DL-010): >0 instancia resonance_clock y usa forward_resonance en inferencia")
	parser.add_argument("--resonance_pos_mode", type=str, default="clock", choices=["none", "entry", "clock"], help="Modo de posicionamiento del bucle de resonancia")
	parser.add_argument("--resonance_eval_steps", type=int, default=3, help="n_steps fijo en inferencia resonante")
	parser.add_argument("--n_emotions", type=int, default=0, help="Nº de emociones del checkpoint resonante (dojo 6 + neutral)")
	parser.add_argument("--emotion_dim", type=int, default=16, help="Dimensión del embedding emocional")
	parser.add_argument("--emotion_mode", type=str, default="first_only", choices=["additive", "gated", "first_only"], help="Modo de inyección emocional")

	args = parser.parse_args()
	passed, score = run_evaluation(args)
	# Exit codes (auditoría 1-sep): un crash de infraestructura NO es un
	# suspenso académico — el falso suspenso de 2_years del glyph v4 disparó
	# una neurogénesis sin examen real. Contrato con state_manager:
	#   0 = aprobado · 2 = suspenso CALIFICADO · 3 = error de infraestructura
	#   (sin nota) · cualquier otro (p.ej. 1 por excepción) = infraestructura.
	if passed is None:
		exit(3)
	exit(0 if passed else 2)
