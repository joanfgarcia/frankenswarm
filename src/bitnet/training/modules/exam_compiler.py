import os
import re

from src.bitnet.training.modules.tokenization import generate_question_variations


def compile_exam_sequences_for_age(age: int, exams_data: dict, word_to_idx: dict, dictionary) -> list[list[int]]:
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
		for _subject, qa_pairs in exams_section.items():
			for qa in qa_pairs:
				raw_q = qa["question"]
				raw_a = qa["answer"]

				# Extraer contenido de la pregunta
				q_match = re.match(r"^(yo|tú|me|you)\s*:\s*(.*)$", raw_q, re.IGNORECASE)
				q_content = q_match.group(2) if q_match else raw_q

				for var in generate_question_variations(q_content):
					q_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", var.lower())
					mapped_q = [dictionary.map_to_base_word(w) for w in q_words]
					mapped_a = dictionary.map_to_base_word(raw_a)

					q_tokens = [word_to_idx.get(w, 1) for w in mapped_q]
					a_token = word_to_idx.get(mapped_a, 1)

					dialogue_triggers = {"hello", "how are you", "who are you", "what is your name", "where are you from", "what is your home", "do you like books"}
					if q_content.lower().strip() in dialogue_triggers:
						tokens = [word_to_idx.get("you", 1)] + q_tokens + [word_to_idx.get("me", 1)] + [a_token]
					else:
						tokens = q_tokens + [a_token]
					exam_sequences.append(tokens)

	# 2. Obtener preguntas específicas de evaluate_samantha_age.py para la edad
	from scripts.evaluate_samantha_age import AGE_QUESTIONS
	age_questions = AGE_QUESTIONS.get(age, [])
	for qa in age_questions:
		raw_q = qa["question"]
		raw_a = qa["expected"]

		q_match = re.match(r"^(yo|tú|me|you)\s*:\s*(.*)$", raw_q, re.IGNORECASE)
		q_content = q_match.group(2) if q_match else raw_q

		for var in generate_question_variations(q_content):
			q_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", var.lower())
			mapped_q = [dictionary.map_to_base_word(w) for w in q_words]
			mapped_a = dictionary.map_to_base_word(raw_a)

			q_tokens = [word_to_idx.get(w, 1) for w in mapped_q]
			a_token = word_to_idx.get(mapped_a, 1)

			dialogue_triggers = {"hello", "how are you", "who are you", "what is your name", "where are you from", "what is the bunker", "do you like borges"}
			if q_content.lower().strip() in dialogue_triggers:
				tokens = [word_to_idx.get("you", 1)] + q_tokens + [word_to_idx.get("me", 1)] + [a_token]
			else:
				tokens = q_tokens + [a_token]
			exam_sequences.append(tokens)

	return exam_sequences


def exam_answer_words_for_age(age: int, exams_data: dict, dictionary, base_dir_arg: str = ".") -> set[str]:
	"""Palabras de RESPUESTA de la batería de examen de una edad (exams + AGE_QUESTIONS),
	ya mapeadas por el diccionario. Se inyectan en el gateo de su etapa: todo target
	de examen debe ser producible en la etapa que lo examina (BIT-003 S5, DL-009)."""
	answers: set[str] = set()
	key = None
	if age in [2, 3, 4]:
		key = "preschool"
	elif age in [5, 6]:
		key = "primary"
	elif age in [7, 8]:
		key = "secondary"
	if key:
		for qas in exams_data.get(key, {}).values():
			for qa in qas:
				answers.add(dictionary.map_to_base_word(qa["answer"]))
	from scripts.evaluate_samantha_age import AGE_QUESTIONS
	for qa in AGE_QUESTIONS.get(age, []):
		answers.add(dictionary.map_to_base_word(qa["expected"]))
	# DL-013: las respuestas del BANK de la etapa (y de las etapas acumuladas
	# que el examen muestrea) también son producibles en su etapa
	stage_of_age = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
	for st in range(1, stage_of_age.get(age, 1) + 1):
		bank = load_stage_bank(st, base_dir_arg)
		for qa in bank:
			answers.add(dictionary.map_to_base_word(qa["a"]))
	return answers


# ═══════════════════════════════════════════════════════════════════
# BANK de examen por etapa (DL-013): el corpus de preguntas curricular
# ═══════════════════════════════════════════════════════════════════

def load_stage_bank(stage_idx: int, base_dir: str) -> list[dict]:
	"""Carga el bank de la etapa (configs/exam_banks/stage_{i}.json).
	Devuelve [{"q": ..., "a": ...}] o [] si no existe."""
	import json as _json
	path = os.path.join(base_dir, "configs", "exam_banks", f"stage_{stage_idx}.json")
	if not os.path.exists(path):
		return []
	return _json.load(open(path, encoding="utf-8")).get("questions", [])


def compile_bank_sequences(stage_idx: int, base_dir: str, word_to_idx: dict, dictionary) -> list[list[int]]:
	"""Compila las preguntas del bank de la etapa con el MISMO pipeline que
	compile_exam_sequences_for_age (variaciones + mapeo a base + wrapper)."""
	import re as _re
	bank = load_stage_bank(stage_idx, base_dir)
	sequences = []
	for qa in bank:
		raw_q, raw_a = qa["q"], qa["a"]
		q_match = _re.match(r"^(yo|tú|me|you)\s*:\s*(.*)$", raw_q, _re.IGNORECASE)
		q_content = q_match.group(2) if q_match else raw_q
		for var in generate_question_variations(q_content):
			q_words = _re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", var.lower())
			mapped_q = [dictionary.map_to_base_word(w) for w in q_words]
			mapped_a = dictionary.map_to_base_word(raw_a)
			q_tokens = [word_to_idx.get(w, 1) for w in mapped_q]
			a_token = word_to_idx.get(mapped_a, 1)
			dialogue_triggers = {"hello", "how are you", "who are you", "what is your name",
				"where are you from", "what is your home", "do you like books"}
			if q_content.lower().strip() in dialogue_triggers:
				sequences.append([word_to_idx.get("you", 1)] + q_tokens + [word_to_idx.get("me", 1)] + [a_token])
			else:
				sequences.append(q_tokens + [a_token])
	return sequences


def sample_milestone_exam(stage_idx: int, base_dir: str, n: int = 80,
						  weights: tuple = (5, 15, 20, 50), seed: int = 42) -> list[dict]:
	"""Muestrea un examen de hito del bank ACUMULADO con peso por recencia.

	weights: (etapa k-3, k-2, k-1, k) — la etapa más reciente con el mayor
	peso (50% por defecto). El muestreo es determinista dado `seed`: 10
	seeds distintas = 10 formas del examen (media±σ, DL-013)."""
	import random as _random
	acc = []
	for k in range(max(0, stage_idx - 3), stage_idx + 1):
		bank = load_stage_bank(k, base_dir)
		w = weights[k - max(0, stage_idx - 3)] if 0 <= k - max(0, stage_idx - 3) < len(weights) else 0
		acc += [(qa, w) for qa in bank for _ in range(w)]
	rng = _random.Random(seed)
	rng.shuffle(acc)
	exam, seen = [], set()
	for qa, _w in acc:
		key = (qa["q"], qa["a"])
		if key in seen:
			continue
		seen.add(key)
		exam.append(qa)
		if len(exam) >= n:
			break
	return exam
