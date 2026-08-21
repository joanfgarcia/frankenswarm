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


def exam_answer_words_for_age(age: int, exams_data: dict, dictionary) -> set[str]:
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
	return answers
