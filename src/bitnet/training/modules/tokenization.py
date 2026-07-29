import re


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
		accents = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'ñ'}
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
	words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>'\-]+", text.lower())
	return [word_to_idx.get(w, 1) for w in words]  # 1 is <unk>


def format_and_tokenize_dialogue(dialogue, word_to_idx):
	dialogue_tokens = []
	for turn in dialogue:
		# Support both English (me/you) and Spanish (yo/tú) speaker prefixes
		match = re.match(r"^(yo|tú|me|you)\s*:\s*(.*)$", turn, re.IGNORECASE)
		if match:
			speaker = match.group(1).lower()
			content = match.group(2)
			turn_words = [speaker] + re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>'\-]+", content.lower())
			turn_tokens = [word_to_idx.get(w, 1) for w in turn_words]
			dialogue_tokens.extend(turn_tokens)
	return dialogue_tokens
