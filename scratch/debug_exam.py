import json
import os
import re

from src.bitnet.dictionary_tool import SovereignDictionary

base_dir = "/home/joan/Documents/IA/frankenswarm"
expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
exams_path = os.path.join(base_dir, "configs", "school_exams.json")

dictionary = SovereignDictionary(expanded_glyphs_path)

with open(expanded_glyphs_path, encoding="utf-8") as f:
	vocab_data = json.load(f)
	words = vocab_data["words"]
word_to_idx = {w: i for i, w in enumerate(words)}

with open(exams_path, encoding="utf-8") as f:
	exams_data = json.load(f)

print("=== DEBÚG DE EXAMEN ===")
for stage_name, categories in exams_data.items():
	print(f"\nGrado: {stage_name}")
	for category_name, qa_pairs in categories.items():
		print(f"  Categoría: {category_name}")
		for qa in qa_pairs:
			raw_q = qa["question"]
			raw_a = qa["answer"]
			
			q_match = re.match(r'^(yo|tú)\s*:\s*(.*)$', raw_q, re.IGNORECASE)
			q_content = q_match.group(2) if q_match else raw_q
				
			q_words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', q_content.lower())
			mapped_q = [dictionary.map_to_base_word(w) for w in q_words]
			mapped_a = dictionary.map_to_base_word(raw_a)
			
			# Mostrar mapeos
			print(f"    Q: {raw_q} -> {mapped_q}")
			print(f"    A: {raw_a} -> {mapped_a}")
			
			# Verificar tokens en word_to_idx
			q_tokens = [word_to_idx.get(w, 1) for w in mapped_q]
			a_token = word_to_idx.get(mapped_a, 1)
			print(f"    Tokens Q: {q_tokens}")
			print(f"    Token A: {a_token}")
			
			# Si algún token es <unk> (1), alertar
			if 1 in q_tokens:
				unk_indices = [i for i, t in enumerate(q_tokens) if t == 1]
				unk_words = [mapped_q[i] for i in unk_indices]
				print(f"    ⚠️ ALERTA: Palabra en Q mapea a <unk>: {unk_words}")
			if a_token == 1:
				print(f"    ⚠️ ALERTA: Respuesta mapea a <unk>: {mapped_a}")
