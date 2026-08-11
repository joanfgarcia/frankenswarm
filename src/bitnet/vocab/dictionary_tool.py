import json
import os
import re
import subprocess

import numpy as np
from fastembed import TextEmbedding


class SovereignDictionary:
	"""
	Diccionario Soberano que utiliza el LLM local Samantha (Mistral 7B) en CPU.
	Si Samantha genera palabras fuera del vocabulario base de 1,000 palabras,
	se proyectan a la palabra más cercana en el vocabulario base usando fastembed.
	"""

	def __init__(self, expanded_glyphs_path: str = None):
		if expanded_glyphs_path is None:
			base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
			expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")

		# Cargar el vocabulario base
		with open(expanded_glyphs_path, encoding="utf-8") as f:
			data = json.load(f)
			self.base_vocab = data["words"]

		self.vocab_set = set(self.base_vocab)
		self.embedding_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
		self.base_embeddings = None  # Se inicializa perezosamente para ahorrar tiempo
		self.sharing_venv_python = "/home/joan/Documents/IA/sharing/.venv/bin/python"
		self.sharing_src = "/home/joan/Documents/IA/sharing/src"
		
		# Ruta del caché persistente en disco
		base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
		self.cache_file = os.path.join(base_dir, "storage", "datasets", "dictionary_mapping_cache.json")
		os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
		self.mapping_cache = {}
		if os.path.exists(self.cache_file):
			try:
				with open(self.cache_file, encoding="utf-8") as f:
					self.mapping_cache = json.load(f)
			except Exception:
				pass

	def _save_cache(self):
		try:
			with open(self.cache_file, "w", encoding="utf-8") as f:
				json.dump(self.mapping_cache, f, ensure_ascii=False)
		except Exception:
			pass

	def _get_base_embeddings(self):
		if self.base_embeddings is None:
			self.base_embeddings = np.array(list(self.embedding_model.embed(self.base_vocab)), dtype=np.float32)
		return self.base_embeddings

	def map_to_base_word(self, word: str) -> str:
		"""Mapea una palabra fuera de vocabulario a la más cercana en el vocabulario base."""
		word = word.lower().strip()
		if word in self.vocab_set:
			return word

		# Comprobar en el caché
		if word in self.mapping_cache:
			return self.mapping_cache[word]

		# Stemming simple para plurals
		if word.endswith("s") and word[:-1] in self.vocab_set:
			self.mapping_cache[word] = word[:-1]
			return word[:-1]
		if word.endswith("es") and word[:-2] in self.vocab_set:
			self.mapping_cache[word] = word[:-2]
			return word[:-2]

		# Calcular embedding de la palabra desconocida
		word_emb = np.array(list(self.embedding_model.embed([word])), dtype=np.float32)[0]

		# Encontrar la palabra base más similar
		base_embs = self._get_base_embeddings()
		# Cosine similarities
		norms_base = np.linalg.norm(base_embs, axis=1)
		norm_word = np.linalg.norm(word_emb)
		similarities = np.dot(base_embs, word_emb) / (norms_base * norm_word + 1e-10)

		best_idx = np.argmax(similarities)
		mapped_word = self.base_vocab[best_idx]
		
		# Guardar en caché y persistir a disco cada 20 palabras nuevas
		self.mapping_cache[word] = mapped_word
		if len(self.mapping_cache) % 20 == 0:
			self._save_cache()
		return mapped_word

	def buscar_en_samantha(self, word: str) -> str:
		"""Invoca a Samantha de forma síncrona usando el intérprete de python de sharing."""
		# Usamos un prompt muy directo para Mistral
		prompt = f'Completa de forma muy corta, simple y directa (máximo 6 palabras): "{word}" es un...'
		system_prompt = "Eres un diccionario extremadamente simple para niños. Define la palabra dada usando palabras muy sencillas como: aparato, cosa, objeto, persona, animal, para, hacer, ver, mover, grande, pequeño. No des rodeos ni explicaciones."

		cmd_code = f"""
import sys
sys.path.append('{self.sharing_src}')
from red_pill.inference import samantha_on_demand
res = samantha_on_demand.invoke({repr(prompt)}, system_prompt={repr(system_prompt)}, max_tokens=30)
print(res)
"""
		try:
			result = subprocess.run(
				[self.sharing_venv_python, "-c", cmd_code],
				capture_output=True,
				text=True,
				timeout=30
			)
			if result.returncode == 0:
				definition = result.stdout.strip()
				if definition:
					return definition
			print(f"[Samantha Error] stdout: {result.stdout} stderr: {result.stderr}")
		except Exception as e:
			print(f"[Samantha Exception] Failed to query: {e}")

		# Fallback si Samantha falla
		return "cosa o objeto"

	def buscar(self, word: str) -> str:
		"""Retorna la definición de la palabra, garantizando que usa SOLO palabras del vocabulario base."""
		raw_definition = self.buscar_en_samantha(word)

		# Limpiar puntuación y separar en palabras
		words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', raw_definition.lower())

		# Filtrar palabras cortas vacías que no aportan o palabras repetidas de la pregunta
		filtered_words = []
		for w in words:
			if w == word.lower():
				continue
			# Mapear al vocabulario base
			mapped = self.map_to_base_word(w)
			filtered_words.append(mapped)

		# Si la definición quedó vacía
		if not filtered_words:
			filtered_words = ["cosa", "o", "objeto"]

		return " ".join(filtered_words)
