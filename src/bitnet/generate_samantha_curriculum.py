import json
import os
import re
import subprocess

from src.bitnet.dictionary_tool import SovereignDictionary

CORE_GRAMMAR_WORDS = [
	# Artículos y pronombres
	"el", "la", "los", "las", "un", "una", "unos", "unas", "que", "lo", "le", "se", "me", "te", "nos",
	"yo", "tú", "él", "ella", "nosotros", "ellos", "ellas", "este", "esta", "ese", "esa", "mío", "tuyo",
	# Preposiciones y conjunciones
	"y", "o", "pero", "porque", "si", "entonces", "aunque", "con", "sin", "de", "para", "por", "en", "sobre", "bajo", "a", "hacia",
	# Verbos auxiliares comunes
	"ser", "estar", "tener", "haber", "hacer", "decir", "ir", "ver", "poder", "comer", "beber", "dormir", "vivir",
	"es", "son", "era", "eran", "sea", "está", "están", "estaba", "estaban", "ha", "han", "hay", "hace", "hacen", "hecho"
]

def run_curriculum_generation():
	print("═══ 🎓 Generador de Currículo Sintáctico de Samantha (EXP_071) ═══")
	
	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	
	# Inicializar diccionario para limpieza y mapeado
	dictionary = SovereignDictionary(expanded_glyphs_path)
	base_vocab = dictionary.base_vocab
	
	# Extraer palabras de contenido (excluyendo las gramaticales básicas)
	grammar_set = set(CORE_GRAMMAR_WORDS)
	[w for w in base_vocab if w not in grammar_set and w not in ["<unk>", "<pad>"]]
	
	# Definir temas variados para los 6 lotes
	topics = [
		"los animales, las plantas y la naturaleza salvaje",
		"la casa, la familia, la escuela y los objetos cotidianos",
		"acciones comunes, juegos, deportes y el movimiento",
		"emociones, sentimientos, colores y descripciones de cosas",
		"la comida, la cocina, el tiempo atmosférico y las estaciones del año",
		"preguntas simples, saludos y conversaciones cotidianas"
	]
	num_batches = len(topics)
	
	sharing_venv_python = "/home/joan/Documents/IA/sharing/.venv/bin/python"
	sharing_src = "/home/joan/Documents/IA/sharing/src"
	
	all_sentences = []
	
	for idx, topic in enumerate(topics):
		print(f"\n[Lote {idx+1}/{num_batches}] Generando frases sobre: {topic}...")
		
		prompt = (
			f"Genera exactamente 80 frases cortas, simples y gramaticalmente correctas en español (de entre 3 y 6 palabras) "
			f"que traten sobre: {topic}. Usa un lenguaje muy sencillo, natural y variado. "
			f"Devuelve solo las frases, una por línea, sin números, viñetas ni rodeos."
		)
		
		system_prompt = (
			"Eres la Profesora Samantha. Tu tarea es enseñar a un niño de 8 años a hablar español. "
			"Devuelve única y exclusivamente las frases generadas, una por línea."
		)
		
		cmd_code = f"""
import logging
logging.basicConfig(level=logging.INFO)
import sys
sys.path.append('{sharing_src}')
from red_pill.inference import samantha_on_demand
res = samantha_on_demand.invoke({repr(prompt)}, system_prompt={repr(system_prompt)}, max_tokens=1500, temperature=0.7)
print(res)
"""
		try:
			result = subprocess.run(
				[sharing_venv_python, "-c", cmd_code],
				capture_output=True,
				text=True,
				timeout=60
			)
			if result.returncode == 0:
				if result.stderr.strip():
					print(f"  [Lote {idx+1} Stderr]: {result.stderr.strip()}")
				lines = result.stdout.strip().split("\n")
				batch_sentences = []
				
				for line in lines:
					line = line.strip()
					if not line or line.startswith("Aquí") or line.startswith("Lote"):
						continue
					
					# Limpiar y mapear palabras mediante el validador
					words_in_line = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', line.lower())
					if not words_in_line:
						continue
						
					mapped_words = [dictionary.map_to_base_word(w) for w in words_in_line]
					cleaned_sentence = " ".join(mapped_words)
					
					# Validar longitud mínima y máxima
					if 3 <= len(mapped_words) <= 8:
						batch_sentences.append(cleaned_sentence)
						
				print(f"  [Lote {idx+1}] Samantha generó {len(lines)} líneas. Validadas y mapeadas: {len(batch_sentences)} frases.")
				all_sentences.extend(batch_sentences)
			else:
				print(f"  [Lote {idx+1}] Error en ejecución de Samantha. Stderr: {result.stderr}")
		except Exception as e:
			print(f"  [Lote {idx+1}] Excepción en ejecución: {e}")
			
	# Eliminar duplicados manteniendo orden
	unique_sentences = []
	seen = set()
	for s in all_sentences:
		if s not in seen:
			seen.add(s)
			unique_sentences.append(s)
			
	print(f"\n✨ Generación completada. Total de frases únicas válidas: {len(unique_sentences)}")
	
	# Guardar en configs/samantha_curriculum.json
	output_dir = os.path.join(base_dir, "configs")
	output_path = os.path.join(output_dir, "samantha_curriculum.json")
	
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(unique_sentences, f, indent=4, ensure_ascii=False)
		
	print(f"📁 Currículo guardado en: {output_path}")

if __name__ == "__main__":
	# Importar numpy localmente para evitar dependencias pesadas si no se requiere
	run_curriculum_generation()
