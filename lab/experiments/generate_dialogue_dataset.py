import json
import os
import re
import subprocess
import sys
import time

# Añadir el path de sharing para poder importar red_pill
sys.path.append('/home/joan/Documents/IA/sharing/src')
sys.path.append('/home/joan/Documents/IA/frankenswarm')

from src.bitnet.vocab.dictionary_tool import SovereignDictionary


def run_dataset_generation():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--part", type=str, default="")
	parser.add_argument("--target", type=int, default=1000)
	args = parser.parse_args()
	
	part_suffix = f"_{args.part}" if args.part else ""
	
	print(f"═══ 🧒 Generador de Diálogos Infantiles (EXP_072{part_suffix}) ═══")
	
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	output_path = os.path.join(base_dir, "configs", f"tiny_dialogues{part_suffix}.json")
	cache_path = os.path.join(base_dir, "configs", f"word_mappings_cache{part_suffix}.json")
	
	dictionary = SovereignDictionary(expanded_glyphs_path)
	set(dictionary.base_vocab)
	
	# Cargar caché de mapeo de palabras
	word_cache = {}
	for w in dictionary.base_vocab:
		word_cache[w] = w
		
	if os.path.exists(cache_path):
		try:
			with open(cache_path, encoding="utf-8") as f:
				loaded_cache = json.load(f)
				word_cache.update(loaded_cache)
			print(f"Loaded {len(loaded_cache)} mappings from cache.")
		except Exception as e:
			print(f"Error loading cache: {e}")
			
	# Cargar diálogos ya generados si existen para reanudación
	dialogues = []
	if os.path.exists(output_path):
		try:
			with open(output_path, encoding="utf-8") as f:
				dialogues = json.load(f)
			print(f"Resuming generation. Loaded {len(dialogues)} existing dialogues.")
		except Exception as e:
			print(f"Error loading existing dialogues: {e}")
			
	target_dialogues = args.target
	sharing_venv_python = "/home/joan/Documents/IA/sharing/.venv/bin/python"
	sharing_src = "/home/joan/Documents/IA/sharing/src"
	
	temas = [
		"juegos, juguetes y el parque de juegos",
		"animales domésticos, el perro, el gato, pájaros",
		"comida, postres, frutas como manzana y plátano",
		"la mamá, el papá, hermanos y la familia en casa",
		"la escuela, dibujar con lápiz, colores y libros",
		"el sol, la lluvia, el viento y salir a correr",
		"emociones simples como estar contento, triste o asustado por un lobo",
		"objetos sencillos como pelota, muñeco, coche de juguete"
	]
	
	batch_idx = 0
	consecutive_errors = 0
	
	while len(dialogues) < target_dialogues:
		tema = temas[batch_idx % len(temas)]
		print(f"\n[Progreso: {len(dialogues)}/{target_dialogues}] Solicitando lote de diálogos sobre: {tema}...")
		
		prompt = (
			f"Escribe exactamente 50 diálogos independientes y variados entre dos niños pequeños hablando en español sobre: {tema}. "
			"Cada diálogo debe constar de 3 o 4 turnos cortos y sencillos. "
			"Formato estricto:\n"
			"1.\n"
			"yo: <frase>\n"
			"tú: <frase>\n"
			"yo: <frase>\n"
			"tú: <frase>\n"
			"---\n"
			"2.\n"
			"yo: <frase>\n"
			"tú: <frase>\n"
			"yo: <frase>\n"
			"---\n\n"
			"Ejemplos de variedad:\n"
			"Ejemplo A:\n"
			"yo: ¿tienes la pelota?\n"
			"tú: sí, está en la bolsa.\n"
			"yo: vamos a tirar.\n"
			"---\n"
			"Ejemplo B:\n"
			"yo: corre, el lobo viene.\n"
			"tú: tengo miedo.\n"
			"yo: dame la mano.\n"
			"---\n\n"
			"Escribe los 50 diálogos independientes ahora. Sé muy creativo y variado (no uses las mismas frases que en los ejemplos)."
		)
		
		system_prompt = (
			"Eres una educadora infantil experta. Tu tarea es generar ejemplos de diálogos cotidianos de niños "
			"para entrenar un modelo lingüístico simple. Devuelve solo los diálogos con el formato solicitado."
		)
		
		# Llamar a Samantha vía subprocess para aislar dependencias
		cmd_code = f"""
import sys
sys.path.append('{sharing_src}')
from red_pill.inference import samantha_on_demand
res = samantha_on_demand.invoke({repr(prompt)}, system_prompt={repr(system_prompt)}, max_tokens=2048, temperature=0.85)
print(res)
"""
		t0 = time.time()
		try:
			result = subprocess.run(
				[sharing_venv_python, "-c", cmd_code],
				capture_output=True,
				text=True,
				timeout=120
			)
			elapsed = time.time() - t0
			
			if result.returncode == 0:
				consecutive_errors = 0
				raw_output = result.stdout.strip()
				raw_dialogues = raw_output.split("---")
				
				batch_added = 0
				new_mappings_added = 0
				
				for raw_d in raw_dialogues:
					raw_d = raw_d.strip()
					if not raw_d:
						continue
					
					turns = []
					lines = raw_d.split("\n")
					for line in lines:
						line = line.strip()
						if not line:
							continue
						
						# Buscar formato yo: o tú: (puede estar precedido por números o guiones)
						match = re.search(r'\b(yo|tú|t&uacute;)\s*:\s*(.*)$', line, re.IGNORECASE)
						if match:
							speaker = match.group(1).lower()
							if speaker == "t&uacute;":
								speaker = "tú"
							content = match.group(2).strip()
							
							# Limpiar y mapear palabras
							words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', content.lower())
							if not words:
								continue
								
							mapped_words = []
							for w in words:
								if w not in word_cache:
									mapped = dictionary.map_to_base_word(w)
									word_cache[w] = mapped
									new_mappings_added += 1
								else:
									mapped = word_cache[w]
								mapped_words.append(mapped)
								
							if mapped_words:
								turns.append(f"{speaker}: " + " ".join(mapped_words))
								
					# Solo guardar si tiene al menos 2 turnos y no excede longitud
					if len(turns) >= 2 and len(turns) <= 4:
						dialogues.append(turns)
						batch_added += 1
						
				print(f"  Batch complete in {elapsed:.1f}s. Added {batch_added} valid dialogues. Cache size: {len(word_cache)}")
				
				# Guardar incrementalmente cada lote
				with open(output_path, "w", encoding="utf-8") as f:
					json.dump(dialogues, f, indent=4, ensure_ascii=False)
					
				# Guardar la caché de palabras
				if new_mappings_added > 0:
					# Filtrar la caché para guardar solo los mapeos dinámicos
					dynamic_mappings = {k: v for k, v in word_cache.items() if k != v}
					with open(cache_path, "w", encoding="utf-8") as f:
						json.dump(dynamic_mappings, f, indent=4, ensure_ascii=False)
						
			else:
				consecutive_errors += 1
				print(f"  Error invoking LLM. Stderr: {result.stderr}")
				if consecutive_errors >= 5:
					print("Too many consecutive errors. Exiting.")
					sys.exit(1)
				time.sleep(2)
		except Exception as e:
			consecutive_errors += 1
			print(f"  Exception in batch: {e}")
			if consecutive_errors >= 5:
				print("Too many consecutive exceptions. Exiting.")
				sys.exit(1)
			time.sleep(2)
			
		batch_idx += 1
		
	print(f"✨ Success! Total dialogues generated: {len(dialogues)}")
	print(f"Saved to {output_path}")

if __name__ == "__main__":
	run_dataset_generation()
