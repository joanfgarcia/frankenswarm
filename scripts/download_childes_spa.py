import json
import os
import re
import urllib.request

from src.bitnet.vocab.dictionary_tool import SovereignDictionary


def clean_and_extract_words(line: str) -> list[str]:
	# Quitar signos de puntuación comunes e interrogaciones/exclamaciones
	line = re.sub(r'[¿?¡!.,;:\"()]', '', line)
	# Extraer palabras en minúscula
	words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', line.lower())
	return words

def run_download_and_mining():
	print("═══ 🍼 Descargador y Limpiador del Corpus CHILDES Español (Pre-cleaned) ═══")
	
	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	configs_dir = os.path.join(base_dir, "configs")
	temp_dir = os.path.join(base_dir, "storage", "temp_childes")
	os.makedirs(temp_dir, exist_ok=True)
	
	# Diccionario base
	expanded_glyphs_path = os.path.join(configs_dir, "expanded_glyphs.json")
	dictionary = SovereignDictionary(expanded_glyphs_path)
	
	url = "https://raw.githubusercontent.com/evaportelance/multilingual-aoa-prediction/main/Data/model_datasets/spa/all_child_directed_data.txt"
	dest_file = os.path.join(temp_dir, "all_child_directed_data.txt")
	
	print("📥 Descargando corpus desde GitHub raw...")
	try:
		urllib.request.urlretrieve(url, dest_file)
		print("✓ Descargado con éxito.")
	except Exception as e:
		print(f"❌ Error al descargar: {e}")
		return
	
	# Procesar con caché local para acelerar el mapeo semántico
	print("\n⛏️ Minando emisiones y mapeando al diccionario base...")
	sentences = []
	mapping_cache = {}
	
	def get_mapped_word(w: str) -> str:
		if w not in mapping_cache:
			mapping_cache[w] = dictionary.map_to_base_word(w)
		return mapping_cache[w]

	try:
		with open(dest_file, encoding="utf-8", errors="ignore") as f:
			for idx, line in enumerate(f):
				words = clean_and_extract_words(line)
				if words:
					mapped_words = [get_mapped_word(w) for w in words]
					# Filtrar oraciones con longitud preescolar/escolar (3 a 20 palabras)
					if 3 <= len(mapped_words) <= 20:
						sentences.append(" ".join(mapped_words))
				if idx > 0 and idx % 5000 == 0:
					print(f"  ...procesadas {idx} líneas... (Caché: {len(mapping_cache)} palabras)")
	except Exception as e:
		print(f"❌ Error al procesar archivo: {e}")
		return
	
	print(f"✓ Total oraciones procesadas: {len(sentences)}")
	
	# Eliminar duplicados manteniendo orden
	unique_sentences = []
	seen = set()
	for s in sentences:
		if s not in seen:
			seen.add(s)
			unique_sentences.append(s)
	
	print(f"✓ Filtradas {len(unique_sentences)} oraciones únicas.")
	
	# Guardar en configs/childes_pre_school.json
	output_path = os.path.join(configs_dir, "childes_pre_school.json")
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(unique_sentences, f, indent=4, ensure_ascii=False)
	print(f"\n📁 Archivo guardado con éxito: {output_path}")
	
	# Limpieza de temporales
	import shutil
	shutil.rmtree(temp_dir)
	print("🗑️ Directorios temporales eliminados.")

if __name__ == "__main__":
	run_download_and_mining()
