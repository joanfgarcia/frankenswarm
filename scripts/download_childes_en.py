"""Descarga el corpus CHILDES en INGLÉS para el reentrenamiento de v1 (BIT-003).

A diferencia de `download_childes_spa.py`, aquí NO se mapea ninguna palabra por
similitud de embeddings (`map_to_base_word`): la RULE 7 y el paso 2 de BIT-003
prohíben el mapeo OOV ciego (colisión "pajaritos"→"cálmate", colapso
madrid→spain). Las oraciones se guardan tal cual, minúsculas y sin puntuación;
el censo de vocabulario decide después qué palabras entran (censo directo).

Salida: configs/childes_pre_school_en.json (lista de oraciones únicas, 3-20 palabras).
El fichero español (configs/childes_pre_school.json) NO se toca: los instrumentos
del run v1 original lo referencian y está congelado por DL-004.
"""

import json
import os
import re
import urllib.request


def clean_and_extract_words(line: str) -> list[str]:
	# Quitar signos de puntuación comunes
	line = re.sub(r"[?!.,;:\"()\[\]]", "", line)
	words = [w for w in re.findall(r"[a-zA-Z\-']+", line.lower()) if w]
	return words


def run_download_and_mining():
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	configs_dir = os.path.join(base_dir, "configs")
	temp_dir = os.path.join(base_dir, "storage", "temp_childes_en")
	os.makedirs(temp_dir, exist_ok=True)

	# Misma fuente que el script spa, rama de idioma inglés (BIT-003 §1).
	url = "https://raw.githubusercontent.com/evaportelance/multilingual-aoa-prediction/main/Data/model_datasets/eng/all_child_directed_data.txt"
	dest_file = os.path.join(temp_dir, "all_child_directed_data.txt")

	print("📥 Descargando corpus CHILDES-en desde GitHub raw...")
	try:
		urllib.request.urlretrieve(url, dest_file)
		print("✓ Descargado con éxito.")
	except Exception as e:
		print(f"❌ Error al descargar: {e}")
		return

	print("\n⛏️ Extrayendo oraciones (sin mapeo OOV — RULE 7)...")
	sentences = []
	try:
		with open(dest_file, encoding="utf-8", errors="ignore") as f:
			for idx, line in enumerate(f):
				words = clean_and_extract_words(line)
				# Filtrar oraciones con longitud preescolar/escolar (3 a 20 palabras)
				if 3 <= len(words) <= 20:
					sentences.append(" ".join(words))
				if idx > 0 and idx % 50000 == 0:
					print(f"  ...procesadas {idx} líneas...")
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

	output_path = os.path.join(configs_dir, "childes_pre_school_en.json")
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(unique_sentences, f, indent=4, ensure_ascii=False)
	print(f"\n📁 Archivo guardado con éxito: {output_path}")

	import shutil
	shutil.rmtree(temp_dir)
	print("🗑️ Directorios temporales eliminados.")


if __name__ == "__main__":
	run_download_and_mining()
