import csv
import json
import os
import re
import time

import requests

from src.bitnet.dictionary_tool import SovereignDictionary


def clean_gutenberg_text(text):
	start_pattern = re.compile(r"\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
	end_pattern = re.compile(r"\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
	
	start_match = start_pattern.search(text)
	end_match = end_pattern.search(text)
	
	start_idx = start_match.end() if start_match else 0
	end_idx = end_match.start() if end_match else len(text)
	
	cleaned = text[start_idx:end_idx]
	cleaned = re.sub(r"Project Gutenberg's.*?\n", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"EBook of.*?\n", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"\r\n", "\n", cleaned)
	cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
	return cleaned.strip()

def split_into_sentences(text):
	# Split by periods, question marks, or exclamation marks
	sentences = re.split(r'[.!?\n]+', text)
	cleaned = []
	for s in sentences:
		s = s.strip()
		# Keep only sentences with at least 3 words and less than 15 words
		words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', s)
		if 3 <= len(words) <= 15:
			cleaned.append(s)
	return cleaned

def run_download_curriculum():
	print("🚀 Iniciando descarga e integración de textos reales de Project Gutenberg...")
	
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	output_curriculum_path = os.path.join(base_dir, "configs", "school_curriculum.json")
	
	dictionary = SovereignDictionary(expanded_glyphs_path)
	
	# Mapeo de IDs de libros de Gutenberg para cada grado
	curriculum_books = {
		"preschool": [
			{"id": 28669, "name": "Cuentos de mi tía Panchita (Infantil)"},
			{"id": 52839, "name": "Fábulas de Esopo (Infantil)"}
		],
		"primary": [
			{"id": 44342, "name": "Nociones de historia de España (Primaria)"},
			{"id": 31548, "name": "Aritmética práctica (Primaria)"}
		],
		"secondary": [
			{"id": 37172, "name": "Nociones de lógica (Secundaria)"},
			{"id": 31550, "name": "Álgebra elemental (Secundaria)"}
		]
	}
	
	# Cargar currículo existente generado por Samantha si existe para combinarlo
	if os.path.exists(output_curriculum_path):
		with open(output_curriculum_path, encoding="utf-8") as f:
			school_dataset = json.load(f)
		print("📖 Cargado currículo sintáctico existente para fusionarlo.")
	else:
		school_dataset = {"preschool": [], "primary": [], "secondary": []}
		
	# Descargar y procesar cada libro
	for grade, books in curriculum_books.items():
		print(f"\n🎒 [Grado: {grade.upper()}] Descargando textos escolares reales...")
		
		for book in books:
			g_id = book["id"]
			name = book["name"]
			url = f"https://www.gutenberg.org/cache/epub/{g_id}/pg{g_id}.txt"
			print(f"  📖 Descargando '{name}' (ID {g_id})...")
			
			try:
				r = requests.get(url, timeout=30)
				if r.status_code != 200:
					# Fallback
					fallback_url = f"https://www.gutenberg.org/files/{g_id}/{g_id}-0.txt"
					r = requests.get(fallback_url, timeout=30)
					if r.status_code != 200:
						print(f"    ❌ Falló la descarga de ID {g_id}.")
						continue
				
				raw_text = clean_gutenberg_text(r.text)
				raw_sentences = split_into_sentences(raw_text)
				print(f"    ✓ Descargado. Encontradas {len(raw_sentences)} frases candidatas.")
				
				# Procesar un subconjunto de frases (máximo 150 frases por libro para mantener equilibrio)
				selected_sentences = raw_sentences[:150]
				processed_count = 0
				
				for sentence in selected_sentences:
					words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', sentence.lower())
					if not words:
						continue
					
					# Mapear palabras al vocabulario base
					mapped_words = [dictionary.map_to_base_word(w) for w in words]
					cleaned_sentence = " ".join(mapped_words)
					
					school_dataset[grade].append(cleaned_sentence)
					processed_count += 1
					
				print(f"    ✓ Procesadas y añadidas {processed_count} frases limpias mapeadas al vocabulario.")
				time.sleep(1) # Respetar el servidor
				
			except Exception as e:
				print(f"    ❌ Error procesando ID {g_id}: {e}")
				
	# ── Integrar el Corpus de Borges local en la etapa de secundaria ──
	borges_corpus_path = os.path.join(base_dir, "storage", "curriculum", "borges", "datasets", "full_corpus.csv")
	if os.path.exists(borges_corpus_path):
		print("\n📚 [Borges] Integrando corpus local de Borges en Secundaria...")
		try:
			count = 0
			with open(borges_corpus_path, encoding="utf-8") as f:
				reader = csv.reader(f)
				next(reader) # skip header
				# El archivo csv tiene texto de Borges
				for row in reader:
					if len(row) > 1:
						text = row[1] # Asumimos la columna de texto
						sentences = split_into_sentences(text)
						for s in sentences[:10]: # Máximo 10 frases por fila para no saturar
							words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', s.lower())
							if words:
								mapped_words = [dictionary.map_to_base_word(w) for w in words]
								cleaned_sentence = " ".join(mapped_words)
								school_dataset["secondary"].append(cleaned_sentence)
								count += 1
							if count >= 200: # Limitar a 200 frases de Borges para equilibrio
								break
					if count >= 200:
						break
			print(f"    ✓ Añadidas {count} frases de Borges a Secundaria.")
		except Exception as e:
			print(f"    ❌ Error cargando Borges: {e}")
	else:
		print("\n⚠️ No se encontró el corpus local de Borges en storage/curriculum/borges/datasets/full_corpus.csv")
		
	# Eliminar duplicados
	for grade in school_dataset:
		unique = []
		seen = set()
		for s in school_dataset[grade]:
			if s not in seen:
				seen.add(s)
				unique.append(s)
		school_dataset[grade] = unique
		print(f"✨ Total frases finales para [Grado {grade.upper()}]: {len(unique)}")
		
	# Guardar currículo combinado
	with open(output_curriculum_path, "w", encoding="utf-8") as f:
		json.dump(school_dataset, f, indent=4, ensure_ascii=False)
	print(f"\n💾 Currículo fusionado (sintáctico + real Gutenberg + Borges) guardado en: {output_curriculum_path}")

if __name__ == "__main__":
	run_download_curriculum()
