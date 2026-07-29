import os
import re
import time

import requests


def clean_gutenberg_text(text):
	# Remover el encabezado de Project Gutenberg
	start_pattern = re.compile(r"\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
	end_pattern = re.compile(r"\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
	
	start_match = start_pattern.search(text)
	end_match = end_pattern.search(text)
	
	start_idx = start_match.end() if start_match else 0
	end_idx = end_match.start() if end_match else len(text)
	
	cleaned = text[start_idx:end_idx]
	
	# Remover líneas de cabecera duplicadas y licencias que a veces quedan
	cleaned = re.sub(r"Project Gutenberg's.*?\n", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"EBook of.*?\n", "", cleaned, flags=re.IGNORECASE)
	
	# Limpieza general de espacios múltiples
	cleaned = re.sub(r"\r\n", "\n", cleaned)
	cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
	
	return cleaned.strip()

def download_corpus():
	print("🚀 Iniciando descarga del corpus de pre-entrenamiento en Español...")
	
	# Lista de IDs de Project Gutenberg de clásicos españoles (Público dominio y gratuitos)
	gutenberg_ids = [
		2000,   # Don Quijote
		17073,  # La Regenta I
		19574,  # La Regenta II
		17150,  # Fortunata y Jacinta
		16279,  # Marianela
		16938,  # Misericordia
		16327,  # Pepita Jiménez
		15949,  # Doña Perfecta
		18797,  # Los Pazos de Ulloa
		18384,  # Cuentos de Amor
		18255,  # Cuentos de Clarín
		28669,  # Cuentos de mi tía Panchita (Infantiles)
		13199,  # El sombrero de tres picos
		12053,  # Rimas y Leyendas (Bécquer)
		19782,  # La de Bringas
		17409,  # Clemencia
		17596,  # El Escándalo
		17798,  # La Gaviota
		18136,  # El Niño de la Bola
		18408,  # Peñas arriba
		18911,  # La Puchera
		19323,  # Sotileza
		19448,  # El sabor de la tierruca
	]
	
	output_dir = "storage/curriculum"
	os.makedirs(output_dir, exist_ok=True)
	output_file = os.path.join(output_dir, "pretrain_corpus_es.txt")
	
	total_bytes = 0
	book_count = 0
	
	with open(output_file, "w", encoding="utf-8") as out_f:
		for i, g_id in enumerate(gutenberg_ids):
			# Intentar con la URL de cache que es más directa y rápida
			url = f"https://www.gutenberg.org/cache/epub/{g_id}/pg{g_id}.txt"
			print(f"[{i+1}/{len(gutenberg_ids)}] Descargando ID {g_id}...")
			
			try:
				r = requests.get(url, timeout=30)
				if r.status_code == 200:
					text = r.text
				else:
					# Fallback a la URL de files si falla cache
					fallback_url = f"https://www.gutenberg.org/files/{g_id}/{g_id}-0.txt"
					r = requests.get(fallback_url, timeout=30)
					if r.status_code == 200:
						text = r.text
					else:
						print(f"  ❌ Fallaron ambas URLs para ID {g_id}. Status: {r.status_code}")
						continue
				
				cleaned = clean_gutenberg_text(text)
				out_f.write(cleaned + "\n\n")
				
				bytes_written = len(cleaned.encode("utf-8"))
				total_bytes += bytes_written
				book_count += 1
				print(f"  ✓ Completado. Guardados {bytes_written / 1024 / 1024:.2f} MB")
				
				# Respetar el servidor de Gutenberg con una pequeña pausa
				time.sleep(1)
				
			except Exception as e:
				print(f"  ❌ Error descargando ID {g_id}: {e}")
				
	print("\n✨ Corpus completado exitosamente.")
	print(f"  Libros procesados: {book_count}")
	print(f"  Archivo de salida: {output_file}")
	print(f"  Tamaño final: {total_bytes / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
	download_corpus()
