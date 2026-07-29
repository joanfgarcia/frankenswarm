import os
import re
import time

import requests


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

def download_corpus():
	print("🚀 Iniciando descarga del corpus de pre-entrenamiento en Inglés...")
	
	gutenberg_ids = [
		11,     # Alice's Adventures in Wonderland
		16,     # Peter Pan
		74,     # Tom Sawyer
		76,     # Huckleberry Finn
		1952,   # The Yellow Fairy Book
		33761,  # The Blue Fairy Book
		120,    # Treasure Island
		10947,  # Grimms' Fairy Tales
		1661,   # Sherlock Holmes
		84,     # Frankenstein
		1342,   # Pride and Prejudice
		2701,   # Moby Dick
	]
	
	output_dir = "storage/curriculum"
	os.makedirs(output_dir, exist_ok=True)
	output_file = os.path.join(output_dir, "pretrain_corpus_en.txt")
	
	total_bytes = 0
	book_count = 0
	
	with open(output_file, "w", encoding="utf-8") as out_f:
		for i, g_id in enumerate(gutenberg_ids):
			url = f"https://www.gutenberg.org/cache/epub/{g_id}/pg{g_id}.txt"
			print(f"[{i+1}/{len(gutenberg_ids)}] Descargando ID {g_id}...")
			
			try:
				r = requests.get(url, timeout=30)
				if r.status_code == 200:
					text = r.text
				else:
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
				
				time.sleep(1)
				
			except Exception as e:
				print(f"  ❌ Error descargando ID {g_id}: {e}")
				
	print("\n✨ Corpus EN completado exitosamente.")
	print(f"  Libros procesados: {book_count}")
	print(f"  Archivo de salida: {output_file}")
	print(f"  Tamaño final: {total_bytes / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
	download_corpus()
