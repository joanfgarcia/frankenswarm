import csv
import os


def combine_corpora():
	gutenberg_path = "storage/curriculum/pretrain_corpus_es.txt"
	borges_csv_path = "storage/curriculum/borges/datasets/full_corpus.csv"
	
	if not os.path.exists(gutenberg_path):
		print(f"❌ Error: El corpus de Gutenberg no existe en {gutenberg_path}")
		return
		
	if not os.path.exists(borges_csv_path):
		print(f"❌ Error: El corpus de Borges no existe en {borges_csv_path}")
		return
		
	print("📖 Cargando corpus de Gutenberg existente...")
	with open(gutenberg_path, encoding="utf-8") as f:
		gutenberg_text = f.read()
	print(f"  ✓ Gutenberg corpus cargado. Longitud: {len(gutenberg_text):,} caracteres.")
	
	print("📖 Cargando y parseando corpus de Borges y autores latinoamericanos...")
	borges_stories = []
	with open(borges_csv_path, encoding="utf-8") as f:
		# Usamos csv.reader para evitar dependencias externas como pandas
		reader = csv.reader(f)
		header = next(reader)
		# Encontrar el índice de la columna 'text' y 'author'
		try:
			text_idx = header.index("text")
			author_idx = header.index("author")
			title_idx = header.index("title")
		except ValueError:
			print("❌ Error: Columnas no encontradas en el CSV de Borges.")
			return
			
		for row in reader:
			if len(row) > max(text_idx, author_idx, title_idx):
				text = row[text_idx].strip()
				author = row[author_idx].strip()
				title = row[title_idx].strip()
				if text:
					# Añadimos un pequeño encabezado para separar historias si queremos,
					# o simplemente el texto limpio para pre-entrenamiento.
					# Unir los párrafos limpiamente
					cleaned_text = f"\n\n--- {title} por {author} ---\n\n" + text
					borges_stories.append(cleaned_text)
					
	combined_borges_text = "\n\n".join(borges_stories)
	print(f"  ✓ Corpus de Borges procesado. Total de cuentos: {len(borges_stories)}. Longitud: {len(combined_borges_text):,} caracteres.")
	
	# Combinar ambos textos
	final_combined_text = gutenberg_text + "\n\n" + combined_borges_text
	print(f"✍️ Escribiendo corpus combinado en {gutenberg_path}...")
	with open(gutenberg_path, "w", encoding="utf-8") as f:
		f.write(final_combined_text)
		
	print(f"✨ ¡Éxito! Corpus combinado y guardado. Nueva longitud total: {len(final_combined_text):,} caracteres.")

if __name__ == "__main__":
	combine_corpora()
