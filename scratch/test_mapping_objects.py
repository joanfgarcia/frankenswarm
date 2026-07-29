import os
import sys

base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(os.path.join(base_dir, "src"))

from bitnet.dictionary_tool import SovereignDictionary

expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
dictionary = SovereignDictionary(expanded_glyphs_path)

words_to_test = [
	"peras", "manzanas", "plátanos", "naranjas", "cajas", "tomates", "lápices",
	"libros", "mesas", "sillas", "lunas", "soles", "estrellas", "juguetes", "flores",
	"más", "menos", "son", "igual", "cuento", "aritmética", "cuenta"
]

print("=== Mapeo de Palabras de Frutas y Objetos ===")
for w in words_to_test:
	mapped = dictionary.map_to_base_word(w)
	print(f"'{w}' -> '{mapped}'")
