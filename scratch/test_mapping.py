import os
import sys

base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(os.path.join(base_dir, "src"))

from bitnet.dictionary_tool import SovereignDictionary

expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
dictionary = SovereignDictionary(expanded_glyphs_path)

words_to_test = [
	"tengo", "manzanas", "rojas", "dos", "pato", "se", "baña", "rápido",
	"nada", "nadie", "profundo", "chimenea", "cálmate", "cuatro", "echar", "tres"
]

print("=== Mapeo de Palabras ===")
for w in words_to_test:
	mapped = dictionary.map_to_base_word(w)
	print(f"'{w}' -> '{mapped}' (en base_vocab: {w in dictionary.vocab_set})")
