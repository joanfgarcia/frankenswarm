"""Léxico EN → índices de primo para el compilador EN→K-65P (DL-012).

Los átomos de K-65P son etiquetas superficiales de vectores de embedding de
primos: la lengua humana es cosmética. Este módulo fija la piel INGLESA
(corpus BIT-003) y las palabras-superficie que activan cada primo como
operador o átomo. El validador nsm_syntax es la fuente de verdad de la
gramática; este léxico es la fuente de verdad del VOCABULARIO de primos.
"""

# Índice de primo → forma superficial EN canónica
PRIME_EN = {
	0: "i", 1: "you", 2: "somebody", 3: "people", 4: "something", 5: "thing",
	6: "body", 7: "part", 8: "good", 9: "bad", 10: "big", 11: "small",
	12: "think", 13: "know", 14: "want", 15: "feel", 16: "see", 17: "hear",
	18: "say", 19: "word", 20: "truth", 21: "do", 22: "happen", 23: "move",
	24: "touch", 25: "be", 26: "mine", 27: "live", 28: "die",
	29: "when", 30: "now", 31: "before", 32: "after",
	33: "long time", 34: "a moment", 35: "moment",
	36: "where", 37: "here", 38: "above", 39: "below", 40: "far", 41: "near",
	42: "side", 43: "inside",
	44: "not", 45: "maybe", 46: "can", 47: "because", 48: "if", 49: "very",
	50: "more", 51: "like", 52: "this", 53: "same", 54: "other",
	55: "one", 56: "two", 57: "some", 58: "all", 59: "much",
	60: "hot", 61: "cold", 62: "water", 63: "light", 64: "dark",
}

# Palabras-superficie EN adicionales que activan el primo (alias)
PRIME_ALIASES = {
	0: {"me", "myself"}, 1: {"your"}, 2: {"someone", "anybody"}, 3: {"folks"},
	4: {"anything", "stuff"}, 5: {"things"}, 8: {"nice", "good"}, 9: {"naughty"},
	10: {"bigger", "big"}, 11: {"little", "tiny", "smaller"},
	12: {"thinks", "thinking"}, 13: {"knows"}, 14: {"wants"}, 15: {"feels"},
	16: {"sees", "look"}, 17: {"hears", "listen"}, 18: {"says", "tell"},
	21: {"does", "make"}, 22: {"happens", "happened"}, 23: {"moves", "go"},
	24: {"touches", "touch"}, 25: {"is", "are", "am", "was", "were", "be"},
	27: {"lives"}, 28: {"dies"}, 30: {"today", "tonight"}, 31: {"ago", "past"},
	32: {"later", "then"}, 35: {"minute"}, 37: {"in here"}, 38: {"up", "over"},
	39: {"down", "under"}, 40: {"away", "far away"}, 41: {"close"},
	44: {"don't", "doesn't", "not", "no"}, 46: {"may", "could"},
	49: {"really", "so"}, 50: {"more", "most"}, 51: {"likes", "like"},
	52: {"the", "a", "an"}, 55: {"a"}, 57: {"a few"}, 58: {"everything", "everybody"},
	59: {"lots", "plenty"}, 60: {"warm", "hot"}, 61: {"cooler"}, 63: {"sun", "lights"},
	64: {"dark", "night"},
}

# Palabras que se DESCARTAN en la traducción (funciones sin primo propio:
# determinantes ya cubiertos por 52, preposiciones gramaticales, auxiliares)
DROP_WORDS = {"of", "to", "with", "and", "or", "but", "at", "on", "in", "for",
	"from", "by", "was", "were", "been", "being", "has", "have", "had",
	"will", "would", "shall", "should", "did", "there", "that", "it",
	"what", "who", "how", "why", "yes", "oh", "well"}


def build_lexicon() -> dict:
	"""palabra EN (minúscula) → índice de primo. Los alias no canónicos se
	resuelven al mismo primo; la forma canónica gana."""
	lex = {}
	for idx, forms in PRIME_ALIASES.items():
		for f in forms:
			lex.setdefault(f.lower(), idx)
	for idx, canon in PRIME_EN.items():
		for f in canon.split():
			lex.setdefault(f.lower(), idx)
	return lex


if __name__ == "__main__":
	lex = build_lexicon()
	print(f"léxico EN→primo: {len(lex)} palabras-superficie → 65 primos")
	for w in ("i", "you", "see", "water", "fire", "not", "because", "one", "two"):
		print(f"  {w!r} → primo {lex.get(w)}")
