"""Explicaciones NSM → glifo de 65 trits (DL-014: doctrina "todo glifo descompone").

CONVENCIÓN DE CODIFICACIÓN (reingenierada de los 28 glifos canónicos):
  trit[i] = +1  el primo i es parte de la definición del concepto (afirmativo)
  trit[i] = -1  el primo i aparece en la definición CONTRASTADO/NEGADO
                ("noche" −ver: no se ve; "piedra" −mover: no se mueve)
  trit[i] =  0  el primo i es irrelevante para este concepto

El glifo es un VECTOR DE RASGOS NSM: 65 juicios semánticos. Todo trit lleva
significado — no hay trits decorativos (doctrina del operador, 1-sep).

Calibración: el codificador debe REPRODUCIR los 28 glifos canónicos a partir
de sus explicaciones NSM (el test de ida-vuelta del RFC-002 §5).
"""
import numpy as np

from src.bitnet.vocab.glyph_vocabulary import SEMANTIC_PRIMES

PRIME_IDX = {name: i for i, name in enumerate(SEMANTIC_PRIMES)}


def explication_to_glyph(features: dict[str, int], vocab=None) -> np.ndarray:
	"""features: {nombre_de_primo: +1 | -1 | 0} (claves canónicas ES, minúsculas).
	Los primos sin mención quedan a 0. Devuelve el glifo (65,) int8.

	Si `vocab` (lista de palabras del censo) se pasa, se valida que cada primo
	mencionado exista en la tabla (un typo del LLM no genera glifos silvestres).
	"""
	glyph = np.zeros(len(SEMANTIC_PRIMES), dtype=np.int8)
	for name, val in features.items():
		key = name.strip().lower()
		if key not in PRIME_IDX:
			if vocab is not None:
				raise ValueError(f"primo desconocido {name!r} — no está en los 65")
			continue
		if val not in (-1, 0, 1):
			raise ValueError(f"valor {val!r} inválido para {name!r} (solo -1/0/1)")
		glyph[PRIME_IDX[key]] = val
	return glyph


def glyph_to_features(glyph: np.ndarray) -> dict[str, int]:
	"""Glifo → explicación (los primos activos con su polaridad)."""
	return {SEMANTIC_PRIMES[i]: int(glyph[i]) for i in np.nonzero(np.asarray(glyph))[0]}


def explicate(words: list[str], features_by_word: dict[str, dict[str, int]]) -> dict[str, np.ndarray]:
	"""Conveniencia: {(palabra → glifo)} para un lote de explicaciones."""
	out = {}
	for w in words:
		feats = features_by_word.get(w)
		if feats is None:
			raise ValueError(f"sin explicación para {w!r}")
		out[w] = explication_to_glyph(feats)
	return out


def consistency_report(glyphs: dict[str, np.ndarray], pairs: list[tuple[str, str, float]]) -> list:
	"""Valida la consistencia semántica entre glifos:
	pairs = (palabra_a, palabra_b, solapamiento_esperado 0..1).
	Solapamiento = Jaccard de primos activos. Devuelve las violaciones:
	si se espera solape alto (puppy~dog) y no lo hay → inconsistencia."""
	errs = []
	for a, b, expected in pairs:
		ga = set(np.nonzero(glyphs[a])[0].tolist())
		gb = set(np.nonzero(glyphs[b])[0].tolist())
		jac = len(ga & gb) / max(1, len(ga | gb))
		if jac < expected:
			errs.append((a, b, f"solape {jac:.2f} < esperado {expected:.2f}"))
	return errs
