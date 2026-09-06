"""Validación del vocabulario v2 (auditoría 3-sep): la batería canónica.

Sustituye a los heredocs inline: corre la semilla + borradores + diferidos en
orden definicional usando la lógica REAL del registro (check_injectivity: L1
silencio, duplicado de primo, colisión) sobre una copia rascable, y emite:
  - gramática (validador k65p)
  - L1/L2/colisiones (las tres leyes DL-017)
  - contenido-propio: trits primo que cada molécula aporta SIN la malla
    (regla de robustez: ≥2 propios; por debajo = aviso, no rechazo — la
    definición relacional es legítima pero frágil si sus referencias evaporan)

Uso:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/validate_vocab.py
"""

import sys
import tempfile
from pathlib import Path

base = Path(__file__).resolve().parents[1]
sys.path.append(str(base))

from k65p.validator import validate  # noqa: E402

from scripts.generate_cata_doc import DEFERRED_DRAFTS, DRAFTS  # noqa: E402
from scripts.seed_vocab_v2 import SEED  # noqa: E402
from src.bitnet.vocab.registry_v2 import InjectionError, VocabularyV2  # noqa: E402
from src.bitnet.vocab.structured_explication import PRIME_IDX, StructuredExplication  # noqa: E402


def parts(e):
	lst = list(e)
	while len(lst) < 4:
		lst.append([])
	return lst[0], lst[1], lst[2], lst[3]


def own_trits(surface: str, clauses: list[str]) -> set[str]:
	"""Trits primo aportados por las cláusulas SIN arrastre de moléculas."""
	exp = StructuredExplication(surface, clauses, molecule_glyphs={})
	hits = exp.project_flat()
	return {PRIME_IDX[i] for i in range(65) if False} | {
		name for name, val in hits.items() if val != 0 and name in PRIME_IDX
	}


def main() -> int:
	v = VocabularyV2()
	with tempfile.TemporaryDirectory() as td:
		v.molecules_path = Path(td) / "scratch.json"
		v.molecules = {}  # rascable limpio: la tabla de primos sí se conserva
		fps: dict = {}
		n_bad, problems = 0, []
		for e in SEED:
			surface, idea, clauses, drags = parts(e)
			for c in clauses + drags:
				if validate(c):
					n_bad += 1
					problems.append(("GRAMÁTICA(seed)", surface, c))
			exp = StructuredExplication(surface, clauses, molecule_glyphs=v._molecule_glyphs(),
				drag_clauses=drags)
			g = tuple(int(x) for x in exp.to_glyph())
			try:
				v.check_injectivity(g)
			except InjectionError as ex:
				problems.append((f"L1/L2(seed)", surface, str(ex)[:80]))
				continue
			fps[g] = surface
			v.molecules[surface] = {"glyph": list(g), "clauses": clauses}
		for w, idea, nsm, clauses, layer in DRAFTS + DEFERRED_DRAFTS:
			for c in clauses:
				if validate(c):
					n_bad += 1
					problems.append(("GRAMÁTICA", w, c))
			exp = StructuredExplication(w, clauses, molecule_glyphs=v._molecule_glyphs())
			g = tuple(int(x) for x in exp.to_glyph())
			try:
				v.check_injectivity(g)
			except InjectionError as ex:
				problems.append(("L1/L2", w, str(ex)[:80]))
				continue
			fps[g] = w
			v.molecules[w] = {"glyph": list(g), "clauses": clauses}

		print(f"gramática: {'OK ✓' if n_bad == 0 else str(n_bad) + ' ERRORES'}")
		for p in problems:
			print(" ", p)
		print(f"registrables: {len(fps)} (semilla + borradores + diferidos)")

		# informe de contenido-propio (robustez ante evaporación de referencias)
		print("\n── contenido-propio (<2 trits primo sin malla = frágil si evaporan refs) ──")
		thin = []
		items = [(e[0], e[2]) for e in SEED] + [(w, c) for w, _, _, c, _ in DRAFTS]
		for surface, clauses in items:
			own = own_trits(surface, clauses)
			if len(own) < 2:
				thin.append((surface, sorted(own)))
		for surface, own in thin:
			print(f"  ⚠ {surface:12s} propios: {own}")
		if not thin:
			print("  (ninguna: todas aportan ≥2 trits propios)")
	return 0 if n_bad == 0 else 1


if __name__ == "__main__":
	sys.exit(main())
