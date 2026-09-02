"""Registro de moléculas K-65P v2 (DL-016/DL-017): el vocabulario refundado.

Las moléculas nacen SOLO desde los 65 primos (DL-016) y solo por composición:
idea fuente (inglés) → cláusulas K-65P → glifo = proyección.

Tres leyes aplicadas EN CREACIÓN (DL-017):
  L1. SILENCIO reservado: la huella todo-a-cero no es registrable — es el
      concepto silencio/abstención, no una molécula.
  L2. Inyectividad: la huella nueva no puede repetir ni el SILENCIO, ni el
      glifo identidad de un primo, ni la huella de otra molécula. El rechazo
      significa: definición incompleta (añade el primo o el contraste −1 que
      captura la esencia) o no es un concepto nuevo.
  L3. La huella incluye CONTRASTES: los trits −1 son definición (hielo =
      {agua:+1, frío:+1, mover:−1} ≠ agua fría con mover:+1).

Uso:
	PYTHONPATH=.:../k65p/src .venv/bin/python -c "
	from src.bitnet.vocab.registry_v2 import VocabularyV2
	v = VocabularyV2()
	v.register('cold-water', 'water that is now cold', ['[cold water]', '[move water]'], anchor_primes=['WATER'])
"
"""

import json
from pathlib import Path

import numpy as np

base_dir = Path(__file__).resolve().parents[3]  # frankenswarm/

from src.bitnet.vocab.structured_explication import PRIME_IDX, StructuredExplication  # noqa: E402

N_PRIMES = 65
SILENCE = tuple([0] * N_PRIMES)  # L1: reservado, no registrable

DEFAULT_V2 = base_dir / "configs" / "k65p_v2"


class InjectionError(ValueError):
	"""L2: la huella ya existe — definición incompleta o concepto no nuevo."""


class VocabularyV2:
	def __init__(self, root: Path = DEFAULT_V2):
		self.root = Path(root)
		self.molecules_path = self.root / "moleculas.json"
		self.primos = json.loads((self.root / "primos.json").read_text(encoding="utf-8"))
		self._prime_fps = {tuple(p["glyph"]): p["names"]["en"] for p in self.primos["primes"]}
		self.molecules: dict[str, dict] = {}
		if self.molecules_path.exists():
			self.molecules = json.loads(self.molecules_path.read_text(encoding="utf-8"))

	def _fingerprint_of(self, surface: str, clauses: list[str], anchor_primes: list[str]) -> tuple[int, ...]:
		exp = StructuredExplication(surface, clauses, anchor_primes=anchor_primes)
		if exp.errors:
			raise ValueError(f"cláusulas inválidas: {exp.errors}")
		return tuple(int(x) for x in exp.to_glyph())

	def check_injectivity(self, fingerprint: tuple[int, ...]) -> None:
		"""L1+L2: la huella debe ser nueva en los tres registros."""
		if fingerprint == SILENCE:
			raise InjectionError("L1: la huella todo-a-cero es SILENCIO — reservado, no registrable")
		if fingerprint in self._prime_fps:
			raise InjectionError(f"L2: la huella coincide con el primo '{self._prime_fps[fingerprint]}' — "
				f"una molécula no puede duplicar un primo; si el concepto es el primo, úsalo como átomo")
		if fingerprint in self._molecule_fps():
			owner = self._molecule_fps()[fingerprint]
			raise InjectionError(f"L2: la huella ya es de '{owner}' — definición incompleta: "
				f"añade el primo o el contraste (−1) que capture la esencia, o no es un concepto nuevo")

	def _molecule_fps(self) -> dict:
		return {tuple(m["glyph"]): name for name, m in self.molecules.items()}

	def register(self, surface: str, idea: str, clauses: list[str], anchor_primes: list[str] | None = None, notes: str = "") -> dict:
		fps = self._fingerprint_of(surface, clauses, anchor_primes or [])
		self.check_injectivity(fps)

		exp = StructuredExplication(surface, clauses, anchor_primes=anchor_primes or [])
		entry = {
			"surface": surface,
			"idea": idea,  # LA FUENTE: lo que la molécula debe transmitir (DL-016)
			"clauses": clauses,
			"anchor_primes": anchor_primes or [],
			"glyph": list(fps),
			"role_profile": exp.role_profile(),
			"kind": exp.kind_candidate(),
			"notes": notes,
		}
		self.molecules[surface] = entry
		self._save()
		return entry

	def _save(self) -> None:
		self.root.mkdir(parents=True, exist_ok=True)
		self.molecules_path.write_text(
			json.dumps(self.molecules, ensure_ascii=False, indent=1), encoding="utf-8"
		)

	def glyph_of(self, surface: str) -> np.ndarray:
		return np.array(self.molecules[surface]["glyph"], dtype=np.int8)
