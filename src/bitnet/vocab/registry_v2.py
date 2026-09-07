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

	def _glyph_of_name(self, name: str) -> np.ndarray:
		"""Glifo de un primo (identidad) o de una molécula del registro."""
		from src.bitnet.vocab.structured_explication import _normalize_prime, PRIME_IDX
		if name in self.molecules:
			return np.array(self.molecules[name]["glyph"], dtype=np.int8)
		p = _normalize_prime(name)
		if p in PRIME_IDX:
			g = np.zeros(65, dtype=np.int8)
			g[PRIME_IDX[p]] = 1
			return g
		raise ValueError(f"'{name}' no es ni primo ni molécula registrada")

	def _molecule_glyphs(self) -> dict[str, list[int]]:
		"""LA MALLA: todos los glifos registrados, disponibles para la proyección."""
		return {name: m["glyph"] for name, m in self.molecules.items()}

	@staticmethod
	def _F(head_g: np.ndarray, mod_g: np.ndarray) -> np.ndarray:
		"""F v1 (candidata, pendiente de calibrar): unión con CABEZA DOMINANTE.
		No conmutativa en conflictos: donde cabeza y modificador discrepan de
		signo, gana la cabeza (la categoría define; el modificador matiza)."""
		out = head_g.copy()
		mask = head_g == 0
		out[mask] = mod_g[mask]
		return out

	def _reference_graph(self) -> dict[str, set[str]]:
		"""Grafo de referencias: molécula → moléculas que nombran sus cláusulas."""
		import re as _re
		graph: dict[str, set[str]] = {}
		names = set(self.molecules)
		for surface, m in self.molecules.items():
			refs = set()
			for c in m["clauses"] + m.get("drags", []):
				for t in _re.findall(r"[a-zA-Z_]+", c):
					tl = t.casefold()
					if tl != surface.casefold() and tl in names:
						refs.add(tl)
			graph[surface.casefold()] = refs
		return graph

	@staticmethod
	def _scc_tarjan(graph: dict[str, set[str]]) -> list[list[str]]:
		"""Componentes fuertemente conexas (Tarjan iterativo)."""
		index_of, low, on_stack = {}, {}, set()
		stack, sccs, counter = [], [], [0]
		for root in graph:
			if root in index_of:
				continue
			work = [(root, iter(sorted(graph.get(root, ()))))]
			while work:
				node, it = work[-1]
				if node not in index_of:
					index_of[node] = low[node] = counter[0]
					counter[0] += 1
					stack.append(node)
					on_stack.add(node)
				advanced = False
				for nxt in it:
					if nxt not in index_of:
						work.append((nxt, iter(sorted(graph.get(nxt, ())))))
						advanced = True
						break
					elif nxt in on_stack:
						low[node] = min(low[node], index_of[nxt])
				if advanced:
					continue
				work.pop()
				if work:
					parent = work[-1][0]
					low[parent] = min(low[parent], low[node])
				if low[node] == index_of[node]:
					comp = []
					while True:
						w = stack.pop()
						on_stack.discard(w)
						comp.append(w)
						if w == node:
							break
					sccs.append(comp)
		return sccs

	def reproject_all(self, max_iters: int = 20) -> dict[str, int]:
		"""Punto fijo por componentes (SCC): arrastre TOTAL en aristas DAG,
		solo-PROPIO dentro de ciclos. Las referencias mutuas (dog↔bark) comparten
		solo su contenido propio — si no, convergerían al mismo glifo y el modelo
		no distinguiría 'dog' de 'bark'. Determinista, sin dependencia de orden.
		Devuelve {superficie: tamaño_de_su_SCC}."""
		from src.bitnet.vocab.structured_explication import StructuredExplication as _SE
		graph = self._reference_graph()
		sccs = self._scc_tarjan(graph)
		comp_of = {n: i for i, comp in enumerate(sccs) for n in comp}
		# orden topológico de componentes (dependencias primero)
		deps: dict[int, set[int]] = {i: set() for i in range(len(sccs))}
		for i, comp in enumerate(sccs):
			for n in comp:
				for r in graph.get(n, ()):
					j = comp_of.get(r, -1)
					if j != -1 and j != i:
						deps[i].add(j)
		ordered, seen = [], set()
		def visit(i: int) -> None:
			if i in seen:
				return
			seen.add(i)
			for j in sorted(deps[i]):
				visit(j)
			ordered.append(i)
		for i in range(len(sccs)):
			visit(i)
		# fase 1: glifos propios de TODAS (mapa vacío — sin arrastre)
		own: dict[str, list[int]] = {}
		for surface, m in self.molecules.items():
			exp = _SE(surface, m["clauses"], anchor_primes=m.get("anchor_primes", []),
				molecule_glyphs={}, drag_clauses=m.get("drags", []))
			if exp.errors:
				continue
			own[surface.casefold()] = [int(x) for x in exp.to_glyph()]
		# fase 2: por componentes en orden — DAG: glifos completos; ciclo: solo propios
		full: dict[str, list[int]] = {}
		for i in ordered:
			comp = sccs[i]
			avail = dict(full)
			for n in comp:
				avail[n] = own.get(n, [0] * 65)
			for surface in comp:
				m = next(mm for ss, mm in self.molecules.items() if ss.casefold() == surface)
				exp = _SE(m.get("surface", surface), m["clauses"],
					anchor_primes=m.get("anchor_primes", []),
					molecule_glyphs={k: v for k, v in avail.items()},
					drag_clauses=m.get("drags", []))
				if exp.errors:
					continue
				full[surface] = [int(x) for x in exp.to_glyph()]
		for surface, m in self.molecules.items():
			if surface.casefold() in full:
				m["glyph"] = full[surface.casefold()]
		self._save()
		multi = {i: comp for i, comp in enumerate(sccs) if len(comp) > 1}
		return {n: len(sccs[comp_of[n]]) for comp in multi.values() for n in comp}

	def register(self, surface: str, idea: str, clauses: list[str], anchor_primes: list[str] | None = None, notes: str = "", drag_clauses: list[str] | None = None) -> dict:
		exp = StructuredExplication(surface, clauses, anchor_primes=anchor_primes or [],
			molecule_glyphs=self._molecule_glyphs(), drag_clauses=drag_clauses or [])
		if exp.errors:
			raise ValueError(f"cláusulas inválidas: {exp.errors}")
		fps = tuple(int(x) for x in exp.to_glyph())
		self.check_injectivity(fps)
		entry = {
			"surface": surface,
			"idea": idea,  # LA FUENTE: lo que la molécula debe transmitir (DL-016)
			"clauses": clauses,
			"drags": drag_clauses or [],  # ARRASTRE asociativo (ablacionable en instrumentos)
			"anchor_primes": anchor_primes or [],
			"glyph": list(fps),  # completo: núcleo + arrastre
			"glyph_core": [int(x) for x in exp.to_glyph(include_drags=False)],
			"role_profile": exp.role_profile(),
			"kind": exp.kind_candidate(),
			"notes": notes,
		}
		self.molecules[surface] = entry
		self._save()
		return entry

	def register_compound(self, surface: str, head: str, own_clauses: list[str], idea: str, notes: str = "") -> dict:
		"""DL-018: molécula COMPUESTA — [cabeza modificador] = "un tipo de cabeza".

		Wierzbicka inversa: "es una clase de {head}; {own_clauses}". El glifo =
		F(glifo_cabeza, proyección de las cláusulas propias) — la cabeza aporta
		su huella entera (la categoría se hereda), las cláusulas aportan lo nuevo.
		Ley de conservación: los trits de la compuesta deben poder descomponerse
		en cabeza + cláusulas propias (*todo glifo descompone*, DL-014).
		"""
		# guardia anti-ciclos (auditoría 3-sep): la cadena cabeza→cabeza no
		# puede regresar a la superficie que se registra (A→B→A imposible)
		seen, h = {surface.casefold()}, head
		while h in self.molecules and "compound" in self.molecules[h]:
			h = self.molecules[h]["compound"]["head"]
			if h.casefold() in seen:
				raise InjectionError(f"ciclo de compuestas: {surface} → ... → {h} → {surface}")
			seen.add(h.casefold())

		head_g = self._glyph_of_name(head)
		exp = StructuredExplication(surface, own_clauses, anchor_primes=[],
			molecule_glyphs=self._molecule_glyphs())
		if exp.errors:
			raise ValueError(f"cláusulas propias inválidas: {exp.errors}")
		mod_g = exp.to_glyph()  # lo NUEVO que aporta la compuesta
		fps = tuple(int(x) for x in self._F(head_g, mod_g))
		self.check_injectivity(fps)
		entry = {
			"surface": surface,
			"idea": idea,
			"compound": {"head": head, "own_clauses": own_clauses},
			"clauses": own_clauses,
			"anchor_primes": [],
			"glyph": list(fps),
			"role_profile": exp.role_profile(),
			"kind": "compound",
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
