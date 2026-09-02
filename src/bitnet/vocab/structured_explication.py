"""Explicaciones estructuradas K-65P (2-sep): el camino inverso de Wierzbicka.

FUENTE ÚNICA: una lista de cláusulas K-65P por molécula (curada a mano). La
vista legible en castellano se GENERA con `k65p.validator.render` — sin
mantenimiento doble.

Del mismo paquete de cláusulas salen las tres salidas de la fórmula:
  1. PERFIL DE ROLES  → valencias de la molécula (qué puede encabezar/llenar)
  2. CATEGORÍA        → las cabezas de atribución que la definen ("es un X")
  3. TRITS            → proyección plana con ley de calibración contra gold

REGLA DE PROYECCIÓN (a calibrar contra los 28 glifos canónicos):
  "contenido" (v1): de cada cláusula donde el ancla (o la compuesta del ancla)
  llena un rol, el primo de la CABEZA cuenta (predica algo del ancla) y los
  argumentos de contenido cuentan; los placeholders genéricos de rol (gente,
  alguien) NO cuentan; los primos bajo el alcance de `no` (44) cuentan -1;
  los primos bajo `quizá`/`poder` cuentan igual (lo posible también define).

ESTADOS: pending → drafted → structured → explicated (el operador cura).
"""

import json
import sys
from pathlib import Path

import numpy as np

base_dir = Path(__file__).resolve().parents[2]
k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

from k65p.core import parse_k65p  # noqa: E402
from k65p.primes import N_PRIMES, PRIMES, SYMBOL_TO_ID  # noqa: E402
from k65p.validator import parse_k65p as _p  # noqa: F401  (alias histórico)
from k65p.validator import validate  # noqa: E402

from src.bitnet.vocab.glyph_vocabulary import SEMANTIC_PRIMES  # noqa: E402

PRIME_IDX = {name: i for i, name in enumerate(SEMANTIC_PRIMES)}
PRIME_NAME = {i: name for name, i in PRIME_IDX.items()}
EN_TO_ES = {row[0].lower(): row[1] for row in PRIMES}  # 'water' → 'agua_prima'


def _normalize_prime(name: str) -> str:
	"""Acepta nombres de primo en es o en; devuelve la clave es de PRIME_IDX."""
	key = name.casefold()
	if key in PRIME_IDX:
		return key
	return EN_TO_ES.get(key, key)

# Placeholders genéricos de rol: aparecen como relleno sin caracterizar el
# concepto (la gente/alguien como agente-experimentador típico). El "algo" en
# cambio SÍ puede ser contenido característico (ver calibración).
GENERIC_ROLES = {"gente", "alguien"}

# Predicados con rol de ancla "portador" (el ancla es tema/contenido/paciente)
# vs "agentivo" (el ancla actúa). El perfil de roles sale de aquí.
ROLE_FRAMES = {  # prime_id → (nombre del rol del 1er argumento, nombre del 2º)
	21: ("agent", "patient"),      # hacer
	22: ("theme", None),           # pasar
	23: ("theme", "destination"),  # mover
	24: ("agent", "patient"),      # tocar
	25: ("theme", "place"),        # existir
	27: ("theme", "place"),        # vivir
	28: ("theme", None),           # morir
	12: ("experiencer", "content"),
	13: ("experiencer", "content"),
	14: ("experiencer", "content"),
	15: ("experiencer", "state"),
	16: ("experiencer", "theme"),
	17: ("experiencer", "theme"),
	18: ("agent", "content"),
}

EVALUATORS = {8, 9, 10, 11, 26, 60, 61, 64}
NEGATION = 44  # no
UNARY_OPS = {44, 45, 46, 31, 30, 32, 37, 41, 40, 38, 39, 43}
BINARY_OPS = {48, 47, 51}


class StructuredExplication:
	def __init__(self, surface: str, clauses: list[str], kind_declared: str | None = None, category: str | None = None, anchor_primes: list[str] | None = None):
		self.surface = surface
		self.clauses = clauses
		self.kind_declared = kind_declared
		self.category = category
		# los primos que CONSTITUYEN el concepto (p.ej. agua = agua_prima):
		# parte de la definición natural-kind, no mágica de proyección.
		# El ANCLA de proyección es el conjunto: superficie + primos ancla
		# (en una compuesta, las cláusulas hablan de sus primos, no de su nombre)
		self.anchor_primes = [_normalize_prime(p) for p in (anchor_primes or [])]
		self.anchor_names = {surface.casefold()} | set(self.anchor_primes)
		# los primos ancla se nombran en las cláusulas con CUALQUIER renderización
		# (en/es/numérico): todas entran al conjunto de ancla
		for p_es in self.anchor_primes:
			idx = PRIME_IDX.get(p_es)
			if idx is not None:
				self.anchor_names.update(str(r).casefold() for r in PRIMES[idx])
				self.anchor_names.add(str(idx))
		self.errors: list[str] = []
		self._trees = []
		for c in clauses:
			try:
				self._trees.append(parse_k65p(c))
			except SyntaxError as exc:
				self.errors.append(f"{c!r}: {exc}")
		for c in clauses:
			errs = validate(c)
			if errs:
				self.errors.extend(errs)

	# ── extracción ────────────────────────────────────────────────────────
	def _resolve(self, atom_name: str) -> str:
		key = atom_name.casefold()
		if key.lstrip("-").isdigit():
			return PRIME_NAME[int(key)]
		if key in SYMBOL_TO_ID:
			return PRIME_NAME[SYMBOL_TO_ID[key]]
		return key

	@staticmethod
	def _has_anchor(node, anchors: set) -> bool:
		"""¿Aparece algún nombre del ancla en cualquier punto del subárbol?"""
		if isinstance(node, list):
			return any(StructuredExplication._has_anchor(c, anchors) for c in node)
		return node.name.casefold() in anchors

	def _walk(self, node, polarity: int, anchors: set, hits: dict, roles: list):
		"""Recorre un árbol resuelto: proyección + rastreo de roles del ancla."""
		if not isinstance(node, list):
			return
		head, args = node[0], node[1:]
		if isinstance(head, list):  # no debería pasar (validador lo rechaza)
			return
		head_name = self._resolve(head.name)
		name_cf = head.name.casefold()
		if name_cf.lstrip("-").isdigit():
			head_id = int(name_cf)
		else:
			head_id = SYMBOL_TO_ID.get(name_cf)
		negated = polarity < 0

		if head_id == NEGATION:
			for a in args:
				self._walk(a, -abs(polarity), anchors, hits, roles)
			return

		# ¿el ancla participa en esta cláusula? A nivel de átomo directo define
		# el rol; a nivel de subárbol basta para que la cabeza predique del ancla
		names = [a.name.casefold() for a in args if not isinstance(a, list)]
		anchor_atom = bool(anchors & set(names))
		anchor_subtree = any(self._has_anchor(a, anchors) for a in args)

		if anchor_subtree and head_id is not None:
			# la cláusula predica algo del ancla → el primo de la cabeza cuenta
			hits[head_name] = hits.get(head_name, 0) + (-1 if negated else 1)
			# rastreo de roles: ¿qué rol llena el ancla?
			if head_id in ROLE_FRAMES:
				role1, role2 = ROLE_FRAMES[head_id]
				roles.append((head_name, role1, negated))
			elif head_id in EVALUATORS:
				# el ancla RECIBE el atributo (es tema de la atribución):
				# esto la define como entidad, no como atributo-capaz
				roles.append((head_name, "receives_attribute", negated))

		# proyección de argumentos de contenido no genéricos
		for i, a in enumerate(args):
			if isinstance(a, list):
				self._walk(a, polarity, anchors, hits, roles)
				continue
			name = self._resolve(a.name)
			if name in anchors:
				continue
			if name in GENERIC_ROLES and not negated:
				# placeholder genérico: no caracteriza el concepto
				if negated:
					hits[name] = hits.get(name, 0) - 1
				continue
			if head_id in UNARY_OPS or head_id in BINARY_OPS:
				hits[name] = hits.get(name, 0) + (-1 if negated else 1)
			elif anchor_atom:
				# argumento de una cláusula sobre el ancla: contenido
				# (los rellenos concretos caracterizan la definición)
				hits[name] = hits.get(name, 0) + (-1 if negated else 1)

	def project_flat(self, anchors: set | None = None) -> dict[str, int]:
		anchors = anchors or self.anchor_names
		hits: dict[str, int] = {}
		roles: list = []
		for tree in self._trees:
			self._walk(tree, 1, anchors, hits, roles)
		self.roles_seen = roles
		return hits

	def to_glyph(self, anchors: set | None = None) -> np.ndarray:
		hits = self.project_flat(anchors)
		g = np.zeros(N_PRIMES, dtype=np.int8)
		for name in self.anchor_primes:
			if name in PRIME_IDX:
				g[PRIME_IDX[name]] = 1
		for name, v in hits.items():
			idx = PRIME_IDX.get(name)
			if idx is None:
				continue
			g[idx] = 1 if v > 0 else (-1 if v < 0 else 0)
		return g

	def role_profile(self) -> dict[str, bool]:
		roles = getattr(self, "roles_seen", None) or []
		profile = {"agent_capable": False, "patient_capable": False, "experiencer_capable": False, "attribute_capable": False, "receives_attributes": False}
		for head, role, negated in roles:
			if negated:
				continue
			if role == "agent":
				profile["agent_capable"] = True
			if role in ("patient", "theme", "place", "content", "state"):
				profile["patient_capable"] = True
			if role == "experiencer":
				profile["experiencer_capable"] = True
			if role == "attribution_head":
				profile["attribute_capable"] = True
			if role == "receives_attribute":
				profile["receives_attributes"] = True
		return profile

	def kind_candidate(self) -> str:
		roles = getattr(self, "roles_seen", None) or []
		if any(r == "attribution_head" and not n for _, r, n in roles):
			return "attribute"
		if any(r == "receives_attribute" and not n for _, r, n in roles):
			return "entity"
		return "event"

	def render_es(self) -> list[str]:
		from k65p.validator import render
		return [render(c, "es") for c in self.clauses]


def load_explication(path: str | Path) -> StructuredExplication:
	data = json.loads(Path(path).read_text(encoding="utf-8"))
	return StructuredExplication(
		data["surface"], data["clauses"], data.get("kind_declared"), data.get("category"), data.get("anchor_primes")
	)


def calibrate(exp: StructuredExplication, gold_glyph: np.ndarray) -> dict:
	proj = exp.to_glyph()
	idx = {name: i for i, name in enumerate(SEMANTIC_PRIMES)}
	diff = []
	for name, i in idx.items():
		if proj[i] != gold_glyph[i]:
			diff.append((name, int(gold_glyph[i]), int(proj[i])))
	return {
		"match": not diff,
		"n_diff": len(diff),
		"diff": diff,  # (primo, gold, proyectado)
		"role_profile": exp.role_profile(),
		"render_es": exp.render_es(),
	}
