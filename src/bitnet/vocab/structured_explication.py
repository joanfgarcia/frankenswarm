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

from k65p.core import Atom, parse_k65p  # noqa: E402
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
MENTAL = {12, 13, 14, 15, 16, 17, 18}  # pensar saber querer sentir ver oír decir
STRUCTURAL_HEADS = {48}  # si: andamiaje hipotético puro — como G, no predica
# (DL-023 refinado): el conectivo condicional no es contenido del concepto;
# sí lo son sus ramas. (porque/like son asertivos y sí cuentan: la causalidad
# y la similitud SÍ se predican del ancla.)

# REGLA MODAL (DL-014 refinado 3-sep): la negación de un predicado MENTAL niega
# el estado, no el complemento — "no sé si puedo hacerlo" niega el SABER, pero
# el hacer de su interior es hipotético, no negado: el contenido hipotético NO
# define (polaridad vacía bajo él). La negación de contenido FÍSICO sí atribuye
# el contraste al ancla ([not [move sleep]] → mover:−1, el patrón hielo).
UNARY_OPS = {44, 45, 46, 31, 30, 32, 37, 41, 40, 38, 39, 43, 49}  # +49 VERY (DL-020)
BINARY_OPS = {48, 47, 51}


class StructuredExplication:
	def __init__(self, surface: str, clauses: list[str], kind_declared: str | None = None, category: str | None = None, anchor_primes: list[str] | None = None, molecule_glyphs: dict[str, list[int]] | None = None):
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
		# LA MALLA (doctrina del operador, 2-sep): los glifos ya posicionan los
		# conceptos antes del entrenamiento. Cuando una cláusula usa una
		# molécula como cabeza ([danger predator]), el glifo de esa molécula
		# PROPAGA sus trits a la proyección — la composición es recursiva.
		self.molecule_glyphs = {
			k.casefold(): np.array(v, dtype=np.int8) for k, v in (molecule_glyphs or {}).items()
		}
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
	@staticmethod
	def _head_id(atom) -> int | None:
		name_cf = atom.name.casefold()
		if name_cf.lstrip("-").isdigit():
			return int(name_cf)
		return SYMBOL_TO_ID.get(name_cf)

	@staticmethod
	def _has_anchor_list(node, anchors: set) -> bool:
		if isinstance(node, list):
			return any(StructuredExplication._has_anchor_list(c, anchors) for c in node[1:])
		return node.name.casefold() in anchors if hasattr(node, "name") else False

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
				if (isinstance(a, list) and a and isinstance(a[0], Atom)
						and self._head_id(a[0]) in MENTAL):
					# el estado negado SÍ define (know:−1); su complemento no:
					# [not [know people [can [do try]]]] niega el SABER de try,
					# pero hacer/poder son hipotéticos, no negados (try = hacer
					# queriendo sin saber — no "hacer cancelado")
					mhid = self._head_id(a[0])
					mname = self._resolve(a[0].name)
					if self._has_anchor_list(a, anchors):
						hits[mname] = hits.get(mname, 0) - abs(polarity)
						roles.append((mname, ROLE_FRAMES[mhid][0], True))
				else:
					self._walk(a, -abs(polarity), anchors, hits, roles)
			return

		# ¿el ancla participa en esta cláusula? A nivel de átomo directo define
		# el rol; a nivel de subárbol basta para que la cabeza predique del ancla
		names = [a.name.casefold() for a in args if not isinstance(a, list)]
		anchor_atom = bool(anchors & set(names))
		anchor_subtree = any(self._has_anchor(a, anchors) for a in args)

		if anchor_subtree and head_id is not None and head_id not in STRUCTURAL_HEADS:
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
		elif anchor_subtree and head_name in self.molecule_glyphs:
			# LA MALLA: cabeza-molécula → su glifo propaga trits con polaridad
			mg = self.molecule_glyphs[head_name]
			sign = -1 if negated else 1
			for j in range(N_PRIMES):
				if mg[j] != 0:
					pname = PRIME_NAME[j]
					hits[pname] = hits.get(pname, 0) + sign * int(mg[j])
			# el ancla recibe el predicado-molécula: atribución compuesta
			roles.append((head_name, "receives_attribute", negated))

		# proyección de argumentos de contenido no genéricos
		# REGLA DE GRUPO (3-sep, auditoría del operador): dentro de un grupo-G
		# los miembros son PERTENENCIA definicional, no relleno de rol — los
		# placeholders genéricos (gente/alguien) SÍ cuentan en [G ...].
		# [G people child] aporta gente:+1; [want people X] no aporta nada.
		is_group = head_name in ("G", "g")
		for i, a in enumerate(args):
			if isinstance(a, list):
				self._walk(a, polarity, anchors, hits, roles)
				continue
			name = self._resolve(a.name)
			if name in anchors:
				continue
			if name in GENERIC_ROLES and not is_group:
				# placeholder genérico de ROL: no caracteriza el concepto
				# (salvo negado: [not [G people X]] = "X no es de la gente")
				if negated:
					hits[name] = hits.get(name, 0) - 1
				continue
			# LA MALLA unificada (DL-018/020/022): CUALQUIER argumento de
			# contenido propaga — molécula → su glifo completo (cadena
			# definicional baby→child→person, prototipos fire→red); primo o
			# palabra → juicio directo.
			# REGLA DE GRUPO-TÉRMINO (3-sep): un grupo-G como argumento aporta sus
			# miembros lleve ancla o no — el prototipo se invoca entero
			# ([like new [G short_time thing]]: new es como la cosa-de-poco-tiempo).
			if head_id in UNARY_OPS or head_id in BINARY_OPS:
				self._contribute(name, negated, hits)
			elif anchor_atom or is_group:
				self._contribute(name, negated, hits)

	def _contribute(self, name: str, negated: bool, hits: dict) -> None:
		"""Contribución de un argumento de contenido: si es molécula, la malla
		propaga su glifo (con polaridad); si no, cuenta como juicio."""
		sign = -1 if negated else 1
		if name in self.molecule_glyphs:
			mg = self.molecule_glyphs[name]
			for j in range(N_PRIMES):
				if mg[j] != 0:
					pn = PRIME_NAME[j]
					hits[pn] = hits.get(pn, 0) + sign * int(mg[j])
		else:
			hits[name] = hits.get(name, 0) + sign

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
