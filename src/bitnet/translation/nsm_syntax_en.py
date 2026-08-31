"""Gramática NSM-C en superficie INGLESA (compañera de nsm_syntax.py, DL-012).

Mismos primos, mismos roles, mismas aridades que la gramática ES fuente de
verdad — la lengua es cosmética: los átomos son etiquetas de vectores de
primos. El parser/linearize son agnósticos (se importan); solo las tablas de
operadores son específicas de superficie.
"""
from src.bitnet.translation.nsm_syntax import NSMSyntaxError, linearize, parse, tokenize

# predicado: (min_args, max_args, roles) — 1:1 con los PREDICADOS ES
PREDICADOS = {
	"do":     (1, 3, ("agent", "action", "patient")),
	"happen": (1, 2, ("theme", "experiencer")),
	"move":   (1, 3, ("theme", "origin", "destination")),
	"touch":  (2, 2, ("agent", "patient")),
	"think":  (1, 2, ("experiencer", "content")),
	"know":   (1, 2, ("experiencer", "content")),
	"want":   (2, 2, ("experiencer", "content")),
	"feel":   (2, 2, ("experiencer", "state")),
	"see":    (1, 2, ("experiencer", "theme")),
	"hear":   (1, 2, ("experiencer", "theme")),
	"say":    (2, 3, ("agent", "content", "recipient")),
	"be":     (1, 2, ("theme", "place")),
	"live":   (1, 2, ("theme", "place")),
	"die":    (1, 1, ("theme",)),
}

EVALUADORES = {"good", "bad", "big", "small", "hot", "cold", "dark", "mine"}

UNARIOS = {
	"not", "maybe", "can",
	"before", "now", "after",
	"here", "near", "far", "above", "below", "inside",
}

BINARIOS = {"if", "because", "like"}

GRUPO = "G"

OPERADORES = set(PREDICADOS) | EVALUADORES | UNARIOS | BINARIOS | {GRUPO}


def validate(entrada, vocab=None) -> list[str]:
	"""Idéntica a nsm_syntax.validate pero con las tablas EN."""
	if isinstance(entrada, str):
		try:
			tree = parse(entrada)
		except NSMSyntaxError as e:
			return [str(e)]
	else:
		tree = entrada

	errores: list[str] = []
	if isinstance(tree, str):
		return [f"root must be a clause, not atom {tree!r}"]

	def _check(node, ruta: str):
		if isinstance(node, str):
			if vocab is not None and node not in vocab and node not in OPERADORES:
				errores.append(f"{ruta}: atom {node!r} outside vocabulary")
			return
		head = node[0]
		args = node[1:]
		if not isinstance(head, str):
			errores.append(f"{ruta}: head must be an atom, not a clause")
			for i, arg in enumerate(args):
				_check(arg, f"{ruta}.{i}")
			return
		if head == GRUPO:
			if not args:
				errores.append(f"{ruta}: group [G] without core")
			for i, arg in enumerate(args):
				if not isinstance(arg, str):
					errores.append(f"{ruta}.{i}: group members are atoms, not clauses")
				else:
					_check(arg, f"{ruta}.{i}")
			return
		if head in PREDICADOS:
			lo, hi, roles = PREDICADOS[head]
			if not (lo <= len(args) <= hi):
				errores.append(f"{ruta}: {head!r} expects {lo}-{hi} args {roles}, got {len(args)}")
		elif head in EVALUADORES:
			if len(args) != 1:
				errores.append(f"{ruta}: evaluator {head!r} is unary, got {len(args)}")
		elif head in UNARIOS:
			if len(args) != 1:
				errores.append(f"{ruta}: unary {head!r} takes 1, got {len(args)}")
			elif isinstance(args[0], str):
				errores.append(f"{ruta}: {head!r} operates on a clause, not atom {args[0]!r}")
		elif head in BINARIOS:
			if len(args) != 2:
				errores.append(f"{ruta}: binary {head!r} is binary, got {len(args)}")
		else:
			errores.append(f"{ruta}: unknown operator {head!r}")
		for i, arg in enumerate(args):
			_check(arg, f"{ruta}.{i}")

	_check(tree, "raíz")
	return errores
