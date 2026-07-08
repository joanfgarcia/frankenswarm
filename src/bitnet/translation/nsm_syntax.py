"""
NSM-C v0 — Sintaxis Canónica de Glifos (RFC-002).

Parser y validador de referencia del lenguaje propio de Bit: linealización
determinista de árboles semánticos sobre los 65 primos de Wierzbicka.
La posición determina el rol, siempre. Este módulo es la fuente de verdad
de la spec: en caso de discrepancia con docs/RFC-002_SINTAXIS_GLIFOS.md,
gana el código.

API:
	parse(texto) -> árbol (str = átomo, list = cláusula/grupo)
	validate(texto_o_árbol, vocab=None) -> lista de errores (vacía = válido)
	is_valid(texto_o_árbol, vocab=None) -> bool
	linearize(árbol) -> str en forma canónica
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════
# Marcos de valencia (RFC-002 §2)
# ═══════════════════════════════════════════════════════════════════

# predicado: (min_args, max_args, roles). Omisión de opcionales solo por la derecha.
PREDICADOS = {
	"hacer":   (1, 3, ("agente", "acción", "paciente")),
	"pasar":   (1, 2, ("tema", "experimentador")),
	"mover":   (1, 3, ("tema", "origen", "destino")),
	"tocar":   (2, 2, ("agente", "paciente")),
	"pensar":  (1, 2, ("experimentador", "contenido")),
	"saber":   (1, 2, ("experimentador", "contenido")),
	"querer":  (2, 2, ("experimentador", "contenido")),
	"sentir":  (2, 2, ("experimentador", "estado")),
	"ver":     (1, 2, ("experimentador", "tema")),
	"oír":     (1, 2, ("experimentador", "tema")),
	"decir":   (2, 3, ("agente", "contenido", "destinatario")),
	"existir": (1, 2, ("tema", "lugar")),
	"vivir":   (1, 2, ("tema", "lugar")),
	"morir":   (1, 1, ("tema",)),
}

# Evaluadores/descriptores como predicados unarios de atribución: "X es Y".
EVALUADORES = {"bueno", "malo", "grande", "pequeño", "caliente", "frío", "oscuro", "mío"}

# Operadores unarios sobre cláusula: lógicos + marcos espacio-temporales.
UNARIOS = {
	"no", "quizá", "poder",
	"antes", "ahora", "después",
	"aquí", "cerca", "lejos", "arriba", "abajo", "dentro",
}

# Conectores binarios, primer-plano-primero (condición/causa/base siempre delante).
BINARIOS = {"si", "porque", "como"}

# Token estructural de sintagma nominal.
GRUPO = "G"

OPERADORES = set(PREDICADOS) | EVALUADORES | UNARIOS | BINARIOS | {GRUPO}


# ═══════════════════════════════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════════════════════════════

class NSMSyntaxError(ValueError):
	"""Error de tokenización o balanceo, con posición de token."""


def tokenize(texto: str) -> list[str]:
	return texto.replace("[", " [ ").replace("]", " ] ").split()


def parse(texto: str):
	"""Parsea una secuencia NSM-C a árbol. Lanza NSMSyntaxError si está malformada."""
	tokens = tokenize(texto)
	if not tokens:
		raise NSMSyntaxError("secuencia vacía")
	pos = 0

	def _parse_expr():
		nonlocal pos
		token = tokens[pos]
		if token == "]":
			raise NSMSyntaxError(f"']' inesperado en token {pos}")
		if token != "[":
			pos += 1
			return token
		pos += 1  # consume '['
		node = []
		while pos < len(tokens) and tokens[pos] != "]":
			node.append(_parse_expr())
		if pos >= len(tokens):
			raise NSMSyntaxError("'[' sin cerrar al final de la secuencia")
		pos += 1  # consume ']'
		if not node:
			raise NSMSyntaxError("cláusula vacía '[]'")
		return node

	tree = _parse_expr()
	if pos != len(tokens):
		raise NSMSyntaxError(f"tokens sobrantes tras la raíz (token {pos}: {tokens[pos]!r})")
	return tree


def linearize(tree) -> str:
	"""Árbol → forma canónica (la que debe sobrevivir al test de ida y vuelta)."""
	if isinstance(tree, str):
		return tree
	return "[" + " ".join(linearize(hijo) for hijo in tree) + "]"


# ═══════════════════════════════════════════════════════════════════
# Validador
# ═══════════════════════════════════════════════════════════════════

def validate(entrada, vocab=None) -> list[str]:
	"""
	Valida una secuencia (str) o un árbol ya parseado.

	Returns: lista de errores; vacía si la secuencia es NSM-C válido.
	vocab: colección opcional de palabras del censo; si se pasa, los átomos
	que no sean operadores deben pertenecer a ella.
	"""
	if isinstance(entrada, str):
		try:
			tree = parse(entrada)
		except NSMSyntaxError as e:
			return [str(e)]
	else:
		tree = entrada

	errores: list[str] = []
	if isinstance(tree, str):
		return [f"la raíz debe ser una cláusula, no el átomo {tree!r}"]

	def _check(node, ruta: str):
		if isinstance(node, str):
			if vocab is not None and node not in vocab and node not in OPERADORES:
				errores.append(f"{ruta}: átomo {node!r} fuera del censo")
			return
		head = node[0]
		args = node[1:]
		if not isinstance(head, str):
			errores.append(f"{ruta}: el operador debe ser un átomo, no una cláusula")
			for i, arg in enumerate(args):
				_check(arg, f"{ruta}.{i}")
			return
		if head == GRUPO:
			if not args:
				errores.append(f"{ruta}: grupo [G] sin núcleo")
			for i, arg in enumerate(args):
				if not isinstance(arg, str):
					errores.append(f"{ruta}.{i}: los miembros de un grupo son átomos, no cláusulas")
				else:
					_check(arg, f"{ruta}.{i}")
			return
		if head in PREDICADOS:
			lo, hi, roles = PREDICADOS[head]
			if not (lo <= len(args) <= hi):
				errores.append(
					f"{ruta}: {head!r} espera {lo}-{hi} argumentos {roles}, recibió {len(args)}"
				)
		elif head in EVALUADORES:
			if len(args) != 1:
				errores.append(f"{ruta}: el evaluador {head!r} es unario, recibió {len(args)}")
		elif head in UNARIOS:
			if len(args) != 1:
				errores.append(f"{ruta}: el operador {head!r} es unario, recibió {len(args)}")
			elif isinstance(args[0], str):
				errores.append(f"{ruta}: {head!r} opera sobre una cláusula, no sobre el átomo {args[0]!r}")
		elif head in BINARIOS:
			if len(args) != 2:
				errores.append(f"{ruta}: el conector {head!r} es binario, recibió {len(args)}")
		else:
			errores.append(f"{ruta}: operador desconocido {head!r}")
		for i, arg in enumerate(args):
			_check(arg, f"{ruta}.{i}")

	_check(tree, "raíz")
	return errores


def is_valid(entrada, vocab=None) -> bool:
	return not validate(entrada, vocab=vocab)
