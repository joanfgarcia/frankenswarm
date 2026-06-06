"""
Operadores de Lógica Proposicional para Frankenswarm.

Se registran en el OperatorRegistry existente vía IoC.
Formato: (operando_a, operador_idx, operando_b, resultado)

Niveles:
  - Nivel 2: Contradicción (A == B? → verdad/falsedad)
  - Nivel 3: Modus Ponens  (A implica B, A presente → B)
  - Nivel 4: Cadena Lógica (A→B, B→C ∴ A→C — transitividad)

Los conceptos usan los índices del ReferentialDatasetBreeder:
  0=gato, 1=perro, 2=casa, 3=árbol, 4=agua, 5=fuego,
  6=tierra, 7=aire, 8=sol, 9=luna, 10=peligro, 11=seguridad

El vector de miedo se aplica en el breeder, no en el operador.
"""

from src.bitnet.operators import OperatorRegistry

# ── Reglas de implicación causal (mundo del agente) ──
# Cada regla: (causa_idx, efecto_idx, fear_weight)
# fear_weight: cuánto "duele" equivocarse en esta regla
#
# Diseñadas para crear PROFUNDIDAD — cadenas de 2+ saltos:
#   luna → agua → seguridad → casa
#   fuego → peligro (terminal — peligro no implica nada bueno)
#   fuego → tierra → agua → seguridad (ciclo de regeneración)
#   árbol → aire → sol (naturaleza)
#   sol → aire (calor) — comparte aire con árbol
#
CAUSAL_RULES = [
	# ── Capa 1: Causas primarias ──
	(5, 10, 0.9),  # fuego → peligro
	(5, 6, 0.7),  # fuego → tierra (ceniza)
	# ── Capa 2: Consecuencias ──
	(4, 11, 0.1),  # agua → seguridad
	(2, 11, 0.1),  # casa → seguridad
	(0, 11, 0.1),  # gato → seguridad
	(1, 11, 0.1),  # perro → seguridad
	# ── Capa intermedia: Puentes (crean cadenas) ──
	(9, 4, 0.2),  # luna → agua       (mareas)
	(6, 4, 0.3),  # tierra → agua     (manantiales)
	(11, 2, 0.0),  # seguridad → casa  (refugio)
	(8, 7, 0.3),  # sol → aire        (calor → corrientes)
	(3, 7, 0.2),  # árbol → aire      (oxígeno)
	(7, 8, 0.1),  # aire → sol        (cielo despejado)
	# ── Regla de supervivencia (fear máximo) ──
	(10, 5, 0.95),  # peligro → fuego   (retroalimentación: peligro enciende más fuego)
]

# Cadenas que DEBEN emerger de las reglas anteriores:
# luna → agua → seguridad           (3 saltos con casa: luna→agua→seguridad→casa)
# fuego → tierra → agua             (regeneración)
# fuego → tierra → agua → seguridad (cadena larga — el fuego eventualmente trae seguridad)
# árbol → aire → sol                (naturaleza)
# sol → aire → sol                  (ciclo — interesante para detectar)
# peligro → fuego → peligro         (ciclo vicioso — trampa lógica)
# peligro → fuego → tierra → agua → seguridad (la cadena MÁS larga: 5 saltos)


# Mapeo concepto_idx → nombre para debug
CONCEPT_NAMES = [
	"gato",
	"perro",
	"casa",
	"árbol",
	"agua",
	"fuego",
	"tierra",
	"aire",
	"sol",
	"luna",
	"peligro",
	"seguridad",
]


def get_causal_graph() -> dict[int, list[tuple[int, float]]]:
	"""Construye grafo dirigido de implicaciones: {causa: [(efecto, fear), ...]}"""
	graph = {}
	for causa, efecto, fear in CAUSAL_RULES:
		if causa not in graph:
			graph[causa] = []
		graph[causa].append((efecto, fear))
	return graph


def find_chains(max_depth: int = 2) -> list[tuple[int, int, float]]:
	"""
	Encuentra todas las cadenas transitivas de hasta max_depth saltos.
	Devuelve: [(origen, destino_final, fear_acumulado)]

	Evita ciclos (no visita un nodo dos veces en la misma cadena).
	"""
	graph = get_causal_graph()
	chains = []
	direct_pairs = {(c, e) for c, e, _ in CAUSAL_RULES}

	def dfs(start: int, current: int, depth: int, visited: set, max_fear: float):
		if depth >= 2 and start != current:
			# Solo si NO es una regla directa (eso sería implica, no cadena)
			if (start, current) not in direct_pairs:
				chains.append((start, current, max_fear))

		if depth >= max_depth:
			return

		for next_node, fear in graph.get(current, []):
			if next_node not in visited:
				visited.add(next_node)
				dfs(start, next_node, depth + 1, visited, max(max_fear, fear))
				visited.remove(next_node)

	for start in graph:
		visited = {start}
		for next_node, fear in graph[start]:
			visited.add(next_node)
			dfs(start, next_node, 1, visited, fear)
			visited.remove(next_node)

	# Deduplicar
	seen = set()
	unique = []
	for s, e, f in chains:
		if (s, e) not in seen:
			seen.add((s, e))
			unique.append((s, e, f))

	return unique


@OperatorRegistry.register("implica", 3)
def gen_implica(op_idx: int) -> list[tuple[int, int, int, int]]:
	"""
	Genera ecuaciones de modus ponens: causa IMPLICA efecto.
	Formato: (causa, op_idx, efecto, verdad_idx)
	"""
	eqs = []
	verdad = 0
	falsedad = 1

	valid_effects = {(rule[0], rule[1]) for rule in CAUSAL_RULES}

	for causa, efecto, _fear in CAUSAL_RULES:
		eqs.append((causa, op_idx, efecto, verdad))

		for other_efecto in range(len(CONCEPT_NAMES)):
			if other_efecto != efecto and (causa, other_efecto) not in valid_effects:
				eqs.append((causa, op_idx, other_efecto, falsedad))

	return eqs


@OperatorRegistry.register("contradice", 4)
def gen_contradice(op_idx: int) -> list[tuple[int, int, int, int]]:
	"""
	Genera ecuaciones de detección de contradicción.
	"""
	eqs = []
	verdad = 0
	falsedad = 1

	contradictions = {
		frozenset({5, 4}),  # fuego ↔ agua
		frozenset({10, 11}),  # peligro ↔ seguridad
		frozenset({8, 9}),  # sol ↔ luna
		frozenset({6, 7}),  # tierra ↔ aire
	}

	for a in range(len(CONCEPT_NAMES)):
		for b in range(len(CONCEPT_NAMES)):
			pair = frozenset({a, b})
			if a == b:
				eqs.append((a, op_idx, b, falsedad))
			elif pair in contradictions:
				eqs.append((a, op_idx, b, verdad))
			else:
				eqs.append((a, op_idx, b, falsedad))

	return eqs


@OperatorRegistry.register("cadena", 5)
def gen_cadena(op_idx: int) -> list[tuple[int, int, int, int]]:
	"""
	Genera ecuaciones de INFERENCIA TRANSITIVA (Nivel 4).

	A cadena C = verdad  SI existe un camino A→...→C en el grafo causal
	A cadena C = falsedad SI NO existe tal camino

	Esto es el test de razonamiento: el modelo nunca vio "luna cadena seguridad"
	directamente, pero debe inferirlo de luna→agua y agua→seguridad.
	"""
	eqs = []
	verdad = 0
	falsedad = 1

	# Obtener todas las cadenas transitivas de 2-3 saltos
	valid_chains = find_chains(max_depth=3)
	valid_pairs = {(s, e) for s, e, _ in valid_chains}

	# Positivos: cadenas que existen
	for start, end, _fear in valid_chains:
		eqs.append((start, op_idx, end, verdad))

	# Negativos: pares que NO tienen cadena
	all_starts = {s for s, _, _ in valid_chains}
	for start in all_starts:
		for end in range(len(CONCEPT_NAMES)):
			if (start, end) not in valid_pairs and start != end:
				# Verificar que tampoco es una implicación directa
				direct = {(c, e) for c, e, _ in CAUSAL_RULES}
				if (start, end) not in direct:
					eqs.append((start, op_idx, end, falsedad))

	return eqs


@OperatorRegistry.register("niega", 6)
def gen_niega(op_idx: int) -> list[tuple[int, int, int, int]]:
	"""
	Genera ecuaciones de MODUS TOLLENS (Nivel 5 — razonamiento hacia atrás).

	Formato: (B, niega, A, verdad/falsedad)
	Pregunta: "Si NO hay B, ¿podemos concluir que NO hay A?"

	Respuesta:
	  - verdad  SI existe una regla A→B (directa) o A→...→B (cadena)
	             Porque: A→B, ¬B ∴ ¬A
	  - falsedad SI NO existe tal regla

	Esto es el REVERSO de implica+cadena. El modelo debe razonar
	hacia atrás: ver el efecto ausente y deducir la causa ausente.

	Ejemplo:
	  "NO peligro niega fuego = verdad"
	  Porque: fuego→peligro, si no hay peligro, no hay fuego.

	  "NO agua niega luna = verdad"
	  Porque: luna→agua, si no hay agua, no hay luna.

	  "NO peligro niega luna = falsedad"
	  Porque: luna no implica peligro (ni directa ni transitivamente).
	"""
	eqs = []
	verdad = 0
	falsedad = 1

	# Recopilar TODAS las implicaciones válidas (directas + transitivas)
	direct_pairs = {(c, e) for c, e, _ in CAUSAL_RULES}
	chain_pairs = {(s, e) for s, e, _ in find_chains(max_depth=3)}
	all_implications = direct_pairs | chain_pairs  # A→B existe

	# Para cada par (B, A): ¿existe A→B?
	# Si sí: ¬B → ¬A es verdad (modus tollens)
	tested = set()
	for a, b in all_implications:
		if (b, a) not in tested:
			# (B_ausente, niega, A_candidato, verdad)
			eqs.append((b, op_idx, a, verdad))
			tested.add((b, a))

	# Negativos: pares donde A NO implica B
	{a for a, _ in all_implications}
	all_effects = {b for _, b in all_implications}

	for b in all_effects:
		for a in range(len(CONCEPT_NAMES)):
			if (a, b) not in all_implications and (b, a) not in tested:
				eqs.append((b, op_idx, a, falsedad))
				tested.add((b, a))

	return eqs


# ═══════════════════════════════════════════════════════════════
# EXP_033 — GRAFO CAUSAL EMOCIONAL (Bifurcaciones)
# ═══════════════════════════════════════════════════════════════
#
# La misma causa lleva a destinos distintos según la emoción.
# "fuego + miedo → peligro" vs "fuego + alegría → sol"
#
# Origen teórico: Joan Garcia — "la emoción es la que toma las decisiones"
# ═══════════════════════════════════════════════════════════════

# Conceptos extendidos (3 nuevos para destinos emocionales)
CONCEPT_NAMES_EXT = CONCEPT_NAMES + [
	"calma",      # 12 — destino emocional positivo
	"refugio",    # 13 — destino de protección
	"libertad",   # 14 — destino de expansión
]

# Emociones disponibles
EMOTION_NAMES = ["miedo", "alegría", "ira", "tristeza", "dolor", "hambre"]
EMOTION_IDS = {name: i for i, name in enumerate(EMOTION_NAMES)}
N_EMOTIONS = len(EMOTION_NAMES)

# Reglas emocionales: (fuente_idx, emocion_id, destino_idx, fear_weight)
# Cada par (fuente, emoción) tiene UN destino determinista.
# La emoción decide el camino.
EMOTIONAL_RULES = [
	# ── fuego: peligro o calidez ──
	(5, EMOTION_IDS["miedo"],    10, 0.9),   # fuego + miedo    → peligro
	(5, EMOTION_IDS["alegría"],   8, 0.2),   # fuego + alegría  → sol (calidez)
	(5, EMOTION_IDS["ira"],      10, 0.9),   # fuego + ira      → peligro
	(5, EMOTION_IDS["tristeza"],  6, 0.5),   # fuego + tristeza → tierra (ceniza)
	(5, EMOTION_IDS["dolor"],    10, 0.9),   # fuego + dolor    → peligro
	(5, EMOTION_IDS["hambre"],    2, 0.1),   # fuego + hambre   → casa (cocinar)

	# ── agua: ahogarse o calma ──
	(4, EMOTION_IDS["miedo"],    10, 0.7),   # agua + miedo     → peligro
	(4, EMOTION_IDS["alegría"],  12, 0.1),   # agua + alegría   → calma
	(4, EMOTION_IDS["ira"],       5, 0.6),   # agua + ira       → fuego (vapor)
	(4, EMOTION_IDS["tristeza"],  9, 0.3),   # agua + tristeza  → luna (lágrimas)
	(4, EMOTION_IDS["dolor"],     6, 0.5),   # agua + dolor     → tierra (estancarse)
	(4, EMOTION_IDS["hambre"],   11, 0.1),   # agua + hambre    → seguridad

	# ── luna: oscuridad o paz ──
	(9, EMOTION_IDS["miedo"],    10, 0.7),   # luna + miedo     → peligro
	(9, EMOTION_IDS["alegría"],  12, 0.1),   # luna + alegría   → calma
	(9, EMOTION_IDS["tristeza"],  4, 0.3),   # luna + tristeza  → agua (mareas)
	(9, EMOTION_IDS["hambre"],    2, 0.2),   # luna + hambre    → casa (volver)

	# ── perro: amenaza o compañía ──
	(1, EMOTION_IDS["miedo"],    10, 0.8),   # perro + miedo    → peligro
	(1, EMOTION_IDS["alegría"],   2, 0.1),   # perro + alegría  → casa (compañero)
	(1, EMOTION_IDS["ira"],      10, 0.8),   # perro + ira      → peligro
	(1, EMOTION_IDS["tristeza"], 12, 0.1),   # perro + tristeza → calma (consuelo)

	# ── sol: quemar o libertad ──
	(8, EMOTION_IDS["miedo"],     5, 0.7),   # sol + miedo      → fuego
	(8, EMOTION_IDS["alegría"],  14, 0.1),   # sol + alegría    → libertad
	(8, EMOTION_IDS["ira"],       5, 0.7),   # sol + ira        → fuego
	(8, EMOTION_IDS["hambre"],    6, 0.3),   # sol + hambre     → tierra (cultivar)

	# ── aire: tormenta o vuelo ──
	(7, EMOTION_IDS["miedo"],    10, 0.6),   # aire + miedo     → peligro (tormenta)
	(7, EMOTION_IDS["alegría"],  14, 0.1),   # aire + alegría   → libertad (volar)

	# ── árbol: perderse o refugiarse ──
	(3, EMOTION_IDS["miedo"],    10, 0.5),   # árbol + miedo    → peligro (bosque oscuro)
	(3, EMOTION_IDS["alegría"],  13, 0.1),   # árbol + alegría  → refugio (sombra)

	# ── gato: arañar o calma ──
	(0, EMOTION_IDS["miedo"],    10, 0.4),   # gato + miedo     → peligro
	(0, EMOTION_IDS["alegría"],  12, 0.1),   # gato + alegría   → calma (ronroneo)
]


def get_emotional_causal_graph() -> dict[tuple[int, int], tuple[int, float]]:
	"""
	Grafo emocional: {(fuente, emocion_id): (destino, fear)}.
	Cada par (fuente, emoción) tiene exactamente un destino.
	"""
	graph = {}
	for src, emo, dst, fear in EMOTIONAL_RULES:
		graph[(src, emo)] = (dst, fear)
	return graph


def build_emotional_chains(max_depth: int = 2) -> list[dict]:
	"""
	Construye cadenas emocionales para EXP_033.
	Cada cadena: {start, end, emotion_id, fear, depth, path_names}
	
	Genera cadenas de profundidad 1 (directas) y opcionalmente 2+
	donde el segundo salto usa el grafo neutral.
	"""
	emo_graph = get_emotional_causal_graph()
	neutral_graph = get_causal_graph()
	names = CONCEPT_NAMES_EXT
	chains = []

	# Profundidad 1: directas emocionales
	for (src, emo), (dst, fear) in emo_graph.items():
		chains.append({
			"start": src,
			"end": dst,
			"emotion_id": emo,
			"fear": fear,
			"depth": 1,
			"path_names": [names[src], names[dst]],
		})

	# Profundidad 2: emocional + neutral
	if max_depth >= 2:
		for (src, emo), (mid, fear1) in emo_graph.items():
			for (dst, fear2) in neutral_graph.get(mid, []):
				if dst != src:  # evitar ciclos
					chains.append({
						"start": src,
						"end": dst,
						"emotion_id": emo,
						"intermediate": mid,
						"fear": max(fear1, fear2),
						"depth": 2,
						"path_names": [names[src], names[mid], names[dst]],
					})

	return chains


def get_bifurcation_pairs() -> list[dict]:
	"""
	Devuelve pares de bifurcación para evaluación.
	Cada par: mismo source, 2 emociones distintas → 2 destinos distintos.
	Útil para medir si el modelo realmente usa la emoción para decidir.
	"""
	emo_graph = get_emotional_causal_graph()
	names = CONCEPT_NAMES_EXT

	# Agrupar por source
	by_source: dict[int, list[tuple[int, int, float]]] = {}
	for (src, emo), (dst, fear) in emo_graph.items():
		by_source.setdefault(src, []).append((emo, dst, fear))

	pairs = []
	for src, entries in by_source.items():
		# Buscar pares con destino distinto
		for i, (emo_a, dst_a, _fear_a) in enumerate(entries):
			for emo_b, dst_b, _fear_b in entries[i + 1:]:
				if dst_a != dst_b:
					pairs.append({
						"source": names[src],
						"source_idx": src,
						"emo_a": EMOTION_NAMES[emo_a],
						"emo_b": EMOTION_NAMES[emo_b],
						"dest_a": names[dst_a],
						"dest_b": names[dst_b],
						"dest_a_idx": dst_a,
						"dest_b_idx": dst_b,
					})
	return pairs
