"""
EXP_034: Vocabulario Vivo — Glifos Ternarios Composicionales.

Cada palabra es un vector de 65 trits {-1, 0, +1} donde cada dimensión
corresponde a un primo semántico de Wierzbicka. El embedding se CONSTRUYE
por composición, no se busca en una tabla opaca.

Referencias:
    - Wierzbicka, A. (1996). Semantics: Primes and Universals.
    - Swadesh, M. (1952). Lexico-statistic dating.
    - Bickerton, D. (1990). Language and Species.
    - Nelson, K. (1973). Structure and strategy in learning to talk.

Origen: Joan Garcia — "cada token debe contener semántica adicional"
"""

import numpy as np
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════
# 1. LOS 65 PRIMOS SEMÁNTICOS (Wierzbicka, adaptados)
# ═══════════════════════════════════════════════════════════════════
# Cada primo es un EJE del espacio semántico. Una palabra es un punto
# en este espacio de 65 dimensiones ternarias.

SEMANTIC_PRIMES = [
	# --- Sustantivos (0-7) ---
	"yo",           # 0  - autoconsciencia
	"tú",           # 1  - otro agente
	"alguien",      # 2  - agente genérico
	"gente",        # 3  - colectivo
	"algo",         # 4  - cosa/entidad
	"cosa",         # 5  - objeto concreto
	"cuerpo",       # 6  - corporalidad
	"parte",        # 7  - componente

	# --- Evaluadores (8-9) ---
	"bueno",        # 8  - valencia positiva
	"malo",         # 9  - valencia negativa

	# --- Descriptores (10-11) ---
	"grande",       # 10 - tamaño alto
	"pequeño",      # 11 - tamaño bajo

	# --- Predicados mentales (12-17) ---
	"pensar",       # 12 - cognición
	"saber",        # 13 - conocimiento
	"querer",       # 14 - deseo/necesidad
	"sentir",       # 15 - emoción/sensación
	"ver",          # 16 - percepción visual
	"oír",          # 17 - percepción auditiva

	# --- Habla (18-20) ---
	"decir",        # 18 - comunicación
	"palabra",      # 19 - unidad lingüística
	"verdad",       # 20 - veracidad

	# --- Acciones (21-24) ---
	"hacer",        # 21 - acción genérica
	"pasar",        # 22 - evento/ocurrencia
	"mover",        # 23 - desplazamiento
	"tocar",        # 24 - contacto físico

	# --- Existencia (25-26) ---
	"existir",      # 25 - ser
	"mío",          # 26 - posesión

	# --- Vida y muerte (27-28) ---
	"vivir",        # 27 - vida
	"morir",        # 28 - muerte/fin

	# --- Tiempo (29-35) ---
	"cuándo",       # 29 - temporal
	"ahora",        # 30 - presente
	"antes",        # 31 - pasado
	"después",      # 32 - futuro
	"mucho_tiempo", # 33 - duración larga
	"poco_tiempo",  # 34 - duración corta
	"momento",      # 35 - instante

	# --- Espacio (36-43) ---
	"dónde",        # 36 - locativo
	"aquí",         # 37 - proximidad
	"arriba",       # 38 - elevación
	"abajo",        # 39 - profundidad
	"lejos",        # 40 - distancia
	"cerca",        # 41 - cercanía
	"lado",         # 42 - lateralidad
	"dentro",       # 43 - interioridad

	# --- Lógica (44-48) ---
	"no",           # 44 - negación
	"quizá",        # 45 - posibilidad
	"poder",        # 46 - capacidad
	"porque",       # 47 - causalidad
	"si",           # 48 - condicionalidad

	# --- Intensificadores (49-50) ---
	"muy",          # 49 - intensidad
	"más",          # 50 - comparación

	# --- Similitud (51) ---
	"como",         # 51 - analogía

	# --- Determinantes (52-55) ---
	"este",         # 52 - demostrativo
	"mismo",        # 53 - identidad
	"otro",         # 54 - alteridad
	"uno",          # 55 - unidad

	# --- Cuantificadores (56-59) ---
	"dos",          # 56 - dualidad
	"algunos",      # 57 - parcialidad
	"todo",         # 58 - totalidad
	"mucho",        # 59 - cantidad

	# --- Extensiones relevantes para supervivencia (60-64) ---
	"caliente",     # 60 - temperatura alta
	"frío",         # 61 - temperatura baja
	"agua_prima",   # 62 - líquido vital (primo, no palabra)
	"luz",          # 63 - luminosidad
	"oscuro",       # 64 - oscuridad
]

N_PRIMES = len(SEMANTIC_PRIMES)  # 65
PRIME_INDEX = {name: i for i, name in enumerate(SEMANTIC_PRIMES)}

# ═══════════════════════════════════════════════════════════════════
# 2. EL VOCABULARIO DE SUPERVIVENCIA (26 palabras, 4 fases)
# ═══════════════════════════════════════════════════════════════════
# Cada palabra es un vector de 65 trits.
# +1 = el primo está presente/aplica positivamente
# -1 = el primo está presente/aplica negativamente
#  0 = el primo no aplica a esta palabra

def _make_glyph(**kwargs) -> np.ndarray:
	"""Crea un glifo de 65 trits a partir de primos nombrados."""
	glyph = np.zeros(N_PRIMES, dtype=np.int8)
	for prime_name, value in kwargs.items():
		if prime_name in PRIME_INDEX:
			glyph[PRIME_INDEX[prime_name]] = int(np.clip(value, -1, 1))
	return glyph


# --- Fase 0: Supervivencia básica (8 palabras, epoch 0-50) ---
VOCABULARY = {
	"agua": _make_glyph(
		algo=1, bueno=1, querer=1, vivir=1, ver=1, mover=1,
		abajo=1, agua_prima=1, frío=1,
	),
	"comida": _make_glyph(
		algo=1, bueno=1, querer=1, vivir=1, tocar=1,
		cuerpo=1,
	),
	"fuego": _make_glyph(
		algo=1, bueno=1, malo=1, ver=1, caliente=1, luz=1,
		morir=1,  # dual: útil Y peligroso
	),
	"sol": _make_glyph(
		algo=1, grande=1, bueno=1, ver=1, arriba=1, lejos=1,
		caliente=1, luz=1, vivir=1,
	),
	"noche": _make_glyph(
		algo=1, malo=-1, grande=1, ver=-1, oscuro=1, frío=1,
		quizá=1,  # incertidumbre
	),
	"cueva": _make_glyph(
		algo=1, bueno=1, grande=1, dentro=1, abajo=1, oscuro=1,
	),
	"yo_palabra": _make_glyph(
		yo=1, existir=1, vivir=1, cuerpo=1, sentir=1,
	),
	"peligro": _make_glyph(
		malo=1, sentir=1, morir=1, no=1, bueno=-1,
		querer=-1,  # no querer
	),

	# --- Fase 1: Acciones (6 palabras, epoch 50-100) ---
	"comer": _make_glyph(
		hacer=1, querer=1, cuerpo=1, tocar=1, bueno=1, vivir=1,
	),
	"beber": _make_glyph(
		hacer=1, querer=1, cuerpo=1, agua_prima=1, vivir=1,
	),
	"mover_accion": _make_glyph(
		hacer=1, mover=1, cuerpo=1, poder=1,
	),
	"ver_accion": _make_glyph(
		hacer=1, ver=1, saber=1, luz=1,
	),
	"dormir": _make_glyph(
		hacer=1, cuerpo=1, vivir=1, ver=-1, oír=-1,
		no=1, mover=-1, dentro=1,
	),
	"dar": _make_glyph(
		hacer=1, tú=1, bueno=1, tocar=1, mío=-1,
	),

	# --- Fase 2: Entorno (6 palabras, epoch 100-150) ---
	"bosque": _make_glyph(
		algo=1, grande=1, vivir=1, ver=1, oscuro=1,
		bueno=1, malo=1, lejos=1,  # recursos + peligro
	),
	"río": _make_glyph(
		algo=1, mover=1, agua_prima=1, ver=1, oír=1,
		grande=1, vivir=1, abajo=1,
	),
	"piedra": _make_glyph(
		algo=1, cosa=1, pequeño=1, tocar=1, hacer=1,
		no=1, vivir=-1, mover=-1,  # inerte
	),
	"árbol": _make_glyph(
		algo=1, grande=1, vivir=1, arriba=1, ver=1,
		no=1, mover=-1,  # fijo
	),
	"tierra": _make_glyph(
		algo=1, grande=1, abajo=1, tocar=1,
		no=1, mover=-1, vivir=-1,
	),
	"lluvia": _make_glyph(
		algo=1, agua_prima=1, arriba=1, abajo=1, mover=1,
		ver=1, oír=1, frío=1, bueno=1, malo=1,
	),

	# --- Fase 3: Amenazas y estados (6 palabras, epoch 150-200) ---
	"depredador": _make_glyph(
		alguien=1, malo=1, grande=1, mover=1, ver=1,
		morir=1, querer=1, cuerpo=1,  # quiere comer
	),
	"tormenta": _make_glyph(
		algo=1, grande=1, malo=1, oír=1, ver=1,
		mover=1, agua_prima=1, morir=1, arriba=1,
	),
	"herida": _make_glyph(
		malo=1, cuerpo=1, sentir=1, morir=1, parte=1,
		ver=1,
	),
	"seguro": _make_glyph(
		bueno=1, vivir=1, dentro=1, no=1, malo=-1,
		sentir=1, querer=1,
	),
	"saciado": _make_glyph(
		bueno=1, cuerpo=1, vivir=1, sentir=1,
		no=1, querer=-1,  # no querer = necesidad cubierta
	),
	"grupo": _make_glyph(
		gente=1, alguien=1, bueno=1, mucho=1,
		decir=1, cerca=1,
	),
}

WORD_NAMES = list(VOCABULARY.keys())
N_WORDS = len(WORD_NAMES)
WORD_INDEX = {name: i for i, name in enumerate(WORD_NAMES)}

# Construir la tabla de glifos: (N_WORDS, N_PRIMES) de trits
GLYPH_TABLE = np.stack([VOCABULARY[w] for w in WORD_NAMES], axis=0)

# Fases del curriculum
CURRICULUM_PHASES = {
	0: WORD_NAMES[:8],    # Fase 0: supervivencia básica
	1: WORD_NAMES[:14],   # Fase 1: + acciones
	2: WORD_NAMES[:20],   # Fase 2: + entorno
	3: WORD_NAMES[:26],   # Fase 3: + amenazas y estados (todo)
}


# ═══════════════════════════════════════════════════════════════════
# 3. EMOCIONES (las mismas 6, como moduladores)
# ═══════════════════════════════════════════════════════════════════

EMOTION_NAMES = ["miedo", "alegría", "ira", "tristeza", "dolor", "hambre"]
N_EMOTIONS = len(EMOTION_NAMES)
EMOTION_INDEX = {name: i for i, name in enumerate(EMOTION_NAMES)}


# ═══════════════════════════════════════════════════════════════════
# 4. REGLAS CAUSALES COHERENTES
# ═══════════════════════════════════════════════════════════════════
# Cada regla: (fuente, destino_neutral)
# El destino cambia según emoción (bifurcaciones)

CAUSAL_RULES = [
	# Necesidades
	("comida", "saciado"),
	("agua", "saciado"),
	("cueva", "seguro"),
	("fuego", "seguro"),     # fuego neutral = seguridad (calor)
	("grupo", "seguro"),

	# Fuentes
	("bosque", "comida"),    # el bosque da comida
	("río", "agua"),         # el río da agua
	("sol", "ver_accion"),   # el sol permite ver
	("lluvia", "agua"),      # la lluvia da agua
	("tierra", "comida"),    # la tierra da comida (recolectar)

	# Ciclos
	("noche", "dormir"),     # la noche trae dormir
	("dormir", "saciado"),   # dormir recupera
	("comer", "saciado"),
	("beber", "saciado"),

	# Peligros
	("depredador", "herida"),
	("tormenta", "peligro"),
	("herida", "peligro"),
]


# ═══════════════════════════════════════════════════════════════════
# 5. BIFURCACIONES EMOCIONALES (coherentes narrativamente)
# ═══════════════════════════════════════════════════════════════════
# (fuente, emoción) → destino_emocional
# Cada bifurcación tiene sentido humano

EMOTIONAL_RULES = {
	# FUEGO: dual por naturaleza
	("fuego", "miedo"):     "herida",      # fuego + miedo = quemadura
	("fuego", "alegría"):   "seguro",      # fuego + alegría = hogar cálido
	("fuego", "hambre"):    "comida",      # fuego + hambre = cocinar
	("fuego", "ira"):       "peligro",     # fuego + ira = incendio
	("fuego", "tristeza"):  "noche",       # fuego + tristeza = hoguera solitaria
	("fuego", "dolor"):     "herida",      # fuego + dolor = quemadura

	# BOSQUE: recurso o amenaza
	("bosque", "miedo"):    "depredador",  # bosque + miedo = hay depredador
	("bosque", "alegría"):  "comida",      # bosque + alegría = recolección exitosa
	("bosque", "hambre"):   "comer",       # bosque + hambre = buscar comida
	("bosque", "ira"):      "depredador",  # bosque + ira = enfrentar amenaza
	("bosque", "tristeza"): "cueva",       # bosque + tristeza = buscar refugio
	("bosque", "dolor"):    "herida",      # bosque + dolor = herido en bosque

	# NOCHE: ciclo natural, significado varía
	("noche", "miedo"):     "peligro",     # noche + miedo = lo desconocido
	("noche", "alegría"):   "fuego",       # noche + alegría = hoguera festiva
	("noche", "hambre"):    "peligro",     # noche + hambre = no hay comida
	("noche", "ira"):       "peligro",     # noche + ira = frustración
	("noche", "tristeza"):  "dormir",      # noche + tristeza = descanso
	("noche", "dolor"):     "cueva",       # noche + dolor = buscar refugio

	# CUEVA: refugio, pero...
	("cueva", "miedo"):     "depredador",  # cueva + miedo = algo dentro
	("cueva", "alegría"):   "seguro",      # cueva + alegría = hogar
	("cueva", "hambre"):    "agua",        # cueva + hambre = buscar agua
	("cueva", "ira"):       "piedra",      # cueva + ira = coger herramienta
	("cueva", "tristeza"):  "dormir",      # cueva + tristeza = descansar
	("cueva", "dolor"):     "seguro",      # cueva + dolor = recuperarse

	# SOL: orientador
	("sol", "miedo"):       "herida",      # sol + miedo = insolación
	("sol", "alegría"):     "seguro",      # sol + alegría = buen día
	("sol", "hambre"):      "bosque",      # sol + hambre = ir a buscar comida
	("sol", "ira"):         "herida",      # sol + ira = calor agresivo
	("sol", "tristeza"):    "noche",       # sol + tristeza = esperar la noche
	("sol", "dolor"):       "cueva",       # sol + dolor = buscar sombra

	# RÍO: fuente vital
	("río", "miedo"):       "tormenta",    # río + miedo = crecida
	("río", "alegría"):     "agua",        # río + alegría = agua fresca
	("río", "hambre"):      "comida",      # río + hambre = pescar
	("río", "ira"):         "tormenta",    # río + ira = corriente violenta
	("río", "tristeza"):    "agua",        # río + tristeza = contemplar
	("río", "dolor"):       "beber",       # río + dolor = beber para aliviar

	# DEPREDADOR: amenaza directa
	("depredador", "miedo"):    "cueva",   # huir al refugio
	("depredador", "alegría"):  "grupo",   # juntos somos fuertes
	("depredador", "hambre"):   "comida",  # el depredador ES comida (cazar)
	("depredador", "ira"):      "piedra",  # coger arma y luchar
	("depredador", "tristeza"): "herida",  # ya nos ha herido
	("depredador", "dolor"):    "peligro", # estamos heridos y en peligro

	# LLUVIA: dual
	("lluvia", "miedo"):    "tormenta",    # lluvia + miedo = tormenta
	("lluvia", "alegría"):  "agua",        # lluvia + alegría = agua fresca
	("lluvia", "hambre"):   "río",         # lluvia + hambre = ir al río
	("lluvia", "ira"):      "tormenta",    # lluvia + ira = tormenta violenta
	("lluvia", "tristeza"): "cueva",       # lluvia + tristeza = refugiarse
	("lluvia", "dolor"):    "cueva",       # lluvia + dolor = buscar cobijo
}


def build_emotional_chains(max_depth: int = 2):
	"""
	Construye cadenas causales emocionales para entrenamiento.
	Cada cadena: (start, emotion_id) → end
	"""
	chains = []

	# Reglas emocionales (bifurcaciones)
	for (src, emo), dst in EMOTIONAL_RULES.items():
		src_idx = WORD_INDEX[src]
		dst_idx = WORD_INDEX[dst]
		emo_idx = EMOTION_INDEX[emo]
		chains.append({
			"start": src_idx,
			"end": dst_idx,
			"emotion_id": emo_idx,
			"fear": 1 if emo == "miedo" else 0,
			"depth": 1,
			"path_names": [src, dst],
			"source_name": src,
			"dest_name": dst,
			"emotion_name": emo,
		})

	# Reglas neutrales (sin emoción específica, para todas las emociones)
	for src, dst in CAUSAL_RULES:
		src_idx = WORD_INDEX[src]
		dst_idx = WORD_INDEX[dst]
		# Solo añadir si no hay bifurcación emocional para esta fuente
		has_bifurcation = any(s == src for (s, _) in EMOTIONAL_RULES)
		if not has_bifurcation:
			for emo_idx in range(N_EMOTIONS):
				chains.append({
					"start": src_idx,
					"end": dst_idx,
					"emotion_id": emo_idx,
					"fear": 0,
					"depth": 1,
					"path_names": [src, dst],
					"source_name": src,
					"dest_name": dst,
					"emotion_name": EMOTION_NAMES[emo_idx],
				})

	return chains


def get_bifurcation_pairs():
	"""
	Retorna pares de bifurcación para evaluación.
	Un par: misma fuente, dos emociones distintas → dos destinos distintos.
	"""
	pairs = []
	sources = set(src for (src, _) in EMOTIONAL_RULES)

	for src in sources:
		emo_dest = {emo: dst for (s, emo), dst in EMOTIONAL_RULES.items() if s == src}
		emos = list(emo_dest.keys())
		for i in range(len(emos)):
			for j in range(i + 1, len(emos)):
				if emo_dest[emos[i]] != emo_dest[emos[j]]:
					pairs.append({
						"source": src,
						"emo_a": emos[i],
						"emo_b": emos[j],
						"dest_a": emo_dest[emos[i]],
						"dest_b": emo_dest[emos[j]],
					})

	return pairs


# ═══════════════════════════════════════════════════════════════════
# 6. GLYPH EMBEDDING MODULE (el corazón composicional)
# ═══════════════════════════════════════════════════════════════════

class GlyphEmbedding(nn.Module):
	"""
	Embedding composicional basado en primos semánticos.

	En vez de una tabla de lookup opaca (vocab_size × embed_dim),
	cada palabra se compone de primos ponderados por trits:

	    embed(word) = sum(trit[i] * prime_embed[i] for i in range(65))

	Los prime_embeddings son APRENDIBLES. El modelo aprende qué
	significa cada eje semántico.
	"""

	def __init__(self, hidden_dim: int = 256, glyph_table: np.ndarray = None):
		super().__init__()
		self.n_primes = N_PRIMES
		self.hidden_dim = hidden_dim

		# Embeddings aprendibles para cada primo (65 × hidden_dim)
		self.prime_embeddings = nn.Parameter(
			torch.randn(N_PRIMES, hidden_dim) * 0.02
		)

		# Tabla de glifos: (vocab_size, 65) trits — NO entrenable
		if glyph_table is None:
			glyph_table = GLYPH_TABLE
		self.register_buffer(
			"glyph_table",
			torch.from_numpy(glyph_table).float()
		)

	@property
	def vocab_size(self):
		return self.glyph_table.shape[0]

	def get_word_embeddings(self) -> torch.Tensor:
		"""
		Computa embeddings de todas las palabras por composición.
		Returns: (vocab_size, hidden_dim)
		"""
		# (vocab_size, 65) @ (65, hidden_dim) → (vocab_size, hidden_dim)
		return self.glyph_table @ self.prime_embeddings

	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
		"""
		Convierte token IDs a embeddings composicionales.

		Args:
		    token_ids: (batch, seq_len) — índices en WORD_NAMES

		Returns:
		    (batch, seq_len, hidden_dim)
		"""
		# Obtener trits para los tokens: (batch, seq, 65)
		trits = self.glyph_table[token_ids]
		# Componer: (batch, seq, 65) @ (65, hidden_dim) → (batch, seq, hidden_dim)
		return trits @ self.prime_embeddings

	def decode_logits(self, hidden: torch.Tensor) -> torch.Tensor:
		"""
		Decodifica hidden states a logits sobre el vocabulario.

		Args:
		    hidden: (batch, seq, hidden_dim)

		Returns:
		    (batch, seq, vocab_size) — logits (similitud coseno escalada)
		"""
		# Embeddings de todas las palabras: (vocab_size, hidden_dim)
		word_embeds = self.get_word_embeddings()

		# Normalizar para coseno
		hidden_norm = torch.nn.functional.normalize(hidden, dim=-1)
		word_norm = torch.nn.functional.normalize(word_embeds, dim=-1)

		# Similitud coseno escalada: (batch, seq, vocab_size)
		logits = torch.matmul(hidden_norm, word_norm.T) * 20.0  # escala para softmax
		return logits


def print_vocabulary_stats():
	"""Imprime estadísticas del vocabulario para debug."""
	print("═══ Vocabulario Vivo ═══")
	print(f"Primos semánticos: {N_PRIMES}")
	print(f"Palabras: {N_WORDS}")
	print(f"Emociones: {N_EMOTIONS}")
	print(f"Reglas causales: {len(CAUSAL_RULES)}")
	print(f"Bifurcaciones emocionales: {len(EMOTIONAL_RULES)}")
	print()

	chains = build_emotional_chains()
	pairs = get_bifurcation_pairs()
	print(f"Cadenas de entrenamiento: {len(chains)}")
	print(f"Pares de bifurcación: {len(pairs)}")
	print()

	# Densidad de trits por palabra
	print("Densidad de trits por palabra:")
	for name in WORD_NAMES:
		glyph = VOCABULARY[name]
		active = np.count_nonzero(glyph)
		pos = np.sum(glyph == 1)
		neg = np.sum(glyph == -1)
		print(f"  {name:<15} {active:>2} trits activos ({pos:>2}+, {neg:>2}-)")

	# Similitud coseno entre glifos
	print("\nPares más similares (coseno de trits):")
	from itertools import combinations
	sims = []
	for i, j in combinations(range(N_WORDS), 2):
		a = GLYPH_TABLE[i].astype(float)
		b = GLYPH_TABLE[j].astype(float)
		norm_a = np.linalg.norm(a)
		norm_b = np.linalg.norm(b)
		if norm_a > 0 and norm_b > 0:
			cos = np.dot(a, b) / (norm_a * norm_b)
			sims.append((cos, WORD_NAMES[i], WORD_NAMES[j]))
	sims.sort(reverse=True)
	for cos, a, b in sims[:10]:
		print(f"  {a:<15} ↔ {b:<15} cos={cos:.3f}")


if __name__ == "__main__":
	print_vocabulary_stats()
