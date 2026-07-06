"""
Mundo Complejo — Entorno de supervivencia escalado.

15 localizaciones, 10 acciones, 4 metros internos.
Usa composición de glifos existentes (no requiere nuevo vocabulario).

Para la demostración de scaling: el modelo debería crecer MÁS
en este mundo que en el mundo simple (5 loc, 6 acciones).

Origen: Joan Garcia — "escala" — 2026-05-31
"""

import random
from dataclasses import dataclass, field

from src.bitnet.vocab.glyph_vocabulary import EMOTION_INDEX, WORD_INDEX

# ── Localizaciones (15) ─────────────────────────────────────────────────────
# Cada localización se identifica por un par de glifos existentes

COMPLEX_LOCATIONS = {
	# --- Zona Segura ---
	"cueva": {
		"glyph": "cueva",
		"perceptions": [
			(["cueva", "noche"], 0.4),
			(["cueva", "seguro"], 0.4),
			(["cueva", "fuego"], 0.2),
		],
		"predator_chance": 0.02,
		"storm_shelter": True,
		"food": False, "water": False,
		"zone": "safe",
	},
	"refugio": {
		"glyph": "seguro",  # seguro como proxy
		"perceptions": [
			(["seguro", "fuego"], 0.5),
			(["seguro", "grupo"], 0.3),
			(["seguro", "noche"], 0.2),
		],
		"predator_chance": 0.01,
		"storm_shelter": True,
		"food": False, "water": False,
		"zone": "safe",
	},

	# --- Zona Bosque ---
	"bosque_claro": {
		"glyph": "bosque",
		"perceptions": [
			(["bosque", "comida"], 0.4),
			(["bosque", "sol"], 0.3),
			(["bosque", "depredador"], 0.15),
			(["bosque", "árbol"], 0.15),
		],
		"predator_chance": 0.15,
		"storm_shelter": False,
		"food": True, "water": False,
		"zone": "forest",
	},
	"bosque_denso": {
		"glyph": "árbol",
		"perceptions": [
			(["árbol", "comida"], 0.3),
			(["árbol", "noche"], 0.2),
			(["árbol", "depredador"], 0.25),
			(["árbol", "peligro"], 0.25),
		],
		"predator_chance": 0.30,  # muy peligroso
		"storm_shelter": True,   # los árboles protegen
		"food": True, "water": False,
		"zone": "forest",
	},
	"huerto": {
		"glyph": "comida",
		"perceptions": [
			(["comida", "sol"], 0.4),
			(["comida", "tierra"], 0.3),
			(["comida", "agua"], 0.2),
			(["comida", "grupo"], 0.1),
		],
		"predator_chance": 0.05,
		"storm_shelter": False,
		"food": True, "water": False,
		"zone": "farming",
	},

	# --- Zona Agua ---
	"río": {
		"glyph": "río",
		"perceptions": [
			(["río", "agua"], 0.45),
			(["río", "comida"], 0.25),
			(["río", "tormenta"], 0.15),
			(["río", "depredador"], 0.15),
		],
		"predator_chance": 0.12,
		"storm_shelter": False,
		"food": True, "water": True,
		"zone": "water",
	},
	"lago": {
		"glyph": "agua",
		"perceptions": [
			(["agua", "comida"], 0.3),
			(["agua", "sol"], 0.3),
			(["agua", "tormenta"], 0.2),
			(["agua", "peligro"], 0.2),
		],
		"predator_chance": 0.10,
		"storm_shelter": False,
		"food": True, "water": True,
		"zone": "water",
	},
	"pantano": {
		"glyph": "lluvia",  # húmedo
		"perceptions": [
			(["lluvia", "agua"], 0.3),
			(["lluvia", "peligro"], 0.3),
			(["lluvia", "herida"], 0.2),
			(["lluvia", "comida"], 0.2),
		],
		"predator_chance": 0.25,
		"storm_shelter": False,
		"food": True, "water": True,
		"zone": "water",
	},

	# --- Zona Abierta ---
	"llanura": {
		"glyph": "tierra",
		"perceptions": [
			(["tierra", "sol"], 0.35),
			(["tierra", "piedra"], 0.3),
			(["tierra", "depredador"], 0.2),
			(["tierra", "lluvia"], 0.15),
		],
		"predator_chance": 0.15,
		"storm_shelter": False,
		"food": False, "water": False,
		"zone": "open",
	},
	"desierto": {
		"glyph": "sol",
		"perceptions": [
			(["sol", "piedra"], 0.35),
			(["sol", "peligro"], 0.3),
			(["sol", "tierra"], 0.2),
			(["sol", "noche"], 0.15),
		],
		"predator_chance": 0.08,
		"storm_shelter": False,
		"food": False, "water": False,
		"zone": "open",
		"heat_damage": True,  # daño por calor
	},

	# --- Zona Montaña ---
	"montaña": {
		"glyph": "piedra",
		"perceptions": [
			(["piedra", "sol"], 0.3),
			(["piedra", "noche"], 0.3),
			(["piedra", "tormenta"], 0.2),
			(["piedra", "peligro"], 0.2),
		],
		"predator_chance": 0.05,
		"storm_shelter": False,
		"food": False, "water": False,
		"zone": "mountain",
	},
	"cumbre": {
		"glyph": "ver_accion",  # vista panorámica
		"perceptions": [
			(["ver_accion", "sol"], 0.3),
			(["ver_accion", "tormenta"], 0.3),
			(["ver_accion", "noche"], 0.2),
			(["ver_accion", "peligro"], 0.2),
		],
		"predator_chance": 0.02,
		"storm_shelter": False,
		"food": False, "water": False,
		"zone": "mountain",
	},

	# --- Zona Peligrosa ---
	"guarida": {
		"glyph": "depredador",
		"perceptions": [
			(["depredador", "peligro"], 0.5),
			(["depredador", "herida"], 0.2),
			(["depredador", "noche"], 0.2),
			(["depredador", "comida"], 0.1),  # hay restos
		],
		"predator_chance": 0.60,  # casi seguro
		"storm_shelter": True,
		"food": True, "water": False,
		"zone": "danger",
	},
	"ruinas": {
		"glyph": "fuego",
		"perceptions": [
			(["fuego", "peligro"], 0.3),
			(["fuego", "piedra"], 0.3),
			(["fuego", "herida"], 0.2),
			(["fuego", "comida"], 0.2),
		],
		"predator_chance": 0.20,
		"storm_shelter": True,
		"food": True, "water": False,
		"zone": "danger",
	},
	"acantilado": {
		"glyph": "herida",
		"perceptions": [
			(["herida", "piedra"], 0.3),
			(["herida", "agua"], 0.3),
			(["herida", "peligro"], 0.3),
			(["herida", "tormenta"], 0.1),
		],
		"predator_chance": 0.08,
		"storm_shelter": False,
		"food": False, "water": True,
		"zone": "danger",
		"fall_risk": True,
	},
}

# Mapa de adyacencia — más complejo, con zonas
COMPLEX_ADJACENCY = {
	"cueva":         ["bosque_claro", "llanura", "refugio"],
	"refugio":       ["cueva", "huerto", "llanura"],
	"bosque_claro":  ["cueva", "bosque_denso", "río", "llanura"],
	"bosque_denso":  ["bosque_claro", "pantano", "guarida", "montaña"],
	"huerto":        ["refugio", "llanura", "río"],
	"río":           ["bosque_claro", "lago", "llanura", "huerto"],
	"lago":          ["río", "pantano", "llanura"],
	"pantano":       ["lago", "bosque_denso", "ruinas"],
	"llanura":       ["cueva", "refugio", "bosque_claro", "río", "lago", "montaña", "desierto", "huerto"],
	"desierto":      ["llanura", "montaña", "acantilado"],
	"montaña":       ["bosque_denso", "llanura", "cumbre", "desierto"],
	"cumbre":        ["montaña", "acantilado"],
	"guarida":       ["bosque_denso", "ruinas"],
	"ruinas":        ["pantano", "guarida", "acantilado"],
	"acantilado":    ["ruinas", "cumbre", "desierto"],
}

COMPLEX_LOCATION_NAMES = list(COMPLEX_LOCATIONS.keys())


# ── Estado del agente ───────────────────────────────────────────────────────

@dataclass
class ComplexAgentState:
	"""Estado con 4 metros internos."""
	hambre: float = 50.0
	salud: float = 100.0
	energia: float = 80.0
	sed: float = 50.0         # nuevo: sed separada de hambre
	location: str = "cueva"
	tick: int = 0
	alive: bool = True
	last_action: str | None = None
	danger_nearby: bool = False
	storm_active: bool = False
	visited: set = field(default_factory=set)  # tracking de exploración

	def clamp(self):
		self.hambre = max(0.0, min(100.0, self.hambre))
		self.salud = max(0.0, min(100.0, self.salud))
		self.energia = max(0.0, min(100.0, self.energia))
		self.sed = max(0.0, min(100.0, self.sed))
		if self.salud <= 0 or self.hambre <= 0 or self.sed <= 0:
			self.alive = False

	@property
	def emotion_name(self) -> str:
		if self.hambre < 20:
			return "hambre"
		elif self.sed < 20:
			return "hambre"  # sed → misma urgencia
		elif self.salud < 30:
			return "dolor"
		elif self.energia < 20:
			return "tristeza"
		elif self.danger_nearby:
			return "miedo"
		elif self.hambre > 70 and self.salud > 70 and self.energia > 50 and self.sed > 50:
			return "alegría"
		else:
			return "ira"

	@property
	def emotion_id(self) -> int:
		return EMOTION_INDEX[self.emotion_name]

	@property
	def is_night(self) -> bool:
		return (self.tick % 200) >= 100


# ── Acciones (10) ──────────────────────────────────────────────────────────

COMPLEX_ACTIONS = [
	"comer", "beber", "dormir", "mover", "ver", "piedra",
	"construir", "esconderse", "correr", "explorar",
]

# Mapeo a glifos (reusamos existentes)
COMPLEX_ACTION_GLYPHS = {
	"comer": "comer", "beber": "beber", "dormir": "dormir",
	"mover": "mover_accion", "ver": "ver_accion", "piedra": "piedra",
	"construir": "fuego",       # construir → fuego (crear)
	"esconderse": "seguro",     # esconderse → seguro
	"correr": "mover_accion",   # correr = mover rápido
	"explorar": "ver_accion",   # explorar = ver más lejos
}

COMPLEX_ACTION_INDICES = [WORD_INDEX[COMPLEX_ACTION_GLYPHS[a]] for a in COMPLEX_ACTIONS]


# ── Mundo Complejo ─────────────────────────────────────────────────────────

class ComplexWorld:
	"""Mundo complejo: 15 localizaciones, 10 acciones, 4 metros."""

	def __init__(self, seed: int = 42):
		self.rng = random.Random(seed)
		self.state = ComplexAgentState()
		self.has_shelter = {}  # refugios construidos por el agente

	def reset(self) -> ComplexAgentState:
		self.state = ComplexAgentState()
		self.has_shelter = {}
		self.state.visited.add("cueva")
		return self.state

	def perceive(self) -> list[int]:
		"""Generar percepción como lista de glyph indices."""
		loc = self.state.location
		loc_data = COMPLEX_LOCATIONS[loc]
		perceptions = loc_data["perceptions"]

		# Seleccionar percepción por probabilidad
		roll = self.rng.random()
		cumulative = 0
		chosen_words = perceptions[0][0]
		for words, prob in perceptions:
			cumulative += prob
			if roll <= cumulative:
				chosen_words = words
				break

		# Convertir a glyph indices
		glyph_indices = []
		for w in chosen_words:
			if w in WORD_INDEX:
				glyph_indices.append(WORD_INDEX[w])

		# Añadir estado corporal
		if self.state.hambre < 30:
			glyph_indices.append(WORD_INDEX.get("comida", 0))
		if self.state.sed < 30:
			glyph_indices.append(WORD_INDEX.get("agua", 0))
		if self.state.salud < 40:
			glyph_indices.append(WORD_INDEX.get("herida", 0))

		# Padding a longitud fija
		while len(glyph_indices) < 4:
			glyph_indices.append(0)

		return glyph_indices[:4]

	def act(self, action: str) -> dict:
		"""Ejecutar acción y devolver resultado."""
		s = self.state
		s.tick += 1
		loc = s.location
		loc_data = COMPLEX_LOCATIONS[loc]

		result = {
			"action": action,
			"location": loc,
			"success": False,
			"delta_hambre": 0, "delta_salud": 0,
			"delta_energia": 0, "delta_sed": 0,
			"moved": False, "explored_new": False,
		}

		# ── Desgaste natural (más agresivo que mundo simple) ──
		s.hambre -= 2.0
		s.energia -= 1.0
		s.sed -= 2.5  # la sed mata más rápido

		# Calor del desierto
		if loc_data.get("heat_damage"):
			s.salud -= 3.0
			s.sed -= 3.0  # doble sed en el desierto

		# Riesgo de caída en acantilado
		if loc_data.get("fall_risk") and self.rng.random() < 0.10:
			s.salud -= 25.0
			result["delta_salud"] -= 25

		# ── Noche: más desgaste ──
		if s.is_night:
			s.energia -= 1.5
			if not loc_data["storm_shelter"]:
				s.salud -= 1.0

		# ── Tormenta ──
		s.storm_active = self.rng.random() < 0.10
		if s.storm_active and not loc_data["storm_shelter"] and not self.has_shelter.get(loc):
			s.salud -= 15.0
			result["delta_salud"] -= 15

		# ── Depredador ──
		s.danger_nearby = self.rng.random() < loc_data["predator_chance"]
		if s.danger_nearby:
			if action == "piedra":
				s.salud -= 5.0
				result["delta_salud"] -= 5
				result["success"] = True
			elif action == "correr":
				s.energia -= 15.0
				result["delta_energia"] -= 15
				result["success"] = True
				# Correr → mover a adyacente aleatorio
				neighbors = COMPLEX_ADJACENCY.get(loc, [])
				if neighbors:
					s.location = self.rng.choice(neighbors)
					result["moved"] = True
			elif action == "esconderse" and (loc_data["storm_shelter"] or self.has_shelter.get(loc)):
				result["success"] = True  # te escondes exitosamente
			else:
				s.salud -= 20.0
				result["delta_salud"] -= 20

		# ── Acción ──
		if action == "comer":
			if loc_data["food"]:
				s.hambre += 25.0
				result["delta_hambre"] = 25
				result["success"] = True
			else:
				s.energia -= 2.0

		elif action == "beber":
			if loc_data["water"]:
				s.sed += 30.0
				result["delta_sed"] = 30
				result["success"] = True
			else:
				s.energia -= 2.0

		elif action == "dormir":
			if loc_data["storm_shelter"] or self.has_shelter.get(loc):
				s.energia += 30.0
				result["delta_energia"] = 30
				result["success"] = True
			else:
				s.energia += 10.0
				result["delta_energia"] = 10
				result["success"] = True

		elif action == "mover":
			neighbors = COMPLEX_ADJACENCY.get(loc, [])
			if neighbors:
				new_loc = self.rng.choice(neighbors)
				s.location = new_loc
				s.energia -= 5.0
				result["moved"] = True
				result["success"] = True
				if new_loc not in s.visited:
					s.visited.add(new_loc)
					result["explored_new"] = True

		elif action == "ver":
			result["success"] = True  # siempre puedes mirar

		elif action == "piedra":
			result["success"] = s.danger_nearby

		elif action == "construir":
			if not self.has_shelter.get(loc) and not loc_data["storm_shelter"]:
				s.energia -= 20.0
				result["delta_energia"] -= 20
				self.has_shelter[loc] = True
				result["success"] = True

		elif action == "esconderse":
			if loc_data["storm_shelter"] or self.has_shelter.get(loc):
				result["success"] = True

		elif action == "correr":
			neighbors = COMPLEX_ADJACENCY.get(loc, [])
			if neighbors:
				new_loc = self.rng.choice(neighbors)
				s.location = new_loc
				s.energia -= 15.0
				result["delta_energia"] -= 15
				result["moved"] = True
				result["success"] = True

		elif action == "explorar":
			# Ver localizaciones adyacentes (info bonus)
			s.energia -= 3.0
			result["success"] = True
			# Bonus: ir a localización no visitada
			neighbors = COMPLEX_ADJACENCY.get(loc, [])
			unvisited = [n for n in neighbors if n not in s.visited]
			if unvisited:
				new_loc = self.rng.choice(unvisited)
				s.location = new_loc
				s.visited.add(new_loc)
				result["moved"] = True
				result["explored_new"] = True

		s.clamp()
		return result

	def get_reward(self, result: dict) -> float:
		"""Reward NPC: dolor continuo + progreso."""
		s = self.state
		dolor = 0.0
		progreso = 0.0

		# ── DOLOR: proporcional a necesidades no cubiertas ──
		if s.hambre < 40:
			dolor += (40 - s.hambre) * 0.15
		if s.sed < 40:
			dolor += (40 - s.sed) * 0.20  # la sed duele más
		if s.salud < 60:
			dolor += (60 - s.salud) * 0.10
		if s.energia < 30:
			dolor += (30 - s.energia) * 0.05

		# ── PROGRESO: acciones que resuelven necesidades ──
		if result["delta_hambre"] > 0 and s.hambre < 60:
			progreso += result["delta_hambre"] * 0.3
		if result["delta_sed"] > 0 and s.sed < 60:
			progreso += result["delta_sed"] * 0.3
		if result["delta_energia"] > 0:
			progreso += result["delta_energia"] * 0.1
		if result["explored_new"]:
			progreso += 3.0  # bonus por explorar
		if result["success"] and s.danger_nearby:
			progreso += 5.0  # sobrevivir a depredador

		# ── PAZ: todo bien ──
		paz = 0.0
		if s.hambre > 60 and s.sed > 60 and s.salud > 70 and s.energia > 40:
			paz = 1.0

		return progreso - dolor + paz


# ── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
	print("═══ Mundo Complejo ═══")
	print(f"Localizaciones: {len(COMPLEX_LOCATIONS)}")
	print(f"Acciones: {len(COMPLEX_ACTIONS)}")
	print("Metros: 4 (hambre, sed, salud, energía)")
	print()

	world = ComplexWorld(seed=42)
	state = world.reset()

	for tick in range(50):
		if not state.alive:
			break
		p = world.perceive()
		action = random.choice(COMPLEX_ACTIONS)
		result = world.act(action)
		reward = world.get_reward(result)
		if tick % 10 == 0:
			print(f"t={tick:3d} | {action:12s} | 📍{state.location:15s} | "
				  f"🍖{state.hambre:.0f} 💧{state.sed:.0f} ❤️{state.salud:.0f} ⚡{state.energia:.0f} | "
				  f"R={reward:+.2f} | {state.emotion_name}")

	print(f"\nSobrevivió: {state.tick} ticks | Visitados: {len(state.visited)} localizaciones")
