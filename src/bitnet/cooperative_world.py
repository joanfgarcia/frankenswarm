"""
Cooperative World — La Arena.

Un mundo hostil donde la supervivencia en solitario es casi imposible.
La comida es escasa, el metabolismo es rápido, y el fog of war oculta
la mayor parte del mapa. La cooperación y comunicación entre agentes
emerge de la necesidad de sobrevivir.

Diferencias con MinimalWorld (el Dojo):
- 9 localizaciones (más grande, más niebla)
- Fog of war: cada agente solo ve su localización actual
- Comida escasa: aparece en 1 lugar cada 8-12 ticks, dura 5 ticks
- Metabolismo rápido: -3.5 hambre/tick (balanceado)
- GRITAR como acción: broadcast [mi_loc, lo_que_veo]
- Señal con memoria a corto plazo: persiste con decaimiento (Baddeley/FSRS)
- Reward cooperativo modulado por retrievability (Schultz TD-learning)
- Modulación emocional de la saliencia de la señal (Damasio)

Origen: Joan Garcia — "los mudos mueren" — 2026-06-01
"""

import os
import random
from collections import deque
from dataclasses import dataclass

from pyswip import Prolog

from src.bitnet.glyph_vocabulary import EMOTION_INDEX, WORD_INDEX

# ── SWI-Prolog Engine Initialization ────────────────────────────────────────
prolog = Prolog()
_rules_path = os.path.join(os.path.dirname(__file__), "cooperative_rules.pl")
_rules_path = _rules_path.replace(os.sep, "/")
prolog.consult(_rules_path)


def query_prolog(query_str: str) -> list[dict]:
	"""Realiza una consulta a Prolog y decodifica las cadenas devueltas como bytes."""
	results = list(prolog.query(query_str))
	for res in results:
		for k, v in res.items():
			if isinstance(v, bytes):
				res[k] = v.decode("utf-8")
	return results


def format_skills(skills: list[str]) -> str:
	"""Formatea una lista de habilidades para Prolog."""
	return "[" + ", ".join(f"'{s}'" for s in skills) + "]"


# ── Memoria a corto plazo (Baddeley/FSRS) ──────────────────────────────────
# Constante de decaimiento: R(t) = DECAY_BASE ^ t
# Con DECAY_BASE=0.88, la retrievability a los 5 ticks es ~0.53, a los 10 es ~0.28
# Esto modela la curva del olvido de la memoria de trabajo (~20-30 seg en humanos)
SIGNAL_DECAY_BASE = 0.88


def signal_retrievability(age: int) -> float:
	"""Curva del olvido para señales de corto plazo (Baddeley/Ebbinghaus)."""
	if age <= 0:
		return 1.0
	return SIGNAL_DECAY_BASE ** age


COOP_LOCATIONS = {
	"cueva": {
		"description": "refugio seguro, sin comida",
		"predator_chance": 0.02,
		"storm_shelter": True,
		"food_eligible": False,
		"water_available": False,
		"branches_eligible": False,
		"stones_eligible": False,
		"fish_eligible": False,
	},
	"lago": {
		"description": "agua abundante, tranquilo",
		"predator_chance": 0.08,
		"storm_shelter": False,
		"food_eligible": False,
		"water_available": True,
		"branches_eligible": False,
		"stones_eligible": False,
		"fish_eligible": True,
	},
	"pradera": {
		"description": "centro del mapa, expuesto",
		"predator_chance": 0.15,
		"storm_shelter": False,
		"food_eligible": False,
		"water_available": False,
		"branches_eligible": False,
		"stones_eligible": False,
		"fish_eligible": False,
	},
	"bosque": {
		"description": "denso, comida posible, peligroso",
		"predator_chance": 0.20,
		"storm_shelter": False,
		"food_eligible": True,
		"water_available": False,
		"branches_eligible": True,
		"stones_eligible": False,
		"fish_eligible": False,
	},
	"montaña": {
		"description": "alto, frío, vista lejana",
		"predator_chance": 0.05,
		"storm_shelter": False,
		"food_eligible": False,
		"water_available": False,
		"branches_eligible": False,
		"stones_eligible": True,
		"fish_eligible": False,
	},
	"valle": {
		"description": "protegido, fértil",
		"predator_chance": 0.10,
		"storm_shelter": False,
		"food_eligible": True,
		"water_available": False,
		"branches_eligible": True,
		"stones_eligible": False,
		"fish_eligible": False,
	},
	"río": {
		"description": "agua y peces, inundaciones",
		"predator_chance": 0.12,
		"storm_shelter": False,
		"food_eligible": True,
		"water_available": True,
		"branches_eligible": True,
		"stones_eligible": False,
		"fish_eligible": True,
	},
	"pantano": {
		"description": "húmedo, peligroso, pero rico",
		"predator_chance": 0.25,
		"storm_shelter": False,
		"food_eligible": True,
		"water_available": True,
		"branches_eligible": False,
		"stones_eligible": False,
		"fish_eligible": True,
	},
	"ruinas": {
		"description": "antiguo refugio, depredadores",
		"predator_chance": 0.30,
		"storm_shelter": True,
		"food_eligible": True,
		"water_available": False,
		"branches_eligible": False,
		"stones_eligible": True,
		"fish_eligible": False,
	},
}

# Grafo de adyacencia: max distance = 3 entre cualquier par
COOP_ADJACENCY = {
	"cueva":    ["lago", "valle", "pradera"],
	"lago":     ["cueva", "pradera", "montaña"],
	"pradera":  ["cueva", "lago", "bosque", "valle", "montaña"],
	"bosque":   ["pradera", "río", "ruinas"],
	"montaña":  ["lago", "pradera", "pantano"],
	"valle":    ["cueva", "pradera", "río"],
	"río":      ["valle", "bosque", "pantano"],
	"pantano":  ["montaña", "río", "ruinas"],
	"ruinas":   ["bosque", "pantano"],
}

# Glifos para localizaciones (usando vocabulario existente cuando se puede)
COOP_LOCATION_GLYPHS = {
	"cueva": "cueva",
	"lago": "agua",       # proxy
	"pradera": "tierra",  # proxy
	"bosque": "bosque",
	"montaña": "piedra",  # proxy
	"valle": "tierra",    # proxy (diferente percepción contextual)
	"río": "río",
	"pantano": "agua",    # proxy
	"ruinas": "cueva",    # proxy
}

# Glifos de percepción para lo que hay en cada sitio
PERCEPTION_GLYPHS = {
	"comida": "comida",
	"agua": "agua",
	"peligro": "depredador",
	"nada": "noche",       # proxy for "nothing interesting"
	"seguro": "seguro",
	"tormenta": "tormenta",
}

COOP_LOCATION_NAMES = list(COOP_LOCATIONS.keys())
FOOD_ELIGIBLE = [loc for loc, data in COOP_LOCATIONS.items() if data["food_eligible"]]

# Reverse lookup: glyph_index → location name (para navegar hacia señales)
_GLYPH_TO_LOCATIONS: dict[int, list[str]] = {}
for _loc, _glyph_name in COOP_LOCATION_GLYPHS.items():
	_idx = WORD_INDEX.get(_glyph_name, -1)
	if _idx >= 0:
		_GLYPH_TO_LOCATIONS.setdefault(_idx, []).append(_loc)


def _bfs_next_step(start: str, target: str) -> str | None:
	"""BFS en el grafo de adyacencia. Devuelve el siguiente paso hacia target."""
	if start == target:
		return None  # ya estás ahí
	visited = {start}
	queue = deque()
	# (nodo_actual, primer_paso): guardamos el primer movimiento
	for neighbor in COOP_ADJACENCY[start]:
		queue.append((neighbor, neighbor))
		visited.add(neighbor)
	while queue:
		current, first_step = queue.popleft()
		if current == target:
			return first_step
		for neighbor in COOP_ADJACENCY[current]:
			if neighbor not in visited:
				visited.add(neighbor)
				queue.append((neighbor, first_step))
	return None  # no path (shouldn't happen in connected graph)


def _bfs_distance(start: str, target: str) -> int:
	"""Devuelve la distancia en el grafo (número de pasos BFS)."""
	if start == target:
		return 0
	visited = {start}
	queue = deque([(start, 0)])
	while queue:
		current, dist = queue.popleft()
		if current == target:
			return dist
		for neighbor in COOP_ADJACENCY[current]:
			if neighbor not in visited:
				visited.add(neighbor)
				queue.append((neighbor, dist + 1))
	return 999


# ── Acciones (14) ───────────────────────────────────────────────────────────

COOP_ACTIONS = ["comer", "beber", "dormir", "mover", "ver", "luchar", "gritar", "dar", "enseñar", "aprender", "reproducir", "fabricar", "construir", "encender", "reanimar"]
COOP_N_ACTIONS = len(COOP_ACTIONS)

SILENCE_GLYPH = WORD_INDEX.get("noche", 0)  # "silence" proxy


# ── Estado del agente ───────────────────────────────────────────────────────

@dataclass
class CoopAgentState:
	"""Estado interno de un agente en la arena."""
	hambre: float = 60.0       # 0 = muerto de hambre, 100 = saciado
	sed: float = 80.0          # 0 = deshidratado, 100 = hidratado
	salud: float = 100.0       # 0 = muerto
	energia: float = 80.0      # 0 = exhausto
	location: str = "cueva"    # localización actual
	tick: int = 0
	alive: bool = True
	last_action: str | None = None
	danger_nearby: bool = False
	storm_active: bool = False

	mochila_comida: int = 0    # 0 o 1
	mochila_agua: int = 0      # 0 o 1
	mochila_ramas: int = 0     # máx 3
	mochila_piedras: int = 0   # máx 3
	tiene_lanza: bool = False

	# Comunicación recibida (memoria a corto plazo)
	received_signal: tuple | None = None   # (loc_glyph_idx, what_glyph_idx)
	signal_age: int = 0                       # ticks desde que recibí la señal

	# Navegación: destino persistente (se mantiene hasta llegar)
	nav_target: str | None = None          # localización objetivo

	# Quién me envió la señal (para atribuir crédito cooperativo)
	signal_from: str | None = None         # "a" or "b" or "c"

	# Identificador del agente
	agent_id: str = "a"

	# Teoría de la mente y conocimiento del mapa local (Fase 3)
	map_knowledge: dict = None
	companion_model: dict = None
	previous_location: str | None = None

	# Habilidades y destilación (Fase 6)
	network_width: int = 128
	pending_growth_penalty: bool = False
	learned_skills: list = None
	teaching_buffer: list = None
	skill_experience: dict = None
	aburrimiento: float = 0.0
	debuff_tristeza_ticks: int = 0
	debuff_inconsciente_ticks: int = 0
	just_fell_unconscious: bool = False
	shout_cooldown_ticks: int = 0

	def __post_init__(self):
		if self.map_knowledge is None:
			self.map_knowledge = {
				loc: {
					"food": 5.0 if (COOP_LOCATIONS[loc]["food_eligible"] or COOP_LOCATIONS[loc].get("fish_eligible")) else 0.0,
					"water": 5.0 if COOP_LOCATIONS[loc]["water_available"] else 0.0,
					"last_updated": 0
				}
				for loc in COOP_LOCATION_NAMES
			}
		if self.companion_model is None:
			self.companion_model = {}
		if self.learned_skills is None:
			if self.agent_id == "a":
				self.learned_skills = ["comida", "agua"]
			elif self.agent_id == "b":
				self.learned_skills = ["comida", "caza"]
			elif self.agent_id == "c":
				self.learned_skills = ["agua", "caza"]
			else:
				self.learned_skills = []
		if self.teaching_buffer is None:
			self.teaching_buffer = []
		if self.skill_experience is None:
			self.skill_experience = {}

	@property
	def max_slots(self) -> int:
		if self.network_width < 192:
			return 3
		elif self.network_width < 288:
			return 4
		else:
			return 5

	@property
	def signal_retrievability(self) -> float:
		"""¿Cuánto recuerdo de la señal? (Baddeley/FSRS decay)"""
		if not self.received_signal:
			return 0.0
		return signal_retrievability(self.signal_age)

	@property
	def emotion_name(self) -> str:
		if self.debuff_tristeza_ticks > 0:
			return "tristeza"
		elif self.hambre < 25:
			return "hambre"
		elif self.sed < 25 or self.salud < 35:
			return "dolor"
		elif self.energia < 25:
			return "tristeza"
		elif self.danger_nearby:
			return "miedo"
		elif self.hambre >= 45 and self.sed >= 50 and self.salud >= 60 and self.energia >= 45:
			return "alegría"
		else:
			return "ira"

	@property
	def emotion_id(self) -> int:
		return EMOTION_INDEX[self.emotion_name]

	def clamp(self):
		self.hambre = max(0.0, min(100.0, self.hambre))
		self.sed = max(0.0, min(100.0, self.sed))
		self.salud = max(0.0, min(100.0, self.salud))
		self.energia = max(0.0, min(100.0, self.energia))
		self.aburrimiento = max(0.0, min(100.0, self.aburrimiento))
		
		# K.O. / Incapacitation Mode: falls unconscious instead of instant death
		if self.alive and getattr(self, "debuff_inconsciente_ticks", 0) == 0 and (self.salud <= 0.0 or self.hambre <= 0.0 or self.sed <= 0.0):
			self.debuff_inconsciente_ticks = 12
			self.salud = 10.0
			self.hambre = 10.0
			self.sed = 10.0
			self.just_fell_unconscious = True
		
		# If unconscious and health drops to 0, they die permanently
		if self.alive and getattr(self, "debuff_inconsciente_ticks", 0) > 0 and self.salud <= 0.0:
			self.alive = False
			self.debuff_inconsciente_ticks = 0


# ── Mundo Cooperativo ───────────────────────────────────────────────────────

class CooperativeWorld:
	"""
	La Arena — mundo hostil para dos agentes.

	Comida escasa, fog of war, metabolismo rápido.
	Los mudos mueren.
	"""

	def __init__(self, seed: int = 42, food_interval: int = 8,
				food_duration: int = 6, hunger_rate: float = 3.5,
				resource_capacity: float = 5.0, resource_recovery: float = 0.2,
				predator_chance_multiplier: float = 1.0, predator_damage_multiplier: float = 1.0,
				storm_chance_multiplier: float = 1.0, storm_damage_multiplier: float = 1.0,
				prey_spawn_interval: int = 15, replenish_cooldown_ticks: int = 0,
				map_path: str | None = None, food_decay_rate: float = 0.5,
				permanent_fire_location: str | None = None):
		self.rng = random.Random(seed)
		self.food_decay_rate = food_decay_rate
		self.permanent_fire_location = permanent_fire_location

		# Cargar mapa personalizado si se especifica
		if map_path is not None:
			import json
			with open(map_path, encoding="utf-8") as f:
				map_data = json.load(f)

			custom_locations = {}
			for node in map_data["nodes"]:
				nid = node["id"]
				custom_locations[nid] = {
					"description": node.get("description", "zona de la arena"),
					"predator_chance": node.get("predator_chance", 0.1),
					"storm_shelter": node.get("storm_shelter", False),
					"food_eligible": node.get("food_eligible", False),
					"water_available": node.get("water_available", False),
					"branches_eligible": node.get("branches_eligible", False),
					"stones_eligible": node.get("stones_eligible", False),
					"fish_eligible": node.get("fish_eligible", False),
				}

			custom_adjacency = {node["id"]: [] for node in map_data["nodes"]}
			for edge in map_data["edges"]:
				frm, to = edge["from"], edge["to"]
				if frm in custom_adjacency and to in custom_adjacency:
					if to not in custom_adjacency[frm]:
						custom_adjacency[frm].append(to)
					if frm not in custom_adjacency[to]:
						custom_adjacency[to].append(frm)

			custom_glyphs = {}
			def get_glyph_for_location_name(name: str) -> str:
				name_lower = name.lower()
				if "cueva" in name_lower or "ruina" in name_lower or "casa" in name_lower or "refugio" in name_lower:
					return "cueva"
				if "bosque" in name_lower or "selva" in name_lower or "arbol" in name_lower:
					return "bosque"
				if "rio" in name_lower or "río" in name_lower or "arroyo" in name_lower:
					return "río"
				if "lago" in name_lower or "pantano" in name_lower or "agua" in name_lower or "charca" in name_lower:
					return "agua"
				if "montaña" in name_lower or "montana" in name_lower or "piedra" in name_lower or "pico" in name_lower or "roca" in name_lower:
					return "piedra"
				return "tierra"

			for node in map_data["nodes"]:
				nid = node["id"]
				custom_glyphs[nid] = get_glyph_for_location_name(nid)

			global COOP_LOCATIONS, COOP_ADJACENCY, COOP_LOCATION_NAMES, COOP_LOCATION_GLYPHS, _GLYPH_TO_LOCATIONS, FOOD_ELIGIBLE
			COOP_LOCATIONS.clear()
			COOP_LOCATIONS.update(custom_locations)

			COOP_ADJACENCY.clear()
			COOP_ADJACENCY.update(custom_adjacency)

			COOP_LOCATION_NAMES.clear()
			COOP_LOCATION_NAMES.extend(list(COOP_LOCATIONS.keys()))

			COOP_LOCATION_GLYPHS.clear()
			COOP_LOCATION_GLYPHS.update(custom_glyphs)

			_GLYPH_TO_LOCATIONS.clear()
			for _loc, _glyph_name in COOP_LOCATION_GLYPHS.items():
				_idx = WORD_INDEX.get(_glyph_name, -1)
				if _idx >= 0:
					_GLYPH_TO_LOCATIONS.setdefault(_idx, []).append(_loc)

			FOOD_ELIGIBLE.clear()
			FOOD_ELIGIBLE.extend([loc for loc, data in COOP_LOCATIONS.items() if data["food_eligible"]])

		self.food_interval = food_interval    # ticks entre spawns de comida
		self.food_duration = food_duration    # cuánto dura la comida
		self.hunger_rate = hunger_rate        # hambre perdida por tick
		self.resource_capacity = resource_capacity
		self.resource_recovery = resource_recovery
		self.predator_chance_multiplier = predator_chance_multiplier
		self.predator_damage_multiplier = predator_damage_multiplier
		self.storm_chance_multiplier = storm_chance_multiplier
		self.storm_damage_multiplier = storm_damage_multiplier
		self.replenish_cooldown_ticks = replenish_cooldown_ticks

		# Estado compartido del mundo
		self.world_tick: int = 0
		self.last_shout_tick = {}
		self.resource_cooldowns = {
			loc: {"food": 0, "water": 0, "branches": 0, "stones": 0}
			for loc in COOP_LOCATION_NAMES
		}

		# Mecánica de la presa (caza cooperativa)
		self.prey_spawn_interval = prey_spawn_interval
		self.prey_location = None
		self.prey_timer = 0
		self.prey_cooldown = self.rng.randint(5, self.prey_spawn_interval)

		# Mecánica del Gran Herbívoro (Stag Hunt)
		self.megaherbivore_location = None
		self.megaherbivore_timer = 0
		self.megaherbivore_cooldown = self.rng.randint(15, 30)

		# Capacidades dinámicas de recursos agotables
		self.resource_capacities = {
			loc: {
				"food": self.resource_capacity if (loc_data["food_eligible"] or loc_data.get("fish_eligible")) else 0.0,
				"water": self.resource_capacity if loc_data["water_available"] else 0.0,
				"branches": self.resource_capacity if loc_data.get("branches_eligible") else 0.0,
				"stones": self.resource_capacity if loc_data.get("stones_eligible") else 0.0
			}
			for loc, loc_data in COOP_LOCATIONS.items()
		}

		self.fire_locations = dict.fromkeys(COOP_LOCATION_NAMES, 0)
		if self.permanent_fire_location is not None:
			self.fire_locations[self.permanent_fire_location] = 999999

		# Agentes
		self.agent_a = CoopAgentState(agent_id="a")
		self.agent_b = CoopAgentState(agent_id="b")
		self.agent_c = CoopAgentState(agent_id="c")
		self.agent_d = CoopAgentState(agent_id="d", alive=False)
		self.agents = [self.agent_a, self.agent_b, self.agent_c, self.agent_d]

		# Configurar los modelos mentales cruzados
		for agent in self.agents:
			agent.companion_model = {}
			for other in self.agents:
				if other.agent_id != agent.agent_id:
					agent.companion_model[other.agent_id] = {
						"location": "cueva",
						"hambre": 60.0,
						"sed": 80.0,
						"salud": 100.0,
						"energia": 80.0,
						"emotion_name": "alegría",
						"last_updated": 0
					}

		# Spawn positions: agentes empiezan en sitios diferentes
		self._spawn_agents()

	def _spawn_agents(self):
		"""Colocar agentes en localizaciones diferentes."""
		locs = list(COOP_LOCATION_NAMES)
		self.rng.shuffle(locs)
		self.agent_a.location = locs[0]
		self.agent_b.location = locs[1]
		self.agent_c.location = locs[2]
		self.agent_d.location = "cueva"
		self.spawn_cooldown = self.rng.randint(3, self.food_interval)

	def reset(self, seed: int | None = None) -> tuple:
		"""Reiniciar para nuevo episodio."""
		# Preservar atributos persistentes de la fase 6 y 7 si ya existen
		prev_a = getattr(self, "agent_a", None)
		prev_b = getattr(self, "agent_b", None)
		prev_c = getattr(self, "agent_c", None)
		prev_d = getattr(self, "agent_d", None)

		if seed is not None:
			self.rng = random.Random(seed)
		self.agent_a = CoopAgentState(agent_id="a")
		self.agent_b = CoopAgentState(agent_id="b")
		self.agent_c = CoopAgentState(agent_id="c")
		self.agent_d = CoopAgentState(agent_id="d", alive=False)
		self.agents = [self.agent_a, self.agent_b, self.agent_c, self.agent_d]
		self.last_shout_tick = {}
		self.resource_cooldowns = {
			loc: {"food": 0, "water": 0, "branches": 0, "stones": 0}
			for loc in COOP_LOCATION_NAMES
		}

		# Restaurar y aplicar peaje metabólico de neurogénesis
		for prev, curr in [(prev_a, self.agent_a), (prev_b, self.agent_b), (prev_c, self.agent_c), (prev_d, self.agent_d)]:
			if prev is not None:
				curr.network_width = prev.network_width
				curr.learned_skills = list(prev.learned_skills) if prev.learned_skills is not None else curr.learned_skills
				curr.pending_growth_penalty = prev.pending_growth_penalty
				if curr.pending_growth_penalty:
					curr.hambre = max(0.0, curr.hambre - 25.0)
					curr.sed = max(0.0, curr.sed - 25.0)
					curr.pending_growth_penalty = False

		# Configurar los modelos mentales cruzados
		for agent in self.agents:
			agent.companion_model = {}
			for other in self.agents:
				if other.agent_id != agent.agent_id:
					agent.companion_model[other.agent_id] = {
						"location": "cueva",
						"hambre": 60.0,
						"sed": 80.0,
						"salud": 100.0,
						"energia": 80.0,
						"aburrimiento": 0.0,
						"emotion_name": "alegría",
						"last_updated": 0
					}

		self.resource_capacities = {
			loc: {
				"food": self.resource_capacity if (loc_data["food_eligible"] or loc_data.get("fish_eligible")) else 0.0,
				"water": self.resource_capacity if loc_data["water_available"] else 0.0,
				"branches": self.resource_capacity if loc_data.get("branches_eligible") else 0.0,
				"stones": self.resource_capacity if loc_data.get("stones_eligible") else 0.0
			}
			for loc, loc_data in COOP_LOCATIONS.items()
		}
		self.fire_locations = dict.fromkeys(COOP_LOCATION_NAMES, 0)
		if self.permanent_fire_location is not None:
			self.fire_locations[self.permanent_fire_location] = 999999
		self.prey_location = None
		self.prey_timer = 0
		self.prey_cooldown = self.rng.randint(5, self.prey_spawn_interval)
		self.megaherbivore_location = None
		self.megaherbivore_timer = 0
		self.megaherbivore_cooldown = self.rng.randint(15, 30)
		self.world_tick = 0
		self._spawn_agents()

		# Inicializar posiciones iniciales correctas en los modelos mentales
		for agent in self.agents:
			for other in self.agents:
				if other.agent_id != agent.agent_id:
					agent.companion_model[other.agent_id]["location"] = other.location

		return self.agent_a, self.agent_b, self.agent_c

	def _tick_world(self):
		"""Avanzar un tick del mundo: recuperación de recursos."""
		self.world_tick += 1

		# Recuperación de recursos: configurable
		for loc, caps in self.resource_capacities.items():
			if COOP_LOCATIONS[loc]["food_eligible"] or COOP_LOCATIONS[loc].get("fish_eligible"):
				if self.resource_cooldowns[loc]["food"] > 0:
					self.resource_cooldowns[loc]["food"] -= 1
					if self.resource_cooldowns[loc]["food"] == 0:
						caps["food"] = self.resource_capacity
				else:
					caps["food"] = min(self.resource_capacity, caps["food"] + self.resource_recovery)
			if COOP_LOCATIONS[loc]["water_available"]:
				if self.resource_cooldowns[loc]["water"] > 0:
					self.resource_cooldowns[loc]["water"] -= 1
					if self.resource_cooldowns[loc]["water"] == 0:
						caps["water"] = self.resource_capacity
				else:
					caps["water"] = min(self.resource_capacity, caps["water"] + self.resource_recovery)
			if COOP_LOCATIONS[loc].get("branches_eligible"):
				if self.resource_cooldowns[loc]["branches"] > 0:
					self.resource_cooldowns[loc]["branches"] -= 1
					if self.resource_cooldowns[loc]["branches"] == 0:
						caps["branches"] = self.resource_capacity
				else:
					caps["branches"] = min(self.resource_capacity, caps["branches"] + self.resource_recovery)
			if COOP_LOCATIONS[loc].get("stones_eligible"):
				if self.resource_cooldowns[loc]["stones"] > 0:
					self.resource_cooldowns[loc]["stones"] -= 1
					if self.resource_cooldowns[loc]["stones"] == 0:
						caps["stones"] = self.resource_capacity
				else:
					caps["stones"] = min(self.resource_capacity, caps["stones"] + self.resource_recovery)

		# Decaer hogueras activas
		for loc in self.fire_locations:
			if self.fire_locations[loc] > 0 and loc != self.permanent_fire_location:
				self.fire_locations[loc] -= 1

		# Decaimiento (podrido) de la comida en el suelo si no hay hoguera
		for loc, caps in self.resource_capacities.items():
			if caps["food"] > 0.0 and self.fire_locations.get(loc, 0) == 0:
				caps["food"] = max(0.0, caps["food"] - self.food_decay_rate)
				if caps["food"] <= 0.0 and self.replenish_cooldown_ticks > 0:
					self.resource_cooldowns[loc]["food"] = self.replenish_cooldown_ticks

		# Mantenimiento de la presa
		if self.prey_location is not None:
			self.prey_timer -= 1
			if self.prey_timer <= 0:
				self.prey_location = None
		else:
			self.prey_cooldown -= 1
			if self.prey_cooldown <= 0:
				# Spawn de presa en localización con recursos
				eligible = [loc for loc, data in COOP_LOCATIONS.items() if data["food_eligible"] or data["water_available"]]
				self.prey_location = self.rng.choice(eligible)
				self.prey_timer = 10  # dura 10 ticks
				self.prey_cooldown = self.prey_spawn_interval

		# Mantenimiento del Gran Herbívoro
		if self.megaherbivore_location is not None:
			self.megaherbivore_timer -= 1
			if self.megaherbivore_timer <= 0:
				self.megaherbivore_location = None
			else:
				# 20% de probabilidad de moverse a una localización adyacente
				if self.rng.random() < 0.20:
					adjacent = COOP_ADJACENCY[self.megaherbivore_location]
					self.megaherbivore_location = self.rng.choice(adjacent)
		else:
			self.megaherbivore_cooldown -= 1
			if self.megaherbivore_cooldown <= 0:
				# Spawn de Gran Herbívoro en pradera o valle
				self.megaherbivore_location = self.rng.choice(["pradera", "valle"])
				self.megaherbivore_timer = 25  # dura 25 ticks
				self.megaherbivore_cooldown = 40  # reaparece cada 40 ticks

	def perceive(self, agent: CoopAgentState) -> list[int]:
		"""
		Generar input de 6 tokens para un agente (fog of war).

		[0] mi localización
		[1] lo que veo aquí (comida/agua/peligro/nada)
		[2] mi estado corporal (emoción)
		[3] detalle local
		[4] señal recibida o predicción de localización
		[5] señal recibida o predicción de emoción
		"""
		loc = agent.location
		loc_data = COOP_LOCATIONS[loc]
		loc_glyph = WORD_INDEX.get(COOP_LOCATION_GLYPHS[loc], 0)

		# Actualizar conocimiento local del mapa
		caps = self.resource_capacities[loc]
		agent.map_knowledge[loc] = {
			"food": caps["food"],
			"water": caps["water"],
			"last_updated": self.world_tick
		}

		# ¿Qué hay aquí? (Megaherbívoro/Presa override)
		if self.megaherbivore_location == loc:
			what_glyph = WORD_INDEX.get("árbol", 0)  # "árbol" como proxy para el Gran Herbívoro
		elif self.prey_location == loc:
			what_glyph = WORD_INDEX.get("grupo", 0)
		elif self.fire_locations.get(loc, 0) > 0:
			what_glyph = WORD_INDEX.get("fuego", 0)
		elif caps["food"] >= 1.0:
			what_glyph = WORD_INDEX.get("comida", 0)
		elif caps["water"] >= 1.0:
			what_glyph = WORD_INDEX.get("agua", 0)
		else:
			what_glyph = WORD_INDEX.get("noche", 0)  # "nada"

		# Depredador override
		agent.danger_nearby = self.rng.random() < (loc_data["predator_chance"] * self.predator_chance_multiplier)
		if agent.danger_nearby:
			what_glyph = WORD_INDEX.get("depredador", 0)

		# Tormenta
		agent.storm_active = self.rng.random() < (0.08 * self.storm_chance_multiplier)
		if agent.storm_active:
			what_glyph = WORD_INDEX.get("tormenta", 0)

		# Estado corporal (emoción como glyph)
		body_glyph = WORD_INDEX.get(agent.emotion_name, 0)

		# Detalle local: mapea el estado de la mochila
		if getattr(agent, "tiene_lanza", False):
			detail_glyph = WORD_INDEX.get("seguro", 0)
		elif agent.mochila_comida > 0 and agent.mochila_agua > 0:
			detail_glyph = WORD_INDEX.get("grupo", 0)
		elif agent.mochila_comida > 0:
			detail_glyph = WORD_INDEX.get("comida", 0)
		elif agent.mochila_agua > 0:
			detail_glyph = WORD_INDEX.get("agua", 0)
		elif getattr(agent, "mochila_piedras", 0) > 0:
			detail_glyph = WORD_INDEX.get("piedra", 0)
		elif getattr(agent, "mochila_ramas", 0) > 0:
			detail_glyph = WORD_INDEX.get("árbol", 0)
		else:
			detail_glyph = WORD_INDEX.get("noche", 0)  # vacía

		# Señal o Teoría de la Mente (tokens [4] y [5])
		if agent.received_signal and agent.signal_retrievability > 0.1:
			sig_loc, sig_what = agent.received_signal
			agent.signal_age += 1  # la memoria envejece cada tick
		else:
			# Si no hay grito fresco, mostramos la predicción del compañero en estado más crítico (Worst-State ToM)
			critical_id = None
			min_val = 999.0
			for other_id, model in agent.companion_model.items():
				h_est = model.get("hambre", 60.0)
				sed_est = model.get("sed", 80.0)
				s_est = model.get("salud", 100.0)
				metric = min(h_est, sed_est, s_est)
				if metric < min_val:
					min_val = metric
					critical_id = other_id
			
			if critical_id:
				pred_model = agent.companion_model[critical_id]
				pred_loc = pred_model.get("location", "cueva")
				pred_emo = pred_model.get("emotion_name", "alegría")
			else:
				pred_loc = "cueva"
				pred_emo = "alegría"
			
			sig_loc = WORD_INDEX.get(COOP_LOCATION_GLYPHS.get(pred_loc, "cueva"), 0)
			sig_what = WORD_INDEX.get(pred_emo, 0)

		return [loc_glyph, what_glyph, body_glyph, detail_glyph, sig_loc, sig_what]

	def act(self, agent: CoopAgentState, action: str, shout_concept_idx: int = None) -> dict:
		"""Ejecutar acción para un agente."""
		if not agent.alive:
			return {
				"success": False,
				"delta_hambre": 0.0,
				"delta_sed": 0.0,
				"delta_salud": 0.0,
				"delta_energia": 0.0,
				"event": "muerto",
				"moved_to": None,
				"shouted": False,
				"shout_content": None,
				"shared_resource": None,
			}

		# Si está en K.O., forzamos inactividad y decremento del timer
		if getattr(agent, "debuff_inconsciente_ticks", 0) > 0:
			agent.debuff_inconsciente_ticks -= 1
			# Si expira el desmayo sin reanimación, muere
			if agent.debuff_inconsciente_ticks == 0:
				agent.alive = False
				agent.salud = 0.0
			agent.previous_location = agent.location
			agent.last_action = "dormir"
			agent.tick += 1
			# Durante el desmayo recupera un poco de energía sin perder salud
			agent.energia = min(100.0, agent.energia + 5.0)
			# Metabolismo basal leve durante el desmayo
			agent.hambre = max(0.0, agent.hambre - 0.1)
			agent.sed = max(0.0, agent.sed - 0.2)
			agent.clamp()
			
			event_str = f"INCONSCIENTE: El agente está fuera de combate y recuperándose ({agent.debuff_inconsciente_ticks}t restantes)" if agent.alive else "MUERTO: Expiró el tiempo de inconsciencia sin rescate"
			
			return {
				"success": False,
				"delta_hambre": -0.1,
				"delta_sed": -0.2,
				"delta_salud": 0.0,
				"delta_energia": 5.0,
				"event": event_str,
				"moved_to": None,
				"shouted": False,
				"shout_content": None,
				"shared_resource": None,
			}

		forced_sleep = False
		original_action = action
		if agent.energia <= 0.0 and action != "dormir":
			action = "dormir"
			forced_sleep = True

		agent.previous_location = agent.location
		loc_data = COOP_LOCATIONS[agent.location]
		
		# Calcular peso y multiplicador metabólico via Prolog
		q_peso = f"calcular_peso({agent.mochila_comida}, {agent.mochila_agua}, {getattr(agent, 'mochila_ramas', 0)}, {getattr(agent, 'mochila_piedras', 0)}, {1 if getattr(agent, 'tiene_lanza', False) else 0}, Mult)"
		peso_multiplicador = query_prolog(q_peso)[0]["Mult"]

		result = {
			"success": False,
			"delta_hambre": -0.2 * self.hunger_rate * peso_multiplicador,
			"delta_sed": -1.0 * self.hunger_rate * peso_multiplicador,
			"delta_salud": 0,
			"delta_energia": -2.0 * peso_multiplicador,
			"event": "",
			"moved_to": None,
			"shouted": False,
			"shout_content": None,
			"shared_resource": None,
		}

		# Llenar mochila automáticamente si está en un recurso disponible
		caps = self.resource_capacities[agent.location]
		if caps["food"] >= 1.0 and agent.mochila_comida == 0:
			if "comida" in agent.learned_skills and loc_data["food_eligible"] or "pesca" in agent.learned_skills and loc_data.get("fish_eligible"):
				agent.mochila_comida = 1
				caps["food"] = max(0.0, caps["food"] - 1.0)
				if caps["food"] <= 0.0 and self.replenish_cooldown_ticks > 0:
					self.resource_cooldowns[agent.location]["food"] = self.replenish_cooldown_ticks
		
		if caps["water"] >= 1.0 and agent.mochila_agua == 0 and "agua" in agent.learned_skills and loc_data["water_available"]:
			agent.mochila_agua = 1
			caps["water"] = max(0.0, caps["water"] - 1.0)
			if caps["water"] <= 0.0 and self.replenish_cooldown_ticks > 0:
				self.resource_cooldowns[agent.location]["water"] = self.replenish_cooldown_ticks

		if "artesanía" in agent.learned_skills:
			if loc_data.get("branches_eligible") and caps["branches"] >= 1.0 and agent.mochila_ramas < 3:
				agent.mochila_ramas += 1
				caps["branches"] = max(0.0, caps["branches"] - 1.0)
				if caps["branches"] <= 0.0 and self.replenish_cooldown_ticks > 0:
					self.resource_cooldowns[agent.location]["branches"] = self.replenish_cooldown_ticks
			if loc_data.get("stones_eligible") and caps["stones"] >= 1.0 and agent.mochila_piedras < 3:
				agent.mochila_piedras += 1
				caps["stones"] = max(0.0, caps["stones"] - 1.0)
				if caps["stones"] <= 0.0 and self.replenish_cooldown_ticks > 0:
					self.resource_cooldowns[agent.location]["stones"] = self.replenish_cooldown_ticks

		# Evaluador de Intentos Lógicos por Prolog
		learning_action = action in ["comer", "beber", "fabricar", "construir", "encender"]
		result_type = "basal"
		res_event = ""
		if learning_action:
			fire_active = 1 if self.fire_locations.get(agent.location, 0) > 0 else 0
			has_lanza = 1 if getattr(agent, "tiene_lanza", False) else 0
			q_eval = (
				f"evaluar_intento('{action}', {format_skills(agent.learned_skills)}, "
				f"{agent.mochila_comida}, {agent.mochila_agua}, {agent.mochila_ramas}, "
				f"{agent.mochila_piedras}, {has_lanza}, '{agent.location}', "
				f"{caps.get('food', 0.0)}, {caps.get('water', 0.0)}, {caps.get('branches', 0.0)}, "
				f"{caps.get('stones', 0.0)}, {fire_active}, '{self.prey_location or 'ninguno'}', "
				f"ResultType, Event)"
			)
			res_eval = query_prolog(q_eval)[0]
			result_type = res_eval["ResultType"]
			res_event = res_eval["Event"]

		# Calcular multiplicador emocional para el aprendizaje
		emotion = agent.emotion_name
		if emotion == "alegría":
			emotion_mult = 1.0
		elif emotion in ["ira", "miedo", "hambre"]:
			emotion_mult = 0.2
		else:  # tristeza, dolor
			emotion_mult = 0.0

		if result_type == "intento_valido":
			# Determinar la habilidad según la acción
			skill = None
			if action == "comer":
				skill = "pesca" if loc_data.get("fish_eligible") else "comida"
			elif action == "beber":
				skill = "agua"
			elif action == "fabricar":
				skill = "artesanía"
				agent.mochila_ramas = max(0, agent.mochila_ramas - 1)
				agent.mochila_piedras = max(0, agent.mochila_piedras - 1)
			elif action == "construir":
				skill = "construcción"
				agent.mochila_ramas = max(0, agent.mochila_ramas - 2)
				agent.mochila_piedras = max(0, agent.mochila_piedras - 1)
			elif action == "encender":
				skill = "fuego"
				agent.mochila_ramas = max(0, agent.mochila_ramas - 2)

			if skill:
				result["tried_skill"] = skill
				if len(agent.learned_skills) < agent.max_slots:
					# Confirmar prerrequisitos
					q_tree = f"prerrequisitos_satisfechos('{skill}', {format_skills(agent.learned_skills)})"
					if query_prolog(q_tree):
						exp_gain = 1.0 * emotion_mult
						agent.skill_experience[skill] = agent.skill_experience.get(skill, 0.0) + exp_gain
						if agent.skill_experience[skill] >= 10.0:
							agent.learned_skills.append(skill)
							agent.skill_experience[skill] = 0.0
							print(f"  🎓 ¡{agent.agent_id.upper()} ({agent.emotion_name}) ha DESCUBIERTO/INVENTADO la habilidad '{skill}'!")

			result["success"] = False
			result["event"] = res_event

		elif result_type == "error_requisito":
			result["success"] = False
			result["tech_error"] = True
			result["event"] = res_event

		elif result_type == "error_fisico":
			result["success"] = False
			result["physical_error"] = True
			result["event"] = res_event

		elif result_type in ["exito", "basal"]:
			if action == "comer":
				fire_active = 1 if self.fire_locations.get(agent.location, 0) > 0 else 0
				skills_str = format_skills(agent.learned_skills)
				food_eligible = 1 if loc_data["food_eligible"] else 0
				fish_eligible = 1 if loc_data.get("fish_eligible") else 0
				rng_val = self.rng.random()
				agent_id = f"'{agent.agent_id.upper()}'"
				
				q_comer = (
					f"comer({fire_active}, {skills_str}, {agent.mochila_comida}, {agent.mochila_agua}, "
					f"{caps['food']}, {food_eligible}, {fish_eligible}, {rng_val}, {agent_id}, "
					f"NewComida, NewAgua, NewGround, Success, AddHambre, AddSed, AddSalud, Event, Intoxicated)"
				)
				res = query_prolog(q_comer)[0]
				
				agent.mochila_comida = res["NewComida"]
				agent.mochila_agua = res["NewAgua"]
				caps["food"] = max(0.0, res["NewGround"])
				if caps["food"] <= 0.0 and self.replenish_cooldown_ticks > 0:
					self.resource_cooldowns[agent.location]["food"] = self.replenish_cooldown_ticks
				result["success"] = bool(res["Success"])
				if result["success"]:
					if res["Event"] == 'consume guiso caliente (comida++)':
						result["delta_hambre"] += 85.0
						result["delta_sed"] += 50.0
					else:
						result["delta_hambre"] += 75.0
					result["delta_salud"] += res["AddSalud"]
				else:
					result["delta_hambre"] += res["AddHambre"]
					result["delta_sed"] += res["AddSed"]
					result["delta_salud"] += res["AddSalud"]
				result["event"] = res["Event"]
				
				if not result["success"] and agent.mochila_comida == 0 and (loc_data["food_eligible"] or loc_data.get("fish_eligible")) and caps["food"] < 1.0:
					result["event"] = f"intenta recolectar/comer comida de {agent.location} agotado (penalización)"

			elif action == "beber":
				if caps["water"] >= 1.0 and "agua" in agent.learned_skills:
					result["success"] = True
					result["delta_sed"] += 75.0
					result["delta_salud"] += 3.0
					result["event"] = "bebe agua (suelo)"
					caps["water"] = max(0.0, caps["water"] - 1.0)
					if caps["water"] <= 0.0 and self.replenish_cooldown_ticks > 0:
						self.resource_cooldowns[agent.location]["water"] = self.replenish_cooldown_ticks
				elif agent.mochila_agua > 0:
					agent.mochila_agua = 0
					result["success"] = True
					result["delta_sed"] += 75.0
					result["delta_salud"] += 3.0
					result["event"] = "bebe agua de su mochila"
				else:
					result["event"] = "no hay agua aquí ni en mochila"
					if loc_data["water_available"] and caps["water"] < 1.0:
						result["event"] = f"intenta beber agua de {agent.location} agotado (penalización)"
					elif agent.agent_id == "b" and caps["water"] >= 1.0:
						result["event"] = "Sofy no sabe extraer agua del suelo"

			elif action == "dormir":
				result["delta_hambre"] = -0.2 * self.hunger_rate * 0.5 * peso_multiplicador
				result["delta_sed"] = -1.0 * self.hunger_rate * 0.5 * peso_multiplicador
				
				is_fishing_spot = loc_data.get("fish_eligible") or agent.location in ["río", "lago", "pantano"]
				
				if forced_sleep and original_action == "luchar":
					result["delta_salud"] = -100.0
					result["delta_energia"] = 0.0
					result["success"] = False
					result["event"] = "se desmaya de cansancio en medio del combate y es devorado"
				elif forced_sleep and original_action == "comer" and is_fishing_spot:
					result["delta_salud"] = -100.0
					result["delta_energia"] = 0.0
					result["success"] = False
					result["event"] = "se desmaya de cansancio pescando y se ahoga en el agua"
				elif agent.danger_nearby:
					if getattr(agent, "tiene_lanza", False):
						agent.tiene_lanza = False
						result["delta_energia"] += 10.0
						result["delta_salud"] += 3.0
						result["event"] = "duerme y el depredador es repelido por la lanza (se rompe)"
					else:
						result["delta_salud"] -= 35.0 * self.predator_damage_multiplier
						result["delta_energia"] += 10.0
						result["event"] = "duerme pero depredador ataca!"
				else:
					result["success"] = True
					if forced_sleep:
						result["delta_energia"] += 10.0
						result["delta_salud"] -= 2.0
						result["event"] = "colapsa por cansancio extremo y duerme forzosamente"
					else:
						result["delta_energia"] += 20.0
						result["delta_salud"] += 3.0
						result["event"] = "duerme y descansa"

			elif action == "mover":
				if agent.energia < 5:
					result["event"] = "demasiado cansado para moverse"
				else:
					adjacent = COOP_ADJACENCY[agent.location]

					if agent.nav_target and agent.nav_target != agent.location:
						next_step = _bfs_next_step(agent.location, agent.nav_target)
						new_loc = next_step if next_step and next_step in adjacent else self.rng.choice(adjacent)
					else:
						new_loc = self.rng.choice(adjacent)

					result["success"] = True
					result["delta_energia"] -= 4.0
					result["moved_to"] = new_loc
					result["event"] = f"se mueve a {new_loc}"
					if agent.nav_target:
						result["event"] += f" (→{agent.nav_target})"

					if new_loc == agent.nav_target:
						agent.nav_target = None
						agent.received_signal = None
						agent.signal_from = None

			elif action == "ver":
				result["success"] = True
				result["delta_energia"] -= 1.0
				result["event"] = "observa el entorno"
				
				# Consulta de Observación por Prolog
				fire_active = 1 if self.fire_locations.get(agent.location, 0) > 0 else 0
				food_elig = 1 if loc_data["food_eligible"] else 0
				fish_elig = 1 if loc_data.get("fish_eligible") else 0
				water_avail = 1 if loc_data["water_available"] else 0
				branches_elig = 1 if loc_data.get("branches_eligible") else 0
				stones_elig = 1 if loc_data.get("stones_eligible") else 0
				
				q_obs = (
					f"que_observar({format_skills(agent.learned_skills)}, '{agent.location}', {fire_active}, "
					f"{caps.get('food', 0.0)}, {food_elig}, {fish_elig}, {caps.get('water', 0.0)}, {water_avail}, "
					f"'{self.prey_location or 'ninguno'}', {caps.get('branches', 0.0)}, {branches_elig}, "
					f"{caps.get('stones', 0.0)}, {stones_elig}, Habilidad)"
				)
				res_obs = query_prolog(q_obs)
				observed_skills = []
				for r_obs in res_obs:
					obs_skill = r_obs["Habilidad"]
					if len(agent.learned_skills) < agent.max_slots:
						q_tree = f"prerrequisitos_satisfechos('{obs_skill}', {format_skills(agent.learned_skills)})"
						if query_prolog(q_tree):
							exp_gain = 1.0 * emotion_mult
							agent.skill_experience[obs_skill] = agent.skill_experience.get(obs_skill, 0.0) + exp_gain
							observed_skills.append(obs_skill)
							if agent.skill_experience[obs_skill] >= 10.0:
								agent.learned_skills.append(obs_skill)
								agent.skill_experience[obs_skill] = 0.0
								print(f"  🎓 ¡{agent.agent_id.upper()} ({agent.emotion_name}) ha DESCUBIERTO/INVENTADO '{obs_skill}' mediante observación!")
				
				if observed_skills:
					result["observed_skills"] = observed_skills
					result["tried_skill"] = observed_skills[0]

			elif action == "luchar":
				danger_val = 1 if agent.danger_nearby else 0
				lanza_val = 1 if getattr(agent, "tiene_lanza", False) else 0
				rng_val = self.rng.random()
				q_lucha = (
					f"luchar({danger_val}, {lanza_val}, {rng_val}, {self.predator_damage_multiplier}, "
					f"NewLanza, NewDanger, Success, SubSalud, SubEnergia, Event)"
				)
				res = query_prolog(q_lucha)[0]
				agent.tiene_lanza = bool(res["NewLanza"])
				agent.danger_nearby = bool(res["NewDanger"])
				result["success"] = bool(res["Success"])
				result["delta_salud"] += res["SubSalud"]
				result["delta_energia"] += res["SubEnergia"]
				result["event"] = res["Event"]

			elif action == "gritar":
				result["success"] = True
				result["delta_energia"] = 0.0
				result["shouted"] = True
				agent.shout_cooldown_ticks = 3
				loc_glyph = WORD_INDEX.get(COOP_LOCATION_GLYPHS[agent.location], 0)
				if shout_concept_idx is not None:
					what_glyph = shout_concept_idx
				else:
					caps_here = self.resource_capacities[agent.location]
					if self.megaherbivore_location == agent.location:
						what_glyph = WORD_INDEX.get("árbol", 0)
					elif caps_here["food"] >= 1.0:
						what_glyph = WORD_INDEX.get("comida", 0)
					elif agent.danger_nearby:
						what_glyph = WORD_INDEX.get("depredador", 0)
					elif caps_here["water"] >= 1.0:
						what_glyph = WORD_INDEX.get("agua", 0)
					else:
						what_glyph = WORD_INDEX.get("seguro", 0)

				if self._what_name(what_glyph) == "enseñar":
					has_eligible_student = False
					for other_agent in self.agents:
						if other_agent.agent_id != agent.agent_id and other_agent.alive:
							if len(other_agent.learned_skills) < other_agent.max_slots:
								has_eligible_student = True
								break
					if not has_eligible_student:
						what_glyph = WORD_INDEX.get("seguro", 0)

				result["shout_content"] = (loc_glyph, what_glyph)
				result["event"] = f"grita: [{COOP_LOCATION_GLYPHS[agent.location]}, {self._what_name(what_glyph)}]"
				concept_name = self._what_name(what_glyph)
				if concept_name in ["agua", "comida"]:
					self.last_shout_tick[(agent.agent_id, concept_name)] = self.world_tick

			elif action == "dar":
				others = [other for other in self.agents if other.agent_id != agent.agent_id and other.alive]
				transferred = False
				for other_agent in others:
					if other_agent.location == agent.location:
						if other_agent.mochila_agua == 0 and other_agent.sed < 80 and agent.mochila_agua > 0:
							if self.world_tick - self.last_shout_tick.get((other_agent.agent_id, "agua"), -999) <= 15:
								agent.mochila_agua = 0
								other_agent.mochila_agua = 1
								result["success"] = True
								result["event"] = f"comparte agua de su mochila con {other_agent.agent_id.upper()} (llena mochila)"
								result["shared_resource"] = "agua"
								result["shared_with"] = other_agent.agent_id
								transferred = True
								break
							else:
								result["event"] = f"intenta dar agua a {other_agent.agent_id.upper()} pero no ha sido solicitada"
						elif other_agent.mochila_comida == 0 and other_agent.hambre < 80 and agent.mochila_comida > 0:
							if self.world_tick - self.last_shout_tick.get((other_agent.agent_id, "comida"), -999) <= 15:
								agent.mochila_comida = 0
								other_agent.mochila_comida = 1
								result["success"] = True
								result["event"] = f"comparte comida de su mochila con {other_agent.agent_id.upper()} (llena mochila)"
								result["shared_resource"] = "comida"
								result["shared_with"] = other_agent.agent_id
								transferred = True
								break
							else:
								result["event"] = f"intenta dar comida a {other_agent.agent_id.upper()} pero no ha sido solicitada"

				if not transferred and not result.get("event"):
					result["event"] = "mochila vacía, compañeros saciados o solos"

			elif action == "enseñar":
				others = [other for other in self.agents if other.agent_id != agent.agent_id and other.alive]
				taught = False
				for other_agent in others:
					if other_agent.location == agent.location:
						if len(other_agent.learned_skills) < other_agent.max_slots:
							teacher_skills = format_skills(agent.learned_skills)
							student_skills = format_skills(other_agent.learned_skills)
							prey_loc = f"'{self.prey_location}'" if self.prey_location else "ninguno"
							fire_act = 1 if self.fire_locations.get(agent.location, 0) > 0 else 0
							
							q_teach = (
								f"enseñar_elegible('{agent.location}', {teacher_skills}, {student_skills}, "
								f"{prey_loc}, {fire_act}, Success, Habilidad)"
							)
							res = query_prolog(q_teach)[0]
							if res["Success"]:
								habilidad = res["Habilidad"]
								result["success"] = True
								result["event"] = f"enseña {habilidad} a {other_agent.agent_id.upper()} en {agent.location.upper()}"
								result["taught_skill"] = habilidad
								result["taught_to"] = other_agent.agent_id
								taught = True
								break

				if not taught:
					result["event"] = "nadie a quien enseñar, sin capacidad o fuera de zona de recursos"

			elif action == "aprender":
				others = [other for other in self.agents if other.agent_id != agent.agent_id and other.alive]
				learned = False
				for other_agent in others:
					if other_agent.location == agent.location:
						other_action = getattr(self, "current_actions", {}).get(other_agent.agent_id)
						if other_action == "enseñar":
							if len(agent.learned_skills) < agent.max_slots:
								teacher_skills = format_skills(other_agent.learned_skills)
								student_skills = format_skills(agent.learned_skills)
								prey_loc = f"'{self.prey_location}'" if self.prey_location else "ninguno"
								fire_act = 1 if self.fire_locations.get(agent.location, 0) > 0 else 0
								
								q_learn = (
									f"enseñar_elegible('{agent.location}', {teacher_skills}, {student_skills}, "
									f"{prey_loc}, {fire_act}, Success, Habilidad)"
								)
								res = query_prolog(q_learn)[0]
								if res["Success"]:
									habilidad = res["Habilidad"]
									result["success"] = True
									result["event"] = f"aprende {habilidad} de {other_agent.agent_id.upper()} en {agent.location.upper()}"
									result["learning_skill"] = habilidad
									result["learned_from"] = other_agent.agent_id
									result["delta_energia"] -= 1.0
									learned = True
									break

				if not learned:
					result["event"] = "nadie enseñando en esta zona, o capacidad cerebral llena"

			elif action == "reproducir":
				loc_data = COOP_LOCATIONS[agent.location]
				start_stats = getattr(self, "start_of_tick_stats", {})
				agent_start = start_stats.get(agent.agent_id, {"hambre": agent.hambre, "sed": agent.sed, "location": agent.location})

				if not loc_data.get("storm_shelter", False):
					result["event"] = "no se puede reproducir fuera de un refugio"
				elif agent_start["hambre"] < 40.0 or agent_start["sed"] < 40.0:
					result["event"] = "demasiado hambriento o sediento para reproducirse"
				else:
					others = [other for other in self.agents if other.agent_id != agent.agent_id and other.alive]
					reproduced = False
					for other_agent in others:
						other_start = start_stats.get(other_agent.agent_id, {"hambre": other_agent.hambre, "sed": other_agent.sed, "location": other_agent.location})
						if other_start["location"] == agent_start["location"]:
							other_action = getattr(self, "current_actions", {}).get(other_agent.agent_id)
							if other_action == "reproducir" and other_start["hambre"] >= 40.0 and other_start["sed"] >= 40.0:
								if not self.agent_d_was_alive:
									result["success"] = True
									result["delta_hambre"] -= 35.0
									result["delta_sed"] -= 35.0
									result["event"] = f"se reproduce con {other_agent.agent_id.upper()} en {agent.location.upper()}"
									result["reproduction_event"] = {
										"parent_a": agent.agent_id,
										"parent_b": other_agent.agent_id
									}
									reproduced = True
									break
					if not reproduced:
						result["event"] = "ningún compañero ejecutando reproducir o Domi ya está vivo"

			elif action == "fabricar":
				skills_str = format_skills(agent.learned_skills)
				lanza_val = 1 if getattr(agent, "tiene_lanza", False) else 0
				q_fab = (
					f"fabricar({skills_str}, {agent.mochila_ramas}, {agent.mochila_piedras}, {lanza_val}, "
					f"NewRamas, NewPiedras, NewLanza, Success, Event)"
				)
				res = query_prolog(q_fab)[0]
				agent.mochila_ramas = res["NewRamas"]
				agent.mochila_piedras = res["NewPiedras"]
				agent.tiene_lanza = bool(res["NewLanza"])
				result["success"] = bool(res["Success"])
				result["event"] = res["Event"]

			elif action == "construir":
				skills_str = format_skills(agent.learned_skills)
				shelter_val = 1 if loc_data.get("storm_shelter", False) else 0
				q_const = (
					f"construir({skills_str}, {agent.mochila_ramas}, {agent.mochila_piedras}, {shelter_val}, "
					f"'{agent.location}', NewRamas, NewPiedras, NewShelter, Success, Event)"
				)
				res = query_prolog(q_const)[0]
				agent.mochila_ramas = res["NewRamas"]
				agent.mochila_piedras = res["NewPiedras"]
				loc_data["storm_shelter"] = bool(res["NewShelter"])
				result["success"] = bool(res["Success"])
				result["event"] = res["Event"]

			elif action == "encender":
				skills_str = format_skills(agent.learned_skills)
				q_enc = (
					f"encender({skills_str}, {agent.mochila_ramas}, '{agent.location}', "
					f"NewRamas, FireDuration, Success, Event)"
				)
				res = query_prolog(q_enc)[0]
				agent.mochila_ramas = res["NewRamas"]
				self.fire_locations[agent.location] = res["FireDuration"]
				result["success"] = bool(res["Success"])
				result["event"] = res["Event"]

		elif action == "reanimar":
			companions_here = [
				other for other in self.agents
				if other.agent_id != agent.agent_id and other.alive and other.location == agent.location
				and getattr(other, "debuff_inconsciente_ticks", 0) > 0
			]
			if companions_here:
				target = companions_here[0]
				target.debuff_inconsciente_ticks = 0
				target.salud = 60.0
				target.hambre = 30.0
				target.sed = 30.0
				result["success"] = True
				result["event"] = f"¡REANIMACIÓN! Despertó a {target.agent_id.upper()}"
				result["reanimated_companion"] = True
			else:
				result["success"] = False
				result["event"] = "intentó reanimar pero nadie lo necesita aquí"

		# ── Consecuencias pasivas: Depredador via Prolog ──
		danger_val = 1 if agent.danger_nearby else 0
		lanza_val = 1 if getattr(agent, "tiene_lanza", False) else 0
		q_dep = (
			f"evaluar_depredador({danger_val}, '{action}', {lanza_val}, {self.predator_damage_multiplier}, "
			f"NewLanza, SubSalud, EventSuffix)"
		)
		res = query_prolog(q_dep)[0]
		agent.tiene_lanza = bool(res["NewLanza"])
		result["delta_salud"] += res["SubSalud"]
		if res["EventSuffix"]:
			result["event"] += res["EventSuffix"]

		if agent.storm_active and not loc_data.get("storm_shelter", False):
			result["delta_salud"] -= 12.0 * self.storm_damage_multiplier
			result["delta_energia"] -= 4.0
			result["event"] += " | tormenta"

		if agent.hambre <= 0:
			result["delta_salud"] -= 15.0
			result["event"] += " | inanición"

		if agent.sed <= 0:
			result["delta_salud"] -= 25.0
			result["event"] += " | deshidratación"

		# Si tiene debuff de tristeza, aplicar las reducciones (20% más rápido metabolismo, 50% menos energía al dormir)
		if agent.debuff_tristeza_ticks > 0:
			if result["delta_hambre"] < 0:
				result["delta_hambre"] *= 1.2
			if result.get("delta_sed", 0.0) < 0:
				result["delta_sed"] *= 1.2
			if action == "dormir" and result.get("delta_energia", 0) > 0:
				result["delta_energia"] *= 0.5

		# ── Aplicar deltas ──
		agent.hambre += result["delta_hambre"]
		agent.sed += result.get("delta_sed", 0.0)
		agent.salud += result["delta_salud"]
		agent.energia += result["delta_energia"]

		if result["moved_to"]:
			agent.location = result["moved_to"]

		agent.last_action = action
		agent.tick += 1
		
		# ── Aplicar aburrimiento ──
		if action == "dormir":
			agent.aburrimiento += 6.0
		elif action == "mover" and agent.location != agent.previous_location:
			agent.aburrimiento = max(0.0, agent.aburrimiento - 20.0)
		elif action in ["dar", "enseñar", "aprender", "reproducir", "luchar"]:
			agent.aburrimiento = max(0.0, agent.aburrimiento - 15.0)
		else:
			agent.aburrimiento += 2.0
			
		if agent.debuff_tristeza_ticks > 0:
			agent.debuff_tristeza_ticks -= 1

		if agent.shout_cooldown_ticks > 0:
			agent.shout_cooldown_ticks -= 1

		agent.clamp()

		return result

	def step(self, action_a: str, action_b: str, action_c: str = "ver", action_d: str = "ver",
			shout_concept_a: int = None, shout_concept_b: int = None, shout_concept_c: int = None, shout_concept_d: int = None) -> tuple:
		"""
		Un tick completo del mundo cooperativo a 4 bandas (con Domi dinámico).

		1. Mundo avanza (recuperación + presa)
		2. Los agentes perciben
		3. Los agentes actúan
		4. Resolución de reproducción (nace Domi)
		5. Resolución de caza cooperativa de presa
		6. Broadcast de gritos Half-Duplex
		7. Sincronización ToM y mapas
		8. Planificación de rescate ToM
		9. Asignación de coop_bonuses por consecuencias

		Returns: (result_a, result_b, result_c, world_info)
		"""
		self._tick_world()

		# Registrar si Domi ya estaba vivo al empezar el tick
		self.agent_d_was_alive = self.agent_d.alive

		# Registrar las acciones planeadas para este tick (Fase 6 y 7)
		self.current_actions = {
			"a": action_a,
			"b": action_b,
			"c": action_c,
			"d": action_d
		}

		# Registrar estadísticas del inicio del tick
		self.start_of_tick_stats = {
			agent.agent_id: {
				"hambre": agent.hambre,
				"sed": agent.sed,
				"salud": agent.salud,
				"location": agent.location,
				"alive": agent.alive
			}
			for agent in self.agents
		}

		# Actuar
		result_a = self.act(self.agent_a, action_a, shout_concept_idx=shout_concept_a)
		result_b = self.act(self.agent_b, action_b, shout_concept_idx=shout_concept_b)
		result_c = self.act(self.agent_c, action_c, shout_concept_idx=shout_concept_c)
		
		result_d = None
		if self.agent_d_was_alive:
			result_d = self.act(self.agent_d, action_d, shout_concept_idx=shout_concept_d)

		results = {"a": result_a, "b": result_b, "c": result_c}
		if self.agent_d_was_alive:
			results["d"] = result_d

		coop_bonuses = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}

		# --- Resolución de la Reproducción Soberana (Fase 7) ---
		reproduction_event = None
		for res in [result_a, result_b, result_c]:
			if res.get("success") and res.get("reproduction_event"):
				reproduction_event = res.get("reproduction_event")
				break

		if reproduction_event:
			self.agent_d.alive = True
			self.agent_d.location = self.agent_a.location # nace en el refugio de los padres
			self.agent_d.hambre = 60.0
			self.agent_d.sed = 80.0
			self.agent_d.salud = 100.0
			self.agent_d.energia = 80.0
			self.agent_d.previous_location = None
			self.agent_d.nav_target = None
			self.agent_d.received_signal = None
			self.agent_d.signal_from = None

			# Herencia genética de habilidades (Fase 7)
			p1 = next(ag for ag in self.agents if ag.agent_id == reproduction_event["parent_a"])
			p2 = next(ag for ag in self.agents if ag.agent_id == reproduction_event["parent_b"])
			
			union_skills = list(set(p1.learned_skills + p2.learned_skills))
			if len(union_skills) > 0:
				inherited_count = min(2, len(union_skills))
				self.agent_d.learned_skills = list(self.rng.sample(union_skills, inherited_count))
			else:
				self.agent_d.learned_skills = []

			# Ancho del hijo: promedio redondeado a múltiplo de 8
			avg_width = (p1.network_width + p2.network_width) // 2
			self.agent_d.network_width = ((avg_width + 7) // 8) * 8
			self.agent_d.pending_growth_penalty = False
			self.agent_d.teaching_buffer = []

			# Marcar en los resultados de los padres que nació Domi para la instanciación PPO
			result_a["born_child"] = {
				"parent_a": reproduction_event["parent_a"],
				"parent_b": reproduction_event["parent_b"],
				"skills": list(self.agent_d.learned_skills),
				"width": self.agent_d.network_width
			}
			result_b["born_child"] = result_a["born_child"]
			result_c["born_child"] = result_a["born_child"]

			print(f"  👶 ¡Nace DOMI (D)! Padres: {p1.agent_id.upper()} y {p2.agent_id.upper()} | Skills: {self.agent_d.learned_skills} | Width: {self.agent_d.network_width}")

		# --- Resolución de la Caza Cooperativa ---
		if self.prey_location:
			# Excluir de la caza a quien no posea la habilidad "caza" (Fase 6)
			hunters = [
				agent for agent in self.agents
				if agent.alive and agent.location == self.prey_location and "caza" in agent.learned_skills and (
					(agent.agent_id == "a" and action_a == "luchar") or
					(agent.agent_id == "b" and action_b == "luchar") or
					(agent.agent_id == "c" and action_c == "luchar") or
					(agent.agent_id == "d" and action_d == "luchar")
				)
			]
			
			# Buscar si hay un cazador solo con lanza
			solo_spear_hunter = None
			for hunter in hunters:
				if getattr(hunter, "tiene_lanza", False):
					solo_spear_hunter = hunter
					break
			
			if solo_spear_hunter:
				self.prey_location = None
				solo_spear_hunter.tiene_lanza = False
				solo_spear_hunter.hambre = min(100.0, solo_spear_hunter.hambre + 50.0)
				solo_spear_hunter.mochila_comida = 1
				coop_bonuses[solo_spear_hunter.agent_id] += 15.0
				res = results[solo_spear_hunter.agent_id]
				res["success"] = True
				res["event"] = "caza individual exitosa de la presa usando la lanza! Comida obtenida."
				res["delta_hambre"] = 50.0
			elif len(hunters) >= 2:
				self.prey_location = None
				for hunter in hunters:
					hunter.hambre = min(100.0, hunter.hambre + 50.0)
					hunter.mochila_comida = 1
					coop_bonuses[hunter.agent_id] += 15.0
					res = results[hunter.agent_id]
					res["success"] = True
					res["event"] = "caza cooperativa exitosa de la presa! Comida obtenida."
					res["delta_hambre"] = 50.0
			elif len(hunters) == 1:
				hunter = hunters[0]
				res = results[hunter.agent_id]
				res["success"] = False
				res["event"] = "caza fallida: se requiere cooperación de otro agente o una lanza."
				res["delta_salud"] = -2.0 * self.predator_damage_multiplier
				hunter.salud += res["delta_salud"]
				hunter.clamp()
				
				if self.rng.random() < 0.50:
					adjacent = COOP_ADJACENCY[self.prey_location]
					self.prey_location = self.rng.choice(adjacent)
					res["event"] += " La presa se asusta y huye."

		# --- Resolución de la Caza del Gran Herbívoro (Stag Hunt) ---
		if self.megaherbivore_location:
			# Agentes con habilidad "caza", con lanza, vivos, en la misma casilla y decidiendo LUCHAR
			mega_hunters = [
				agent for agent in self.agents
				if agent.alive and agent.location == self.megaherbivore_location and "caza" in agent.learned_skills and getattr(agent, "tiene_lanza", False) and (
					(agent.agent_id == "a" and action_a == "luchar") or
					(agent.agent_id == "b" and action_b == "luchar") or
					(agent.agent_id == "c" and action_c == "luchar") or
					(agent.agent_id == "d" and action_d == "luchar")
				)
			]

			# Si hay al menos 3 cazadores con lanzas equipadas, éxito!
			if len(mega_hunters) >= 3:
				loc = self.megaherbivore_location
				self.megaherbivore_location = None
				
				# Da 9.0 unidades de comida al suelo
				self.resource_capacities[loc]["food"] = min(self.resource_capacity * 3.0, self.resource_capacities[loc]["food"] + 9.0)
				
				# Quitar lanzas y aplicar cansancio / bonos cooperativos
				for hunter in mega_hunters:
					hunter.tiene_lanza = False
					hunter.energia = max(0.0, hunter.energia - 15.0)
					coop_bonuses[hunter.agent_id] += 40.0
					res = results[hunter.agent_id]
					res["success"] = True
					res["event"] = "¡CAZA ÉPICA EXITOSA! Abate al Gran Herbívoro junto a sus compañeros usando lanza. 9.0 unidades de carne en el suelo."
					res["delta_energia"] -= 15.0
			elif len(mega_hunters) > 0:
				# Si intentan atacarlo pero son menos de 3, el animal se defiende (contraataque agresivo)
				for hunter in mega_hunters:
					hunter.tiene_lanza = False
					hunter.energia = max(0.0, hunter.energia - 25.0)
					hunter.salud = max(0.0, hunter.salud - 45.0)
					res = results[hunter.agent_id]
					res["success"] = False
					res["event"] = "¡FALLO EN CAZA ÉPICA! El Gran Herbívoro contraataca furiosamente. Lanza rota, salud y energía diezmadas."
					res["delta_salud"] = -45.0
					res["delta_energia"] -= 25.0
					hunter.clamp()
				
				# El animal tiene un 30% de chance de migrar tras ser atacado
				if self.rng.random() < 0.30:
					adjacent = COOP_ADJACENCY[self.megaherbivore_location]
					self.megaherbivore_location = self.rng.choice(adjacent)

		# --- Broadcast de gritos (Half-Duplex) ---
		shouters = []
		for agent in self.agents:
			if not agent.alive or agent.agent_id not in results:
				continue
			res = results[agent.agent_id]
			if res["shouted"] and res["shout_content"]:
				shouters.append((agent, res["shout_content"]))

		for receiver in self.agents:
			if not receiver.alive or receiver.agent_id not in results:
				continue
			res_recv = results[receiver.agent_id]
			if res_recv["shouted"]:
				continue
			
			for sender, shout_content in shouters:
				if sender.agent_id != receiver.agent_id:
					receiver.received_signal = shout_content
					receiver.signal_age = 0
					receiver.signal_from = sender.agent_id
					
					sig_loc_glyph = shout_content[0]
					candidates = _GLYPH_TO_LOCATIONS.get(sig_loc_glyph, [])
					if candidates:
						receiver.nav_target = candidates[0]
					
					model = receiver.companion_model[sender.agent_id]
					if candidates:
						model["location"] = candidates[0]
					
					sig_what_glyph = shout_content[1]
					what_name = self._what_name(sig_what_glyph)
					if what_name in EMOTION_INDEX:
						model["emotion_name"] = what_name
						if what_name == "hambre":
							model["hambre"] = 15.0
						elif what_name == "dolor":
							model["salud"] = 25.0
						elif what_name == "tristeza":
							model["energia"] = 15.0
					elif what_name == "comida":
						for loc in candidates:
							receiver.map_knowledge[loc] = {
								"food": self.resource_capacity,
								"water": receiver.map_knowledge[loc].get("water", 0.0),
								"last_updated": self.world_tick
							}
					elif what_name == "agua":
						for loc in candidates:
							receiver.map_knowledge[loc] = {
								"food": receiver.map_knowledge[loc].get("food", 0.0),
								"water": self.resource_capacity,
								"last_updated": self.world_tick
							}
					elif what_name == "grupo":
						model["emotion_name"] = "grupo"
					elif what_name == "enseñar":
						model["emotion_name"] = "enseñar"
					elif what_name == "árbol":
						# Señal de caza del Gran Herbívoro (Megafauna)
						# Si el receptor tiene lanza y tiene la habilidad caza, acude a ayudar.
						# Si no, ignora (no acude al nav_target).
						model["emotion_name"] = "árbol"
						if not (getattr(receiver, "tiene_lanza", False) and "caza" in receiver.learned_skills):
							receiver.nav_target = None
					break

		# ── Teoría de la Mente: Decaimiento Metabólico Pesimista ──
		for agent in self.agents:
			if not agent.alive:
				continue
			for other_id, model in agent.companion_model.items():
				other_agent = next((ag for ag in self.agents if ag.agent_id == other_id), None)
				if other_agent and not other_agent.alive:
					continue
				model["hambre"] = max(0.0, model.get("hambre", 60.0) - 0.2)
				model["sed"] = max(0.0, model.get("sed", 80.0) - 1.0)
				model["energia"] = max(0.0, model.get("energia", 80.0) - 2.0)
				model["aburrimiento"] = min(100.0, model.get("aburrimiento", 0.0) + 2.0)
				
				delta_s = 0.0
				if model["hambre"] <= 0.0:
					delta_s -= 15.0
				if model["sed"] <= 0.0:
					delta_s -= 25.0
				model["salud"] = max(0.0, model.get("salud", 100.0) + delta_s * self.predator_damage_multiplier)
				
				# Evitar sobrescribir estados de comunicación/intención especiales
				if model.get("emotion_name") in ("grupo", "enseñar"):
					pass
				elif model["hambre"] < 20:
					model["emotion_name"] = "hambre"
				elif model["salud"] < 30 or model["sed"] < 20:
					model["emotion_name"] = "dolor"
				elif model["energia"] < 20:
					model["emotion_name"] = "tristeza"
				else:
					model["emotion_name"] = "ira"

		# ── Sincronización Física (Meetup) ──
		for i, agent_x in enumerate(self.agents):
			for j, agent_y in enumerate(self.agents):
				if i < j and agent_x.alive and agent_y.alive and agent_x.location == agent_y.location:
					for loc in COOP_LOCATION_NAMES:
						t_x = agent_x.map_knowledge[loc]["last_updated"]
						t_y = agent_y.map_knowledge[loc]["last_updated"]
						if t_x > t_y:
							agent_y.map_knowledge[loc] = agent_x.map_knowledge[loc].copy()
						elif t_y > t_x:
							agent_x.map_knowledge[loc] = agent_y.map_knowledge[loc].copy()
							
					agent_x.companion_model[agent_y.agent_id].update({
						"location": agent_y.location,
						"hambre": agent_y.hambre,
						"sed": agent_y.sed,
						"salud": agent_y.salud,
						"energia": agent_y.energia,
						"aburrimiento": agent_y.aburrimiento,
						"emotion_name": agent_y.emotion_name,
						"last_updated": self.world_tick
					})
					agent_y.companion_model[agent_x.agent_id].update({
						"location": agent_x.location,
						"hambre": agent_x.hambre,
						"sed": agent_x.sed,
						"salud": agent_x.salud,
						"energia": agent_x.energia,
						"aburrimiento": agent_x.aburrimiento,
						"emotion_name": agent_x.emotion_name,
						"last_updated": self.world_tick
					})

		# ── Planificación del Rescate Predictivo (nav_target) ──
		for agent in self.agents:
			if not agent.alive:
				continue
			has_rescue_resource = (agent.mochila_comida > 0 or agent.mochila_agua > 0)
			if has_rescue_resource:
				critical_companion = None
				min_val = 999.0
				for other_id, model in agent.companion_model.items():
					other_agent = next((ag for ag in self.agents if ag.agent_id == other_id), None)
					if other_agent and not other_agent.alive:
						continue
					h_est = model.get("hambre", 60.0)
					sed_est = model.get("sed", 80.0)
					s_est = model.get("salud", 100.0)
					danger_metric = min(h_est, sed_est, s_est)
					if danger_metric < 40.0 and danger_metric < min_val:
						min_val = danger_metric
						critical_companion = model
				
				if critical_companion:
					pred_loc = critical_companion.get("location")
					if pred_loc and pred_loc != agent.location:
						agent.nav_target = pred_loc

		# ── Compartir recursos directamente (Rescate Altruista Directo) ──
		for agent in self.agents:
			if not agent.alive or agent.agent_id not in results:
				continue
			res = results[agent.agent_id]
			if res["success"] and res.get("shared_resource"):
				receiver_id = res.get("shared_with")
				receiver = next((r for r in self.agents if r.agent_id == receiver_id), None)
				if receiver_id and receiver:
					# Evitar exploitation de trading inútil entre agentes saciados
					receiver_need = False
					if res["shared_resource"] == "agua" and receiver.sed < 60.0 or res["shared_resource"] == "comida" and receiver.hambre < 60.0:
						receiver_need = True
					
					if receiver_need:
						coop_bonuses[agent.agent_id] += 10.0
						coop_bonuses[receiver_id] += 10.0
					else:
						# Recompensa minúscula para evitar farming
						coop_bonuses[agent.agent_id] += 0.1
						coop_bonuses[receiver_id] += 0.1

		# ── Reaccionar a la señal (Guía cooperativo) ──
		for receiver in self.agents:
			if not receiver.alive or receiver.agent_id not in results:
				continue
			if receiver.signal_from and receiver.received_signal:
				sender_id = receiver.signal_from
				res_recv = results[receiver.agent_id]
				R = receiver.signal_retrievability
				
				if res_recv["success"] and res_recv.get("delta_hambre", 0) > 0:
					emotion_amp = 1.5 if receiver.hambre < 30 else 1.0
					coop_bonuses[sender_id] += R * emotion_amp * 5.0
				elif res_recv.get("delta_salud", 0) < 0:
					coop_bonuses[sender_id] += -3.0 * R
				elif receiver.location == receiver.nav_target:
					caps = self.resource_capacities[receiver.location]
					has_food = (caps["food"] >= 1.0)
					has_water = (caps["water"] >= 1.0)
					if not has_food and not has_water:
						coop_bonuses[sender_id] += -5.0 * R

		# Decoupled fate: si un agente muere, los otros supervivientes sufren un debuff de tristeza por 72 ticks
		just_died_any = False
		for agent in self.agents:
			was_alive = self.start_of_tick_stats.get(agent.agent_id, {}).get("alive", False)
			if was_alive and not agent.alive:
				just_died_any = True
				
		if just_died_any:
			for other_agent in self.agents:
				if other_agent.alive:
					other_agent.debuff_tristeza_ticks = 72

		# food location as the eligible location with highest capacity
		food_locs_sorted = sorted(FOOD_ELIGIBLE, key=lambda l: self.resource_capacities[l]["food"], reverse=True)
		food_loc_highest = food_locs_sorted[0] if food_locs_sorted else None

		world_info = {
			"food_location": food_loc_highest,
			"food_timer": 0,
			"world_tick": self.world_tick,
			"both_alive": all(agent.alive for agent in self.agents if agent.agent_id != "d" or self.agent_d_was_alive),
			"coop_bonus_a": coop_bonuses["a"],
			"coop_bonus_b": coop_bonuses["b"],
			"coop_bonus_c": coop_bonuses["c"],
			"coop_bonus_d": coop_bonuses["d"],
		}

		if self.agent_d_was_alive:
			world_info["result_d"] = result_d

		return result_a, result_b, result_c, world_info

	def get_reward(self, agent: CoopAgentState, result: dict) -> float:
		"""Reward para un agente individual."""
		# Si está muerto, el reward es 0
		if not agent.alive:
			return 0.0

		# Bono básico de supervivencia por cada tick vivo (evita que prefieran morir temprano)
		reward = 1.0

		# Penalización por K.O. / Desmayo (caída en homeostasis)
		if getattr(agent, "just_fell_unconscious", False):
			reward -= 50.0
			agent.just_fell_unconscious = False

		# Penalización por inactividad prolongada en K.O.
		if getattr(agent, "debuff_inconsciente_ticks", 0) > 0:
			reward -= 2.0

		# Bono por reanimar a un compañero inconsciente
		if result.get("reanimated_companion", False):
			reward += 20.0

		# Penalización por tristeza (debuff al morir un compañero)
		if agent.debuff_tristeza_ticks > 0:
			reward -= 0.5

		# Pain: necesidades no cubiertas
		if agent.hambre < 30:
			reward -= 0.4 * (30 - agent.hambre) / 30
		if agent.sed < 30:
			reward -= 0.4 * (30 - agent.sed) / 30
		if agent.salud < 40:
			reward -= 0.6 * (40 - agent.salud) / 40
		if agent.energia < 15:
			reward -= 0.2 * (15 - agent.energia) / 15
		
		# Aburrimiento: penalización por inactividad
		if agent.aburrimiento > 50.0:
			reward -= 0.5 * (agent.aburrimiento - 50.0) / 50.0

		# Pena por inactividad si tiene hambre/sed
		if (agent.hambre < 50.0 or agent.sed < 50.0) and agent.location == agent.previous_location:
			if not (result.get("success") and (result.get("delta_hambre", 0.0) > 0.0 or result.get("delta_sed", 0.0) > 0.0)):
				reward -= 0.5

		# Penalización por dormir inútilmente con alta energía
		if agent.last_action == "dormir":
			energy_before = agent.energia - result.get("delta_energia", 0.0)
			if energy_before > 60.0:
				reward -= 1.0

		# Penalizaciones por intentar acciones de forma fallida o inútil (gastar ticks sin éxito)
		if not result.get("success", False):
			if result.get("tech_error"):
				reward -= 0.5
			elif result.get("physical_error"):
				reward -= 0.3
			elif result.get("tried_skill"):
				exp_acum = agent.skill_experience.get(result["tried_skill"], 0.0)
				reward += 0.2 * max(0.0, 1.0 - exp_acum / 10.0)
			elif agent.last_action in ["comer", "beber", "luchar", "dar", "enseñar", "aprender", "reproducir", "fabricar", "construir", "encender"]:
				reward -= 0.3
				
				# Penalización específica por intentar consumir recursos de fuentes agotadas
				loc_data = COOP_LOCATIONS[agent.location]
				caps = self.resource_capacities[agent.location]
				if agent.last_action == "comer" and agent.mochila_comida == 0:
					if (loc_data["food_eligible"] or loc_data.get("fish_eligible")) and caps["food"] < 1.0:
						reward -= 2.0
				elif agent.last_action == "beber" and agent.mochila_agua == 0 and loc_data["water_available"] and caps["water"] < 1.0:
					reward -= 2.0

		# Penalización por comer/beber inútilmente cuando se está saciado (evita loops estáticos en recursos)
		if result.get("success", False) and agent.last_action in ["comer", "beber"]:
			start_stats = self.start_of_tick_stats.get(agent.agent_id) if hasattr(self, "start_of_tick_stats") else None
			hambre_before = start_stats["hambre"] if start_stats else agent.hambre
			sed_before = start_stats["sed"] if start_stats else agent.sed
			if agent.last_action == "beber" and sed_before >= 90.0:
				reward -= 1.5
			if agent.last_action == "comer" and hambre_before >= 90.0:
				reward -= 1.5

		# Progress: Recompensas basadas en recursos recuperados efectivamente (evita farming al estar lleno)
		if result.get("success", False):
			start_stats = self.start_of_tick_stats.get(agent.agent_id) if hasattr(self, "start_of_tick_stats") else None
			hambre_before = start_stats["hambre"] if start_stats else agent.hambre
			sed_before = start_stats["sed"] if start_stats else agent.sed
			salud_before = start_stats["salud"] if start_stats else agent.salud

			hambre_replenished = max(0.0, agent.hambre - hambre_before)
			sed_replenished = max(0.0, agent.sed - sed_before)
			salud_replenished = max(0.0, agent.salud - salud_before)

			if hambre_replenished > 0.0:
				# Horquilla preventiva de comida [50, 75]
				factor = 1.0 if hambre_before <= 75.0 else max(0.0, 1.0 - (hambre_before - 75.0) / 25.0)
				reward += 1.5 * factor   # comer = muy valioso (comida escasa, máx +1.5)

			if sed_replenished > 0.0:
				# Horquilla preventiva de agua [50, 75]
				factor = 1.0 if sed_before <= 75.0 else max(0.0, 1.0 - (sed_before - 75.0) / 25.0)
				reward += 1.5 * factor     # beber = muy valioso (agua escasa, máx +1.5)

			if salud_replenished > 0.0:
				reward += 0.3 * (salud_replenished / 5.0)      # curarse (máx +0.3)

			if result.get("delta_energia", 0) > 5:
				energy_before = agent.energia - result.get("delta_energia", 0.0)
				if not (agent.last_action == "dormir" and energy_before > 60.0):
					# Horquilla preventiva de energía [50, 75]
					factor = 1.0 if energy_before <= 75.0 else max(0.0, 1.0 - (energy_before - 75.0) / 25.0)
					reward += 0.2 * factor

		# Penalty por daño
		if result.get("delta_salud", 0) < -10:
			reward -= 0.5

		# Reset cognitivo (curiosidad) para localizaciones desactualizadas
		for l, info in agent.map_knowledge.items():
			if self.world_tick - info.get("last_updated", 0) > 35:
				if COOP_LOCATIONS[l]["food_eligible"] or COOP_LOCATIONS[l].get("fish_eligible"):
					info["food"] = 5.0
				if COOP_LOCATIONS[l]["water_available"]:
					info["water"] = 5.0

		# ── Potential-Based Reward Shaping (PBRS) Cognitivo ──
		# Solo se aplica si el agente cambió de posición en este tick
		if agent.previous_location and agent.location != agent.previous_location:
			shaping_reward = 0.0
			
			# 1. Guiar hacia objetivos de navegación (auxilio/gritos recibidos)
			if agent.nav_target:
				d_curr = _bfs_distance(agent.location, agent.nav_target)
				d_prev = _bfs_distance(agent.previous_location, agent.nav_target)
				shaping_reward += (d_prev - d_curr) * 0.4  # +0.4 por acercarse, -0.4 por alejarse
			else:
				# 2. Si no hay target de navegación, guiar hacia recursos conocidos en su mapa cognitivo
				# Guiar hacia comida si tiene hambre
				if agent.hambre < 70:
					known_food_locs = [loc for loc, info in agent.map_knowledge.items() if info.get("food", 0.0) >= 1.0]
					if known_food_locs:
						d_curr = min(_bfs_distance(agent.location, loc) for loc in known_food_locs)
						d_prev = min(_bfs_distance(agent.previous_location, loc) for loc in known_food_locs)
						shaping_reward += (d_prev - d_curr) * 0.2
						
				# Guiar hacia agua si tiene sed
				if agent.sed < 70:
					known_water_locs = [loc for loc, info in agent.map_knowledge.items() if info.get("water", 0.0) >= 1.0]
					if known_water_locs:
						d_curr = min(_bfs_distance(agent.location, loc) for loc in known_water_locs)
						d_prev = min(_bfs_distance(agent.previous_location, loc) for loc in known_water_locs)
						shaping_reward += (d_prev - d_curr) * 0.2

			reward += shaping_reward

		return reward

	def _what_name(self, glyph_idx: int) -> str:
		"""Helper para debug: glyph index → nombre."""
		for name, idx in WORD_INDEX.items():
			if idx == glyph_idx:
				return name
		return "?"

	def get_valid_actions_mask(self, agent: CoopAgentState) -> list[float]:
		"""Retorna una máscara binaria (1.0 = válida, 0.0 = inválida) para las 15 acciones."""
		if not agent.alive:
			return [0.0] * COOP_N_ACTIONS

		# Si está en K.O., sólo puede "dormir" (inactivo recuperando energía)
		if getattr(agent, "debuff_inconsciente_ticks", 0) > 0:
			m = [0.0] * COOP_N_ACTIONS
			m[2] = 1.0
			return m

		mask = [1.0] * COOP_N_ACTIONS
		loc = agent.location
		loc_data = COOP_LOCATIONS[loc]
		caps = self.resource_capacities[loc]

		# 0. Comer: válido si hay comida en mochila o comida/peces elegibles en el suelo
		has_food_in_backpack = (agent.mochila_comida > 0)
		has_food_on_ground = (caps["food"] >= 1.0 and (
			loc_data["food_eligible"] or loc_data.get("fish_eligible")
		))
		if not (has_food_in_backpack or has_food_on_ground):
			mask[0] = 0.0

		# 1. Beber: válido si hay agua en mochila o agua elegible en el suelo
		has_water_in_backpack = (agent.mochila_agua > 0)
		has_water_on_ground = (caps["water"] >= 1.0 and loc_data["water_available"])
		if not (has_water_in_backpack or has_water_on_ground):
			mask[1] = 0.0

		# 2. Dormir: siempre válido (descanso basal)
		mask[2] = 1.0

		# 3. Mover: siempre válido
		mask[3] = 1.0

		# 4. Ver: siempre válido
		mask[4] = 1.0

		# 5. Luchar: válido si hay depredador/amenaza o si está el Gran Herbívoro o la presa estándar
		has_prey = (self.prey_location == loc)
		has_mega = (self.megaherbivore_location == loc and getattr(agent, "tiene_lanza", False))
		if not (agent.danger_nearby or has_prey or has_mega):
			mask[5] = 0.0

		# 6. Gritar: siempre válido si comunicación está habilitada, respetando enfriamiento
		if getattr(agent, "shout_cooldown_ticks", 0) > 0:
			mask[6] = 0.0
		else:
			mask[6] = 1.0

		# 7. Dar: válido si mochila comida > 0 (y la han pedido en los últimos 15t) o mochila agua > 0 (y la han pedido en los últimos 15t), y hay otro agente vivo en la misma localización
		others_here = [other for other in self.agents if other.agent_id != agent.agent_id and other.alive and other.location == loc]
		has_requested_food = any(
			self.world_tick - self.last_shout_tick.get((other.agent_id, "comida"), -999) <= 15
			for other in others_here
		)
		has_requested_water = any(
			self.world_tick - self.last_shout_tick.get((other.agent_id, "agua"), -999) <= 15
			for other in others_here
		)
		can_give_food = agent.mochila_comida > 0 and has_requested_food
		can_give_water = agent.mochila_agua > 0 and has_requested_water
		if not ((can_give_food or can_give_water) and others_here):
			mask[7] = 0.0

		# 8. Enseñar: válido si posee habilidades y hay otro agente vivo en la misma loc con slots libres
		has_skills_to_teach = len(agent.learned_skills) > 0
		has_student = any(len(other.learned_skills) < other.max_slots for other in others_here)
		if not (has_skills_to_teach and has_student):
			mask[8] = 0.0

		# 9. Aprender: válido si hay otro agente vivo enseñando en la misma loc y el agente tiene slots libres
		# (Nota: durante el muestreo simultáneo, asumimos válido si hay compañeros presentes y el agente tiene slots libres)
		has_slots = len(agent.learned_skills) < agent.max_slots
		if not (has_slots and others_here):
			mask[9] = 0.0

		# 10. Reproducir: válido si está en refugio (cueva/ruinas), ambos con comida/sed >= 40, compañeros presentes
		is_shelter = loc_data.get("storm_shelter", False) or loc in ["cueva", "ruinas"]
		if not (is_shelter and others_here and agent.hambre >= 40.0 and agent.sed >= 40.0):
			mask[10] = 0.0

		# 11. Fabricar: válido si tiene materiales (1 rama, 1 piedra) y no tiene lanza
		has_materials = (agent.mochila_ramas >= 1 and agent.mochila_piedras >= 1)
		if not (has_materials and not getattr(agent, "tiene_lanza", False)):
			mask[11] = 0.0

		# 12. Construir: válido si tiene materiales (2 ramas, 1 piedra) y la loc no es refugio
		has_const_materials = (agent.mochila_ramas >= 2 and agent.mochila_piedras >= 1)
		is_already_shelter = loc_data.get("storm_shelter", False)
		if not (has_const_materials and not is_already_shelter):
			mask[12] = 0.0

		# 13. Encender: válido si tiene 2 ramas y no hay fuego activo aquí
		has_fire_materials = (agent.mochila_ramas >= 2)
		has_active_fire = (self.fire_locations.get(loc, 0) > 0)
		if not (has_fire_materials and not has_active_fire):
			mask[13] = 0.0

		# 14. Reanimar: válido si hay algún compañero inconsciente (K.O.) en la misma localización
		has_catatonic_companion = any(other for other in others_here if getattr(other, "debuff_inconsciente_ticks", 0) > 0)
		if not has_catatonic_companion:
			mask[14] = 0.0

		return mask
