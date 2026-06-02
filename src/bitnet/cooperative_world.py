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

import random
from collections import deque
from dataclasses import dataclass

from src.bitnet.glyph_vocabulary import EMOTION_INDEX, WORD_INDEX

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


# ── Localizaciones ──────────────────────────────────────────────────────────

COOP_LOCATIONS = {
	"cueva": {
		"description": "refugio seguro, sin comida",
		"predator_chance": 0.02,
		"storm_shelter": True,
		"food_eligible": False,
		"water_available": False,
	},
	"lago": {
		"description": "agua abundante, tranquilo",
		"predator_chance": 0.08,
		"storm_shelter": False,
		"food_eligible": False,
		"water_available": True,
	},
	"pradera": {
		"description": "centro del mapa, expuesto",
		"predator_chance": 0.15,
		"storm_shelter": False,
		"food_eligible": False,
		"water_available": False,
	},
	"bosque": {
		"description": "denso, comida posible, peligroso",
		"predator_chance": 0.20,
		"storm_shelter": False,
		"food_eligible": True,
		"water_available": False,
	},
	"montaña": {
		"description": "alto, frío, vista lejana",
		"predator_chance": 0.05,
		"storm_shelter": False,
		"food_eligible": False,
		"water_available": False,
	},
	"valle": {
		"description": "protegido, fértil",
		"predator_chance": 0.10,
		"storm_shelter": False,
		"food_eligible": True,
		"water_available": False,
	},
	"río": {
		"description": "agua y peces, inundaciones",
		"predator_chance": 0.12,
		"storm_shelter": False,
		"food_eligible": True,
		"water_available": True,
	},
	"pantano": {
		"description": "húmedo, peligroso, pero rico",
		"predator_chance": 0.25,
		"storm_shelter": False,
		"food_eligible": True,
		"water_available": True,
	},
	"ruinas": {
		"description": "antiguo refugio, depredadores",
		"predator_chance": 0.30,
		"storm_shelter": True,
		"food_eligible": True,
		"water_available": False,
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


# ── Acciones (7) ────────────────────────────────────────────────────────────

COOP_ACTIONS = ["comer", "beber", "dormir", "mover", "ver", "luchar", "gritar", "dar"]
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

	def __post_init__(self):
		if self.map_knowledge is None:
			self.map_knowledge = {
				loc: {"food": 5.0, "water": 5.0, "last_updated": 0}
				for loc in COOP_LOCATION_NAMES
			}
		if self.companion_model is None:
			self.companion_model = {}

	@property
	def signal_retrievability(self) -> float:
		"""¿Cuánto recuerdo de la señal? (Baddeley/FSRS decay)"""
		if not self.received_signal:
			return 0.0
		return signal_retrievability(self.signal_age)

	@property
	def emotion_name(self) -> str:
		if self.hambre < 20:
			return "hambre"
		elif self.salud < 30 or self.sed < 20:
			return "dolor"
		elif self.energia < 20:
			return "tristeza"
		elif self.danger_nearby:
			return "miedo"
		elif self.hambre > 70 and self.salud > 70 and self.sed > 60 and self.energia > 50:
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
		if self.salud <= 0 or self.hambre <= 0 or self.sed <= 0:
			self.alive = False


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
				prey_spawn_interval: int = 15):
		self.rng = random.Random(seed)
		self.food_interval = food_interval    # ticks entre spawns de comida
		self.food_duration = food_duration    # cuánto dura la comida
		self.hunger_rate = hunger_rate        # hambre perdida por tick
		self.resource_capacity = resource_capacity
		self.resource_recovery = resource_recovery
		self.predator_chance_multiplier = predator_chance_multiplier
		self.predator_damage_multiplier = predator_damage_multiplier
		self.storm_chance_multiplier = storm_chance_multiplier
		self.storm_damage_multiplier = storm_damage_multiplier

		# Estado compartido del mundo
		self.world_tick: int = 0

		# Mecánica de la presa (caza cooperativa)
		self.prey_spawn_interval = prey_spawn_interval
		self.prey_location = None
		self.prey_timer = 0
		self.prey_cooldown = self.rng.randint(5, self.prey_spawn_interval)

		# Capacidades dinámicas de recursos agotables
		self.resource_capacities = {
			loc: {
				"food": self.resource_capacity if loc_data["food_eligible"] else 0.0,
				"water": self.resource_capacity if loc_data["water_available"] else 0.0
			}
			for loc, loc_data in COOP_LOCATIONS.items()
		}

		# Agentes
		self.agent_a = CoopAgentState(agent_id="a")
		self.agent_b = CoopAgentState(agent_id="b")
		self.agent_c = CoopAgentState(agent_id="c")
		self.agents = [self.agent_a, self.agent_b, self.agent_c]

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
		self.spawn_cooldown = self.rng.randint(3, self.food_interval)

	def reset(self, seed: int | None = None) -> tuple:
		"""Reiniciar para nuevo episodio."""
		if seed is not None:
			self.rng = random.Random(seed)
		self.agent_a = CoopAgentState(agent_id="a")
		self.agent_b = CoopAgentState(agent_id="b")
		self.agent_c = CoopAgentState(agent_id="c")
		self.agents = [self.agent_a, self.agent_b, self.agent_c]

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

		self.resource_capacities = {
			loc: {
				"food": self.resource_capacity if loc_data["food_eligible"] else 0.0,
				"water": self.resource_capacity if loc_data["water_available"] else 0.0
			}
			for loc, loc_data in COOP_LOCATIONS.items()
		}
		self.prey_location = None
		self.prey_timer = 0
		self.prey_cooldown = self.rng.randint(5, self.prey_spawn_interval)
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
			if COOP_LOCATIONS[loc]["food_eligible"]:
				caps["food"] = min(self.resource_capacity, caps["food"] + self.resource_recovery)
			if COOP_LOCATIONS[loc]["water_available"]:
				caps["water"] = min(self.resource_capacity, caps["water"] + self.resource_recovery)

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

		# ¿Qué hay aquí? (Presa override: grupo)
		if self.prey_location == loc:
			what_glyph = WORD_INDEX.get("grupo", 0)
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
		if agent.mochila_comida > 0 and agent.mochila_agua > 0:
			detail_glyph = WORD_INDEX.get("grupo", 0)
		elif agent.mochila_comida > 0:
			detail_glyph = WORD_INDEX.get("comida", 0)
		elif agent.mochila_agua > 0:
			detail_glyph = WORD_INDEX.get("agua", 0)
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
		agent.previous_location = agent.location
		loc_data = COOP_LOCATIONS[agent.location]
		result = {
			"success": False,
			"delta_hambre": -self.hunger_rate,  # metabolismo rápido
			"delta_sed": -self.hunger_rate * 2.0,  # sed decae el doble de rápido
			"delta_salud": 0,
			"delta_energia": -2.0,
			"event": "",
			"moved_to": None,
			"shouted": False,
			"shout_content": None,
			"shared_resource": None,
		}

		# Llenar mochila automáticamente si está en un recurso disponible
		caps = self.resource_capacities[agent.location]
		if caps["food"] >= 1.0 and agent.mochila_comida == 0:
			# Hugo (C) no puede recolectar comida
			if agent.agent_id != "c":
				agent.mochila_comida = 1
				caps["food"] -= 1.0
		if caps["water"] >= 1.0 and agent.mochila_agua == 0:
			# Sofy (B) no puede extraer agua
			if agent.agent_id != "b":
				agent.mochila_agua = 1
				caps["water"] -= 1.0

		if action == "comer":
			# Hugo (C) no puede comer del suelo
			if caps["food"] >= 1.0 and agent.agent_id != "c":
				result["success"] = True
				result["delta_hambre"] += 40.0
				result["delta_sed"] -= 3.0  # penalización por digestión
				result["event"] = "come y se sacia (suelo)"
				caps["food"] -= 1.0
			elif agent.mochila_comida > 0:
				agent.mochila_comida = 0
				result["success"] = True
				result["delta_hambre"] += 40.0
				result["delta_sed"] -= 3.0  # penalización por digestión
				result["event"] = "come de su mochila"
			else:
				result["event"] = "no hay comida aquí ni en mochila"
				if agent.agent_id == "c" and caps["food"] >= 1.0:
					result["event"] = "Hugo no sabe recolectar comida del suelo"

		elif action == "beber":
			# Sofy (B) no puede beber del suelo
			if caps["water"] >= 1.0 and agent.agent_id != "b":
				result["success"] = True
				result["delta_sed"] += 50.0
				result["delta_salud"] += 3.0
				result["event"] = "bebe agua (suelo)"
				caps["water"] -= 1.0
			elif agent.mochila_agua > 0:
				agent.mochila_agua = 0
				result["success"] = True
				result["delta_sed"] += 50.0
				result["delta_salud"] += 3.0
				result["event"] = "bebe agua de su mochila"
			else:
				result["event"] = "no hay agua aquí ni en mochila"
				if agent.agent_id == "b" and caps["water"] >= 1.0:
					result["event"] = "Sofy no sabe extraer agua del suelo"

		elif action == "dormir":
			# Tasa metabólica basal: desgaste a la mitad durante el sueño
			result["delta_hambre"] = -self.hunger_rate * 0.5
			result["delta_sed"] = -self.hunger_rate * 2.0 * 0.5
			if agent.danger_nearby:
				result["delta_salud"] -= 35.0 * self.predator_damage_multiplier
				result["delta_energia"] += 15.0
				result["event"] = "duerme pero depredador ataca!"
			else:
				result["success"] = True
				result["delta_energia"] += 35.0
				result["delta_salud"] += 3.0
				result["event"] = "duerme y descansa"

		elif action == "mover":
			if agent.energia < 5:
				result["event"] = "demasiado cansado para moverse"
			else:
				adjacent = COOP_ADJACENCY[agent.location]

				# Movimiento dirigido: si tiene destino, camina hacia él
				if agent.nav_target and agent.nav_target != agent.location:
					next_step = _bfs_next_step(agent.location, agent.nav_target)
					if next_step and next_step in adjacent:
						new_loc = next_step
					else:
						new_loc = self.rng.choice(adjacent)
				else:
					# Sin destino: exploración aleatoria
					new_loc = self.rng.choice(adjacent)

				result["success"] = True
				result["delta_energia"] -= 4.0
				result["moved_to"] = new_loc
				result["event"] = f"se mueve a {new_loc}"
				if agent.nav_target:
					result["event"] += f" (→{agent.nav_target})"

				# Si llegó al destino, limpiar navegación y señal
				if new_loc == agent.nav_target:
					agent.nav_target = None
					agent.received_signal = None
					agent.signal_from = None

		elif action == "ver":
			result["success"] = True
			result["delta_energia"] -= 1.0
			result["event"] = "observa el entorno"

		elif action == "luchar":
			if agent.danger_nearby:
				if self.rng.random() < 0.55:
					result["success"] = True
					result["delta_energia"] -= 12.0
					agent.danger_nearby = False
					result["event"] = "lucha y gana"
				else:
					result["delta_salud"] -= 30.0 * self.predator_damage_multiplier
					result["delta_energia"] -= 12.0
					result["event"] = "lucha y pierde"
			else:
				result["event"] = "no hay amenaza"

		elif action == "gritar":
			result["success"] = True
			result["delta_energia"] = 0.0  # Coste de energía cero para comunicarse
			result["shouted"] = True
			# El contenido del grito: [mi_loc, lo_que_veo]
			loc_glyph = WORD_INDEX.get(COOP_LOCATION_GLYPHS[agent.location], 0)
			if shout_concept_idx is not None:
				what_glyph = shout_concept_idx
			else:
				caps = self.resource_capacities[agent.location]
				if caps["food"] >= 1.0:
					what_glyph = WORD_INDEX.get("comida", 0)
				elif agent.danger_nearby:
					what_glyph = WORD_INDEX.get("depredador", 0)
				elif caps["water"] >= 1.0:
					what_glyph = WORD_INDEX.get("agua", 0)
				else:
					what_glyph = WORD_INDEX.get("seguro", 0)
			result["shout_content"] = (loc_glyph, what_glyph)
			result["event"] = f"grita: [{COOP_LOCATION_GLYPHS[agent.location]}, {self._what_name(what_glyph)}]"

		elif action == "dar":
			# Compartir recursos de la mochila con cualquier compañero si están juntos
			others = [other for other in self.agents if other.agent_id != agent.agent_id and other.alive]
			transferred = False
			for other_agent in others:
				if other_agent.location == agent.location:
					# 1. Dar agua si el otro tiene la mochila vacía y la necesita (sed < 80)
					if other_agent.mochila_agua == 0 and other_agent.sed < 80 and agent.mochila_agua > 0:
						agent.mochila_agua = 0
						other_agent.mochila_agua = 1
						result["success"] = True
						result["event"] = f"comparte agua de su mochila con {other_agent.agent_id.upper()} (llena mochila)"
						result["shared_resource"] = "agua"
						result["shared_with"] = other_agent.agent_id
						transferred = True
						break
					# 2. Dar comida si el otro tiene la mochila vacía y la necesita (hambre < 80)
					elif other_agent.mochila_comida == 0 and other_agent.hambre < 80 and agent.mochila_comida > 0:
						agent.mochila_comida = 0
						other_agent.mochila_comida = 1
						result["success"] = True
						result["event"] = f"comparte comida de su mochila con {other_agent.agent_id.upper()} (llena mochila)"
						result["shared_resource"] = "comida"
						result["shared_with"] = other_agent.agent_id
						transferred = True
						break

			if not transferred:
				result["event"] = "mochila vacía, compañeros saciados o solos"

		# ── Consecuencias pasivas ──
		if agent.danger_nearby and action not in ("mover", "luchar"):
			if action != "dormir":
				result["delta_salud"] -= 20.0 * self.predator_damage_multiplier
				result["event"] += " | depredador ataca"

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

		# ── Aplicar deltas ──
		agent.hambre += result["delta_hambre"]
		agent.sed += result.get("delta_sed", 0.0)
		agent.salud += result["delta_salud"]
		agent.energia += result["delta_energia"]

		if result["moved_to"]:
			agent.location = result["moved_to"]

		agent.last_action = action
		agent.tick += 1
		agent.clamp()

		return result

	def step(self, action_a: str, action_b: str, action_c: str = "ver",
			shout_concept_a: int = None, shout_concept_b: int = None, shout_concept_c: int = None) -> tuple:
		"""
		Un tick completo del mundo cooperativo a 3 bandas.

		1. Mundo avanza (recuperación + presa)
		2. Los 3 agentes perciben
		3. Los 3 agentes actúan
		4. Resolución de caza cooperativa de presa
		5. Broadcast de gritos Half-Duplex
		6. Sincronización ToM y mapas
		7. Planificación de rescate ToM
		8. Asignación de coop_bonuses por consecuencias

		Returns: (result_a, result_b, result_c, world_info)
		"""
		self._tick_world()

		# Actuar
		result_a = self.act(self.agent_a, action_a, shout_concept_idx=shout_concept_a)
		result_b = self.act(self.agent_b, action_b, shout_concept_idx=shout_concept_b)
		result_c = self.act(self.agent_c, action_c, shout_concept_idx=shout_concept_c)
		results = {"a": result_a, "b": result_b, "c": result_c}

		coop_bonuses = {"a": 0.0, "b": 0.0, "c": 0.0}

		# --- Resolución de la Caza Cooperativa ---
		prey_hunted = False
		if self.prey_location:
			# Nico (A) no sabe cazar presas y queda excluido del recuento de cazadores
			hunters = [
				agent for agent in self.agents
				if agent.alive and agent.location == self.prey_location and (
					(agent.agent_id == "b" and action_b == "luchar") or
					(agent.agent_id == "c" and action_c == "luchar")
				)
			]
			
			if len(hunters) >= 2:
				# Éxito de la caza cooperativa!
				prey_hunted = True
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
				# Fallo de la caza: luchador solitario
				hunter = hunters[0]
				res = results[hunter.agent_id]
				res["success"] = False
				res["event"] = "caza fallida: se requiere cooperación de otro agente."
				res["delta_salud"] = -2.0 * self.predator_damage_multiplier
				hunter.salud += res["delta_salud"]
				hunter.clamp()
				
				# La presa tiene 50% de probabilidad de huir
				if self.rng.random() < 0.50:
					adjacent = COOP_ADJACENCY[self.prey_location]
					self.prey_location = self.rng.choice(adjacent)
					res["event"] += " La presa se asusta y huye."

		# --- Broadcast de gritos (Half-Duplex) ---
		shouters = []
		for agent in self.agents:
			res = results[agent.agent_id]
			if res["shouted"] and res["shout_content"]:
				shouters.append((agent, res["shout_content"]))

		for receiver in self.agents:
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
					break

		# ── Teoría de la Mente: Decaimiento Metabólico Pesimista ──
		for agent in self.agents:
			for other_id, model in agent.companion_model.items():
				model["hambre"] = max(0.0, model.get("hambre", 60.0) - self.hunger_rate)
				model["sed"] = max(0.0, model.get("sed", 80.0) - self.hunger_rate * 2.0)
				model["energia"] = max(0.0, model.get("energia", 80.0) - 2.0)
				
				delta_s = 0.0
				if model["hambre"] <= 0.0:
					delta_s -= 15.0
				if model["sed"] <= 0.0:
					delta_s -= 25.0
				model["salud"] = max(0.0, model.get("salud", 100.0) + delta_s * self.predator_damage_multiplier)
				
				if model["hambre"] < 20:
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
				if i < j and agent_x.alive and agent_y.alive:
					if agent_x.location == agent_y.location:
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
							"emotion_name": agent_y.emotion_name,
							"last_updated": self.world_tick
						})
						agent_y.companion_model[agent_x.agent_id].update({
							"location": agent_x.location,
							"hambre": agent_x.hambre,
							"sed": agent_x.sed,
							"salud": agent_x.salud,
							"energia": agent_x.energia,
							"emotion_name": agent_x.emotion_name,
							"last_updated": self.world_tick
						})

		# ── Planificación del Rescate Predictivo (nav_target) ──
		for agent in self.agents:
			has_rescue_resource = (agent.mochila_comida > 0 or agent.mochila_agua > 0)
			if has_rescue_resource:
				critical_companion = None
				min_val = 999.0
				for other_id, model in agent.companion_model.items():
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
			res = results[agent.agent_id]
			if res["success"] and res.get("shared_resource"):
				receiver_id = res.get("shared_with")
				if receiver_id:
					coop_bonuses[agent.agent_id] += 10.0
					coop_bonuses[receiver_id] += 10.0

		# ── Reaccionar a la señal (Guía cooperativo) ──
		for receiver in self.agents:
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
						coop_bonuses[sender_id] += -1.0 * R

		# Shared fate: si uno muere, la tribu entera muere
		any_dead = any(not agent.alive for agent in self.agents)
		if any_dead:
			for agent in self.agents:
				agent.alive = False

		# food location as the eligible location with highest capacity
		food_locs_sorted = sorted(FOOD_ELIGIBLE, key=lambda l: self.resource_capacities[l]["food"], reverse=True)
		food_loc_highest = food_locs_sorted[0] if food_locs_sorted else None

		world_info = {
			"food_location": food_loc_highest,
			"food_timer": 0,
			"world_tick": self.world_tick,
			"both_alive": all(agent.alive for agent in self.agents),
			"coop_bonus_a": coop_bonuses["a"],
			"coop_bonus_b": coop_bonuses["b"],
			"coop_bonus_c": coop_bonuses["c"],
		}

		return result_a, result_b, result_c, world_info

	def get_reward(self, agent: CoopAgentState, result: dict) -> float:
		"""Reward para un agente individual."""
		reward = 0.0

		# Pain: necesidades no cubiertas
		if agent.hambre < 30:
			reward -= 0.4 * (30 - agent.hambre) / 30
		if agent.sed < 30:
			reward -= 0.4 * (30 - agent.sed) / 30
		if agent.salud < 40:
			reward -= 0.6 * (40 - agent.salud) / 40
		if agent.energia < 15:
			reward -= 0.2 * (15 - agent.energia) / 15

		# Progress
		if result["success"]:
			if result.get("delta_hambre", 0) > 0:
				reward += 1.5   # comer = muy valioso (comida escasa)
			if result.get("delta_sed", 0) > 0:
				reward += 1.5   # beber = muy valioso (agua escasa)
			if result.get("delta_salud", 0) > 0:
				reward += 0.3
			if result.get("delta_energia", 0) > 5:
				reward += 0.2

		# Penalty por daño
		if result.get("delta_salud", 0) < -10:
			reward -= 0.5

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
