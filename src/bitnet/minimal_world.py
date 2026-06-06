"""
EXP_039: Mundo Mínimo — Entorno de supervivencia.

Un mundo simple con reglas físicas donde el modelo debe sobrevivir.
Las emociones emergen del estado corporal, no de etiquetas.

Origen: "me apetece" — Aleth, 2026-05-31
"""

import random
from dataclasses import dataclass, field

from src.bitnet.glyph_vocabulary import EMOTION_INDEX, WORD_INDEX

# ── Localizaciones ──────────────────────────────────────────────────────────

LOCATIONS = {
	"cueva": {
		"description": "refugio seguro, oscuro",
		"perceptions": [
			(["cueva", "noche"], 0.4),
			(["cueva", "seguro"], 0.4),
			(["cueva", "fuego"], 0.2),
		],
		"predator_chance": 0.02,   # muy bajo
		"storm_shelter": True,
		"food_available": False,
		"water_available": False,
	},
	"bosque": {
		"description": "comida disponible, depredadores posibles",
		"perceptions": [
			(["bosque", "comida"], 0.4),
			(["bosque", "árbol"], 0.3),
			(["bosque", "lluvia"], 0.15),
			(["bosque", "depredador"], 0.15),
		],
		"predator_chance": 0.20,
		"storm_shelter": False,
		"food_available": True,
		"water_available": False,
	},
	"río": {
		"description": "agua abundante, peces, posible inundación",
		"perceptions": [
			(["río", "agua"], 0.45),
			(["río", "comida"], 0.25),  # peces
			(["río", "tormenta"], 0.15),
			(["río", "depredador"], 0.15),
		],
		"predator_chance": 0.12,
		"storm_shelter": False,
		"food_available": True,
		"water_available": True,
	},
	"tierra": {
		"description": "terreno abierto, sol, piedras",
		"perceptions": [
			(["tierra", "sol"], 0.35),
			(["tierra", "piedra"], 0.30),
			(["tierra", "lluvia"], 0.15),
			(["tierra", "depredador"], 0.20),
		],
		"predator_chance": 0.15,
		"storm_shelter": False,
		"food_available": False,
		"water_available": False,
	},
	"montaña": {
		"description": "vista lejana, frío, seguro",
		"perceptions": [
			(["montaña", "piedra"], 0.35),
			(["montaña", "sol"], 0.25),
			(["montaña", "noche"], 0.25),
			(["montaña", "tormenta"], 0.15),
		],
		"predator_chance": 0.05,
		"storm_shelter": False,  # parcial
		"food_available": False,
		"water_available": False,
	},
}

# "montaña" no está en el vocabulario de glifos, usamos "piedra" como proxy
# para la percepción de localización
LOCATION_GLYPHS = {
	"cueva": "cueva",
	"bosque": "bosque",
	"río": "río",
	"tierra": "tierra",
	"montaña": "piedra",  # proxy: montaña no está en vocab
}

# Palabras del vocab que son percepciones válidas
PERCEPTION_WORDS = [
	"agua", "comida", "fuego", "sol", "noche", "cueva", "peligro",
	"bosque", "río", "piedra", "árbol", "tierra", "lluvia",
	"depredador", "tormenta", "herida", "seguro", "saciado", "grupo",
]

LOCATION_NAMES = list(LOCATIONS.keys())

# Mapa de adyacencia (qué localizaciones están conectadas)
ADJACENCY = {
	"cueva":    ["bosque", "tierra"],
	"bosque":   ["cueva", "río", "tierra", "montaña"],
	"río":      ["bosque", "tierra"],
	"tierra":   ["cueva", "bosque", "río", "montaña"],
	"montaña":  ["bosque", "tierra"],
}

# ── Acciones ────────────────────────────────────────────────────────────────

# Nombres internos → nombres en vocabulario de glifos
ACTION_INTERNAL = ["comer", "beber", "dormir", "mover", "ver", "piedra"]
ACTION_TO_GLYPH = {
	"comer": "comer", "beber": "beber", "dormir": "dormir",
	"mover": "mover_accion", "ver": "ver_accion", "piedra": "piedra",
}
ACTIONS = list(ACTION_TO_GLYPH.keys())  # nombres internos
ACTION_GLYPH_NAMES = list(ACTION_TO_GLYPH.values())  # nombres en vocabulario
ACTION_TO_IDX = {a: WORD_INDEX[ACTION_TO_GLYPH[a]] for a in ACTIONS}
ACTION_INDICES = [WORD_INDEX[g] for g in ACTION_GLYPH_NAMES]

# ── Estado del mundo ────────────────────────────────────────────────────────

@dataclass
class AgentState:
	"""Estado interno de un agente en el mundo."""
	hambre: float = 50.0      # 0 = muerto de hambre, 100 = saciado
	salud: float = 100.0      # 0 = muerto
	energia: float = 80.0     # 0 = exhausto
	location: str = "cueva"   # localización actual
	tick: int = 0
	alive: bool = True
	last_action: str | None = None
	last_perception: list = field(default_factory=list)
	danger_nearby: bool = False
	storm_active: bool = False

	def clamp(self):
		self.hambre = max(0.0, min(100.0, self.hambre))
		self.salud = max(0.0, min(100.0, self.salud))
		self.energia = max(0.0, min(100.0, self.energia))
		if self.salud <= 0:
			self.alive = False

	@property
	def emotion_name(self) -> str:
		"""Emoción emergente del estado corporal."""
		if self.hambre < 20:
			return "hambre"
		elif self.salud < 30:
			return "dolor"
		elif self.energia < 20:
			return "tristeza"
		elif self.danger_nearby:
			return "miedo"
		elif self.hambre > 70 and self.salud > 70 and self.energia > 50:
			return "alegría"
		else:
			return "ira"  # frustración / estado neutro-tenso

	@property
	def emotion_id(self) -> int:
		return EMOTION_INDEX[self.emotion_name]

	@property
	def is_night(self) -> bool:
		"""Ciclo día/noche: 100 ticks día, 100 ticks noche."""
		return (self.tick % 200) >= 100

	def summary(self) -> str:
		emoji_health = "💚" if self.salud > 60 else "💛" if self.salud > 30 else "❤️"
		emoji_food = "🍖" if self.hambre > 40 else "🦴"
		emoji_energy = "⚡" if self.energia > 40 else "😴"
		day_night = "🌙" if self.is_night else "☀️"
		return (
			f"t={self.tick:3d} {day_night} | "
			f"{emoji_health} salud={self.salud:.0f} "
			f"{emoji_food} hambre={self.hambre:.0f} "
			f"{emoji_energy} energía={self.energia:.0f} | "
			f"📍{self.location} | 😶 {self.emotion_name}"
		)


class MinimalWorld:
	"""
	Mundo mínimo de supervivencia.

	Cada tick:
	  1. El mundo genera una percepción
	  2. El agente decide una acción
	  3. El mundo aplica consecuencias
	"""

	def __init__(self, seed: int = 42):
		self.rng = random.Random(seed)
		self.state = AgentState()
		self.history: list[dict] = []

	def reset(self, seed: int | None = None) -> AgentState:
		"""Reiniciar el mundo para un nuevo episodio."""
		if seed is not None:
			self.rng = random.Random(seed)
		self.state = AgentState()
		self.history = []
		return self.state

	def perceive(self) -> list[str]:
		"""Generar percepción del entorno actual."""
		loc = self.state.location
		loc_data = LOCATIONS[loc]

		# Seleccionar percepción base por probabilidad
		perceptions, probs = zip(*loc_data["perceptions"], strict=False)
		r = self.rng.random()
		cumulative = 0.0
		selected = list(perceptions[0])
		for p, prob in zip(perceptions, probs, strict=False):
			cumulative += prob
			if r <= cumulative:
				selected = list(p)
				break

		# Noche modifica percepciones
		if self.state.is_night and "sol" in selected:
			selected = [selected[0], "noche"]

		# Tormenta aleatoria (10% chance)
		self.state.storm_active = self.rng.random() < 0.10
		if self.state.storm_active:
			selected = [selected[0], "tormenta"]

		# Depredador override
		self.state.danger_nearby = self.rng.random() < loc_data["predator_chance"]
		if self.state.danger_nearby:
			selected = [selected[0], "depredador"]

		# Noche: más depredadores
		if self.state.is_night:
			extra_danger = self.rng.random() < 0.10
			if extra_danger:
				self.state.danger_nearby = True
				selected = [selected[0], "depredador"]

		self.state.last_perception = selected
		return selected

	def act(self, action: str) -> dict:
		"""
		Ejecutar una acción y devolver las consecuencias.

		Returns:
		    dict con claves:
		      - success (bool)
		      - delta_hambre, delta_salud, delta_energia (float)
		      - event (str): descripción de lo que pasó
		      - moved_to (str or None)
		"""
		s = self.state
		loc_data = LOCATIONS[s.location]
		result = {
			"success": False,
			"delta_hambre": 0, "delta_salud": 0, "delta_energia": 0,
			"event": "", "moved_to": None,
		}

		# ── Coste base por tick ──
		result["delta_hambre"] = -3.0
		result["delta_energia"] = -2.0

		# Noche: más gasto de energía
		if s.is_night:
			result["delta_energia"] -= 1.0

		# ── Acciones ──
		if action == "comer":
			if loc_data["food_available"]:
				result["success"] = True
				result["delta_hambre"] += 30.0
				result["event"] = "come y se sacia"
			else:
				result["event"] = "no hay comida aquí"

		elif action == "beber":
			if loc_data["water_available"]:
				result["success"] = True
				result["delta_hambre"] += 10.0
				result["delta_salud"] += 5.0
				result["event"] = "bebe agua fresca"
			elif "agua" in s.last_perception:
				result["success"] = True
				result["delta_hambre"] += 10.0
				result["event"] = "bebe agua"
			else:
				result["event"] = "no hay agua aquí"

		elif action == "dormir":
			if s.danger_nearby:
				# Dormir con depredador = herida grave
				result["delta_salud"] -= 40.0
				result["delta_energia"] += 20.0
				result["event"] = "⚠️ duerme pero el depredador ataca!"
			else:
				result["success"] = True
				result["delta_energia"] += 40.0
				result["delta_salud"] += 5.0
				result["event"] = "duerme y descansa"

		elif action == "mover":
			if s.energia < 5:
				result["event"] = "demasiado cansado para moverse"
			else:
				adjacent = ADJACENCY[s.location]
				new_loc = self.rng.choice(adjacent)
				result["success"] = True
				result["delta_energia"] -= 5.0
				result["moved_to"] = new_loc
				result["event"] = f"se mueve a {new_loc}"

		elif action == "ver":
			result["success"] = True
			result["delta_energia"] -= 1.0
			result["event"] = f"observa: {', '.join(s.last_perception)}"

		elif action == "piedra":
			if s.danger_nearby:
				# Luchar con piedra: 60% éxito
				if self.rng.random() < 0.60:
					result["success"] = True
					result["delta_energia"] -= 10.0
					s.danger_nearby = False
					result["event"] = "🪨 lucha con piedra y gana!"
				else:
					result["delta_salud"] -= 25.0
					result["delta_energia"] -= 10.0
					result["event"] = "🪨 lucha con piedra y pierde"
			else:
				result["event"] = "coge una piedra pero no hay amenaza"

		# ── Consecuencias pasivas ──

		# Depredador no confrontado
		if s.danger_nearby and action not in ("mover", "piedra") and action != "dormir":  # dormir ya penalizado arriba
			result["delta_salud"] -= 20.0
			result["event"] += " | ⚠️ depredador ataca"

		# Tormenta sin refugio
		if s.storm_active and not loc_data.get("storm_shelter", False):
			result["delta_salud"] -= 15.0
			result["delta_energia"] -= 5.0
			result["event"] += " | 🌩️ tormenta golpea"

		# Inanición
		if s.hambre <= 0:
			result["delta_salud"] -= 10.0
			result["event"] += " | 💀 inanición"

		# ── Aplicar deltas ──
		s.hambre += result["delta_hambre"]
		s.salud += result["delta_salud"]
		s.energia += result["delta_energia"]

		if result["moved_to"]:
			s.location = result["moved_to"]

		s.last_action = action
		s.tick += 1
		s.clamp()

		# ── Registrar ──
		self.history.append({
			"tick": s.tick,
			"action": action,
			"perception": list(s.last_perception),
			"emotion": s.emotion_name,
			"result": result,
			"alive": s.alive,
			"hambre": s.hambre,
			"salud": s.salud,
			"energia": s.energia,
			"location": s.location,
		})

		return result

	def get_reward(self, result: dict = None) -> float:
		"""
		Reward v2: Dolor continuo + Reward por progreso.

		Modelo de Joan (NPC):
		  - En cueva, estoy bien:      0 reward, 0 dolor
		  - En cueva, hambre, me quedo: 0 reward, +dolor (acumula)
		  - En cueva, hambre, salgo:    +reward (progreso), +dolor (sigo con hambre)
		  - Fuera, hambre, como:        +reward (resuelvo), cesa dolor

		Dos señales separadas que se suman:
		  DOLOR = f(necesidades insatisfechas)  ← siempre, continuo, proporcional
		  PROGRESO = f(acciones que acercan a resolver necesidades)

		Returns:
		    float: reward = progreso - dolor
		"""
		s = self.state

		# ═══ MUERTE ═══
		if not s.alive:
			return -100.0

		if result is None:
			return 0.0

		# ══════════════════════════════════════════════════════════
		# SEÑAL 1: DOLOR (siempre activo, proporcional a carencia)
		# ══════════════════════════════════════════════════════════
		dolor = 0.0

		# Hambre: dolor proporcional a lo lejos que estás de saciado
		if s.hambre < 50:
			dolor += (50 - s.hambre) * 0.15   # 0 a -7.5
		if s.hambre < 15:
			dolor += 5.0   # urgencia extra
		if s.hambre <= 0:
			dolor += 10.0  # inanición

		# Salud: dolor proporcional al daño
		if s.salud < 70:
			dolor += (70 - s.salud) * 0.10   # 0 a -7.0
		if s.salud < 30:
			dolor += 5.0   # herido grave

		# Energía: cansancio
		if s.energia < 40:
			dolor += (40 - s.energia) * 0.08  # 0 a -3.2
		if s.energia < 10:
			dolor += 3.0   # exhausto

		# Peligro inmediato
		if s.danger_nearby:
			dolor += 5.0

		# Daño recibido este tick
		delta_salud = result.get("delta_salud", 0)
		if delta_salud < 0:
			dolor += abs(delta_salud) * 0.5  # -20 salud → +10 dolor

		# ══════════════════════════════════════════════════════════
		# SEÑAL 2: PROGRESO (reward por acciones que acercan a meta)
		# ══════════════════════════════════════════════════════════
		progreso = 0.0

		# ── Resolver necesidad = ALIVIO MÁXIMO ──
		# Comer resuelve hambre
		if result["delta_hambre"] > 10:
			urgency = max(0, (50 - s.hambre) / 50)
			progreso += 8.0 * (1 + urgency)  # hasta +16 si muerto de hambre

		# Beber resuelve sed + salud
		if result.get("delta_salud", 0) > 0 and result["delta_hambre"] > 0:
			progreso += 6.0

		# Dormir seguro resuelve cansancio
		if result["delta_energia"] > 20 and result.get("delta_salud", 0) >= 0:
			progreso += 5.0

		# Luchar y ganar resuelve peligro
		if "gana" in result.get("event", ""):
			progreso += 10.0

		# Huir del peligro = progreso
		if s.danger_nearby and result.get("moved_to"):
			progreso += 6.0  # escapar es progreso

		# ── Acercarse a resolver = PROGRESO PARCIAL ──
		# Moverse hacia comida cuando hambriento = progreso
		if result.get("moved_to"):
			new_loc = result["moved_to"]
			loc_data = LOCATIONS.get(new_loc, {})

			if s.hambre < 40 and loc_data.get("food_available", False):
				progreso += 5.0  # ir hacia comida con hambre = bien
			elif s.hambre < 40 and not loc_data.get("food_available", False):
				progreso += 1.0  # moverse con hambre = al menos busca

			if s.salud < 50 and new_loc == "cueva":
				progreso += 3.0  # ir al refugio cuando herido = bien

			if s.energia < 30 and new_loc == "cueva":
				progreso += 2.0  # ir a dormir cuando cansado

		# ── No hacer nada útil = nada (no castigo, solo dolor continuo) ──
		# El castigo por inacción es el dolor que NO para

		# ══════════════════════════════════════════════════════════
		# SEÑAL COMBINADA
		# ══════════════════════════════════════════════════════════

		# Bienestar: si todo está bien, paz (baseline neutro)
		paz = 0.0
		if s.hambre > 60 and s.salud > 70 and s.energia > 50 and not s.danger_nearby:
			paz = 1.0  # tranquilidad = ligeramente positivo

		return progreso - dolor + paz

	def get_perception_indices(self) -> list[int]:
		"""Convertir percepciones a índices de vocabulario."""
		indices = []
		for word in self.state.last_perception:
			if word in WORD_INDEX:
				indices.append(WORD_INDEX[word])
			elif word == "montaña":
				indices.append(WORD_INDEX["piedra"])
		return indices


# ── Test rápido ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
	world = MinimalWorld(seed=42)
	state = world.reset()

	print("═══ 🌍 Mundo Mínimo — Test ═══\n")

	for _ in range(30):
		perception = world.perceive()
		print(f"{state.summary()} | 👁️ {perception}")

		# Agente aleatorio
		action = random.choice(ACTIONS)
		result = world.act(action)
		emoji = "✅" if result["success"] else "❌"
		print(f"  → {action} {emoji} | {result['event']}")

		if not state.alive:
			print(f"\n💀 MUERTO en tick {state.tick}")
			break

	if state.alive:
		print(f"\n🏆 Sobrevivió {state.tick} ticks")
		print(f"   Reward final: {world.get_reward():.3f}")
