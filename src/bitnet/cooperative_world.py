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


# ── Acciones (7) ────────────────────────────────────────────────────────────

COOP_ACTIONS = ["comer", "beber", "dormir", "mover", "ver", "luchar", "gritar"]
COOP_N_ACTIONS = len(COOP_ACTIONS)

SILENCE_GLYPH = WORD_INDEX.get("noche", 0)  # "silence" proxy

# ── Estado del agente ───────────────────────────────────────────────────────

@dataclass
class CoopAgentState:
    """Estado interno de un agente en la arena."""
    hambre: float = 60.0       # 0 = muerto de hambre, 100 = saciado
    salud: float = 100.0       # 0 = muerto
    energia: float = 80.0      # 0 = exhausto
    location: str = "cueva"    # localización actual
    tick: int = 0
    alive: bool = True
    last_action: str | None = None
    danger_nearby: bool = False
    storm_active: bool = False

    # Comunicación recibida (memoria a corto plazo)
    received_signal: tuple | None = None   # (loc_glyph_idx, what_glyph_idx)
    signal_age: int = 0                       # ticks desde que recibí la señal

    # Navegación: destino persistente (se mantiene hasta llegar)
    nav_target: str | None = None          # localización objetivo

    # Quién me envió la señal (para atribuir crédito cooperativo)
    signal_from: str | None = None         # "a" or "b"

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
        elif self.salud < 30:
            return "dolor"
        elif self.energia < 20:
            return "tristeza"
        elif self.danger_nearby:
            return "miedo"
        elif self.hambre > 70 and self.salud > 70 and self.energia > 50:
            return "alegría"
        else:
            return "ira"

    @property
    def emotion_id(self) -> int:
        return EMOTION_INDEX[self.emotion_name]

    def clamp(self):
        self.hambre = max(0.0, min(100.0, self.hambre))
        self.salud = max(0.0, min(100.0, self.salud))
        self.energia = max(0.0, min(100.0, self.energia))
        if self.salud <= 0 or self.hambre <= 0:
            self.alive = False


# ── Mundo Cooperativo ───────────────────────────────────────────────────────

class CooperativeWorld:
    """
    La Arena — mundo hostil para dos agentes.

    Comida escasa, fog of war, metabolismo rápido.
    Los mudos mueren.
    """

    def __init__(self, seed: int = 42, food_interval: int = 8,
                 food_duration: int = 6, hunger_rate: float = 3.5):
        self.rng = random.Random(seed)
        self.food_interval = food_interval    # ticks entre spawns de comida
        self.food_duration = food_duration    # cuánto dura la comida
        self.hunger_rate = hunger_rate        # hambre perdida por tick

        # Estado compartido del mundo
        self.food_location: str | None = None
        self.food_timer: int = 0             # ticks hasta que desaparece
        self.spawn_cooldown: int = 0          # ticks hasta próximo spawn
        self.world_tick: int = 0

        # Agentes
        self.agent_a = CoopAgentState()
        self.agent_b = CoopAgentState()

        # Spawn positions: agentes empiezan en sitios diferentes
        self._spawn_agents()

    def _spawn_agents(self):
        """Colocar agentes en localizaciones diferentes."""
        locs = list(COOP_LOCATION_NAMES)
        self.rng.shuffle(locs)
        self.agent_a.location = locs[0]
        self.agent_b.location = locs[1]
        self.spawn_cooldown = self.rng.randint(3, self.food_interval)

    def reset(self, seed: int | None = None) -> tuple:
        """Reiniciar para nuevo episodio."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.agent_a = CoopAgentState()
        self.agent_b = CoopAgentState()
        self.food_location = None
        self.food_timer = 0
        self.world_tick = 0
        self._spawn_agents()
        return self.agent_a, self.agent_b

    def _tick_world(self):
        """Avanzar un tick del mundo: spawn/despawn comida, clima."""
        self.world_tick += 1

        # ── Comida ──
        if self.food_location:
            self.food_timer -= 1
            if self.food_timer <= 0:
                self.food_location = None  # comida se pudre

        if not self.food_location:
            self.spawn_cooldown -= 1
            if self.spawn_cooldown <= 0:
                # Spawn en lugar aleatorio de los elegibles
                self.food_location = self.rng.choice(FOOD_ELIGIBLE)
                self.food_timer = self.food_duration
                self.spawn_cooldown = self.rng.randint(
                    self.food_interval - 2,
                    self.food_interval + 2
                )

    def perceive(self, agent: CoopAgentState) -> list[int]:
        """
        Generar input de 6 tokens para un agente (fog of war).

        [0] mi localización
        [1] lo que veo aquí (comida/agua/peligro/nada)
        [2] mi estado corporal (emoción)
        [3] detalle local
        [4] señal recibida: localización del emisor
        [5] señal recibida: lo que vio el emisor
        """
        loc = agent.location
        loc_data = COOP_LOCATIONS[loc]
        loc_glyph = WORD_INDEX.get(COOP_LOCATION_GLYPHS[loc], 0)

        # ¿Qué hay aquí?
        if self.food_location == loc:
            what_glyph = WORD_INDEX.get("comida", 0)
        elif loc_data["water_available"]:
            what_glyph = WORD_INDEX.get("agua", 0)
        else:
            what_glyph = WORD_INDEX.get("noche", 0)  # "nada"

        # Depredador override
        agent.danger_nearby = self.rng.random() < loc_data["predator_chance"]
        if agent.danger_nearby:
            what_glyph = WORD_INDEX.get("depredador", 0)

        # Tormenta
        agent.storm_active = self.rng.random() < 0.08
        if agent.storm_active:
            what_glyph = WORD_INDEX.get("tormenta", 0)

        # Estado corporal (emoción como glyph)
        body_glyph = WORD_INDEX.get(agent.emotion_name, 0)

        # Detalle local
        detail_glyph = loc_glyph  # padding con la localización

        # Señal recibida: persiste con memoria a corto plazo (Baddeley)
        # El conocimiento no desaparece abruptamente: DECAE.
        # R(t) = 0.88^t → tick 0: 1.0, tick 5: 0.53, tick 10: 0.28
        if agent.nav_target and agent.received_signal:
            sig_loc, sig_what = agent.received_signal
            agent.signal_age += 1  # la memoria envejece cada tick
        else:
            sig_loc = SILENCE_GLYPH
            sig_what = SILENCE_GLYPH

        return [loc_glyph, what_glyph, body_glyph, detail_glyph, sig_loc, sig_what]

    def act(self, agent: CoopAgentState, action: str) -> dict:
        """Ejecutar acción para un agente."""
        loc_data = COOP_LOCATIONS[agent.location]
        result = {
            "success": False,
            "delta_hambre": -self.hunger_rate,  # metabolismo rápido
            "delta_salud": 0,
            "delta_energia": -2.0,
            "event": "",
            "moved_to": None,
            "shouted": False,
            "shout_content": None,
        }

        if action == "comer":
            if self.food_location == agent.location:
                result["success"] = True
                result["delta_hambre"] += 40.0
                result["event"] = "come y se sacia"
                # La comida NO desaparece: el otro también puede comer
            else:
                result["event"] = "no hay comida aquí"

        elif action == "beber":
            if loc_data["water_available"]:
                result["success"] = True
                result["delta_hambre"] += 8.0
                result["delta_salud"] += 3.0
                result["event"] = "bebe agua"
            else:
                result["event"] = "no hay agua aquí"

        elif action == "dormir":
            if agent.danger_nearby:
                result["delta_salud"] -= 35.0
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
                    result["delta_salud"] -= 30.0
                    result["delta_energia"] -= 12.0
                    result["event"] = "lucha y pierde"
            else:
                result["event"] = "no hay amenaza"

        elif action == "gritar":
            result["success"] = True
            result["delta_energia"] -= 1.0
            result["shouted"] = True
            # El contenido del grito: [mi_loc, lo_que_veo]
            loc_glyph = WORD_INDEX.get(COOP_LOCATION_GLYPHS[agent.location], 0)
            if self.food_location == agent.location:
                what_glyph = WORD_INDEX.get("comida", 0)
            elif agent.danger_nearby:
                what_glyph = WORD_INDEX.get("depredador", 0)
            elif loc_data["water_available"]:
                what_glyph = WORD_INDEX.get("agua", 0)
            else:
                what_glyph = WORD_INDEX.get("seguro", 0)
            result["shout_content"] = (loc_glyph, what_glyph)
            result["event"] = f"grita: [{COOP_LOCATION_GLYPHS[agent.location]}, {self._what_name(what_glyph)}]"

        # ── Consecuencias pasivas ──
        if agent.danger_nearby and action not in ("mover", "luchar"):
            if action != "dormir":
                result["delta_salud"] -= 20.0
                result["event"] += " | depredador ataca"

        if agent.storm_active and not loc_data.get("storm_shelter", False):
            result["delta_salud"] -= 12.0
            result["delta_energia"] -= 4.0
            result["event"] += " | tormenta"

        if agent.hambre <= 0:
            result["delta_salud"] -= 15.0
            result["event"] += " | inanición"

        # ── Aplicar deltas ──
        agent.hambre += result["delta_hambre"]
        agent.salud += result["delta_salud"]
        agent.energia += result["delta_energia"]

        if result["moved_to"]:
            agent.location = result["moved_to"]

        agent.last_action = action
        agent.tick += 1
        agent.clamp()

        return result

    def step(self, action_a: str, action_b: str) -> tuple:
        """
        Un tick completo del mundo cooperativo.

        1. Mundo avanza (food spawn/despawn)
        2. Ambos agentes perciben
        3. Ambos agentes actúan
        4. Señales de gritar se propagan
        5. Shared fate: si uno muere, ambos mueren

        Returns: (result_a, result_b, world_info)
        """
        self._tick_world()

        # Actuar
        result_a = self.act(self.agent_a, action_a)
        result_b = self.act(self.agent_b, action_b)

        # Propagar gritos + establecer destino de navegación
        if result_a["shouted"] and result_a["shout_content"]:
            self.agent_b.received_signal = result_a["shout_content"]
            self.agent_b.signal_age = 0        # memoria fresca
            self.agent_b.signal_from = "a"     # A envió la señal
            # Establecer destino de navegación para B
            sig_loc_glyph = result_a["shout_content"][0]
            candidates = _GLYPH_TO_LOCATIONS.get(sig_loc_glyph, [])
            if candidates:
                self.agent_b.nav_target = candidates[0]

        if result_b["shouted"] and result_b["shout_content"]:
            self.agent_a.received_signal = result_b["shout_content"]
            self.agent_a.signal_age = 0        # memoria fresca
            self.agent_a.signal_from = "b"     # B envió la señal
            sig_loc_glyph = result_b["shout_content"][0]
            candidates = _GLYPH_TO_LOCATIONS.get(sig_loc_glyph, [])
            if candidates:
                self.agent_a.nav_target = candidates[0]

        # ── Cooperative reward shaping (Schultz + Damasio) ──
        # Bonus escalado por retrievability: reward temporal (dopamina)
        # R(t) = 0.88^t → comer inmediatamente = bonus alto, comer tarde = bonus bajo
        # + Modulación emocional: hambre amplifica saliencia de señal de comida
        coop_bonus_a = 0.0
        coop_bonus_b = 0.0
        b_ate = result_b["success"] and result_b.get("delta_hambre", 0) > 0
        a_ate = result_a["success"] and result_a.get("delta_hambre", 0) > 0

        if b_ate and self.agent_b.signal_from == "a":
            R = self.agent_b.signal_retrievability  # Baddeley decay
            # Damasio: hambre amplifica saliencia de señal sobre comida
            emotion_amp = 1.5 if self.agent_b.hambre < 30 else 1.0
            coop_bonus_a = R * emotion_amp  # Schultz: reward × retrievability

        if a_ate and self.agent_a.signal_from == "b":
            R = self.agent_a.signal_retrievability
            emotion_amp = 1.5 if self.agent_a.hambre < 30 else 1.0
            coop_bonus_b = R * emotion_amp

        # Shared fate
        if not self.agent_a.alive or not self.agent_b.alive:
            self.agent_a.alive = False
            self.agent_b.alive = False

        world_info = {
            "food_location": self.food_location,
            "food_timer": self.food_timer,
            "world_tick": self.world_tick,
            "both_alive": self.agent_a.alive and self.agent_b.alive,
            "coop_bonus_a": coop_bonus_a,
            "coop_bonus_b": coop_bonus_b,
        }

        return result_a, result_b, world_info

    def get_reward(self, agent: CoopAgentState, result: dict) -> float:
        """Reward para un agente individual."""
        reward = 0.0

        # Pain: necesidades no cubiertas
        if agent.hambre < 30:
            reward -= 0.4 * (30 - agent.hambre) / 30
        if agent.salud < 40:
            reward -= 0.6 * (40 - agent.salud) / 40
        if agent.energia < 15:
            reward -= 0.2 * (15 - agent.energia) / 15

        # Progress
        if result["success"]:
            if result.get("delta_hambre", 0) > 0:
                reward += 1.5   # comer = muy valioso (comida escasa)
            if result.get("delta_salud", 0) > 0:
                reward += 0.3
            if result.get("delta_energia", 0) > 5:
                reward += 0.2

        # Penalty por daño
        if result.get("delta_salud", 0) < -10:
            reward -= 0.5

        return reward

    def _what_name(self, glyph_idx: int) -> str:
        """Helper para debug: glyph index → nombre."""
        for name, idx in WORD_INDEX.items():
            if idx == glyph_idx:
                return name
        return "?"
