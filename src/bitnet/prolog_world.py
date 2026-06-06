"""
prolog_world.py

Entorno de supervivencia integrado con SWI-Prolog via pyswip.
El agente se encuentra con objetos desconocidos y debe consultar a Prolog
para verificar su seguridad antes de consumirlos.
"""

import os
import random
from dataclasses import dataclass

from pyswip import Prolog

from src.bitnet.minimal_world import ADJACENCY, LOCATIONS, AgentState, MinimalWorld


@dataclass
class PrologAgentState(AgentState):
	current_item: str | None = None          # 'objeto_desconocido' o None
	item_is_safe: bool = False               # Determinado por el entorno
	last_prolog_feedback: str | None = None  # 'seguro', 'peligro' o None

class PrologSurvivalWorld(MinimalWorld):
	def __init__(self, seed: int = 42):
		super().__init__(seed)
		self.state = PrologAgentState()
		
		# Inicializar SWI-Prolog
		self.prolog = Prolog()
		base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
		kb_path = os.path.join(base_dir, "src", "bitnet", "kb_supervivencia.pl")
		self.prolog.consult(kb_path)
		
		# Limpiar hechos previos
		self._clear_kb()

	def _clear_kb(self):
		self.prolog.retractall("query_property(_, _)")
		self.prolog.retractall("venenoso(objeto_desconocido)")
		self.prolog.retractall("seguro(objeto_desconocido)")

	def reset(self, seed: int | None = None) -> PrologAgentState:
		if seed is not None:
			self.rng = random.Random(seed)
		self.state = PrologAgentState()
		self.history = []
		self._clear_kb()
		return self.state

	def perceive(self) -> list[str]:
		# Percepción base del MinimalWorld
		perception = super().perceive()
		
		# Si hay un objeto en la ubicación, lo añadimos a la percepción
		if self.state.current_item:
			perception.append("grupo")  # Usamos grupo como proxy del objeto

			
		# Si se realizó una consulta Prolog en el último tick, añadimos el feedback
		if self.state.last_prolog_feedback:
			perception.append(self.state.last_prolog_feedback)
			
		# Forzar a 4 tokens usando el último elemento o "tierra" como relleno
		while len(perception) < 4:
			perception.append(perception[-1] if perception else "tierra")
		return perception[:4]

	def act(self, action: str) -> dict:
		s = self.state
		result = {
			"success": False,
			"delta_hambre": -3.0, "delta_salud": 0.0, "delta_energia": -2.0,
			"event": "", "moved_to": None,
		}

		if s.is_night:
			result["delta_energia"] -= 1.0

		# ── ACCIONES ──
		if action == "ver":
			# ACCIÓN DE CONSULTA A PROLOG
			if s.current_item == "objeto_desconocido":
				result["success"] = True
				result["delta_energia"] -= 1.0
				
				# Configurar hechos en la base de conocimientos dinámicamente
				self._clear_kb()
				if s.item_is_safe:
					self.prolog.assertz("seguro(objeto_desconocido)")
				else:
					self.prolog.assertz("venenoso(objeto_desconocido)")
				
				# Consultar a Prolog si es comestible
				query_res = list(self.prolog.query("comestible(objeto_desconocido)"))
				
				if query_res:
					s.last_prolog_feedback = "seguro"
					result["event"] = "Prolog unifica: objeto es seguro"
				else:
					s.last_prolog_feedback = "peligro"
					result["event"] = "Prolog unifica: objeto es PELIGROSO"
			else:
				result["success"] = True
				result["event"] = "observa el entorno vacío"
				s.last_prolog_feedback = "seguro"

		elif action == "comer":
			if s.current_item == "objeto_desconocido":
				result["success"] = True
				if s.item_is_safe:
					result["delta_hambre"] += 35.0
					result["delta_salud"] += 5.0
					result["event"] = "come objeto seguro y se sacia"
				else:
					# Envenenamiento grave
					result["delta_salud"] -= 35.0
					result["delta_hambre"] += 5.0
					result["event"] = "⚠️ come veneno! sufre dolor extremo"
				# Consumir el objeto
				s.current_item = None
				s.last_prolog_feedback = None
			else:
				# Intentar comer del entorno normal (MinimalWorld fallback)
				loc_data = LOCATIONS[s.location]
				if loc_data["food_available"]:
					result["success"] = True
					result["delta_hambre"] += 30.0
					result["event"] = "come del bosque"
				else:
					result["event"] = "no hay nada comestible aquí"

		elif action == "beber":
			loc_data = LOCATIONS[s.location]
			if loc_data["water_available"]:
				result["success"] = True
				result["delta_hambre"] += 10.0
				result["delta_salud"] += 5.0
				result["event"] = "bebe agua limpia"
			else:
				result["event"] = "no hay agua aquí"

		elif action == "dormir":
			if s.danger_nearby:
				result["delta_salud"] -= 40.0
				result["delta_energia"] += 20.0
				result["event"] = "⚠️ depredador ataca mientras duerme"
			else:
				result["success"] = True
				result["delta_energia"] += 40.0
				result["delta_salud"] += 5.0
				result["event"] = "duerme plácidamente"

		elif action == "mover":
			if s.energia < 5:
				result["event"] = "sin energía para moverse"
			else:
				adjacent = ADJACENCY[s.location]
				new_loc = self.rng.choice(adjacent)
				result["success"] = True
				result["delta_energia"] -= 5.0
				result["moved_to"] = new_loc
				result["event"] = f"se mueve a {new_loc}"
				
				# Spawneo de objeto desconocido en la nueva localización
				if new_loc in ("bosque", "río") and self.rng.random() < 0.60:
					s.current_item = "objeto_desconocido"
					s.item_is_safe = self.rng.random() < 0.50
				else:
					s.current_item = None
				s.last_prolog_feedback = None

		elif action == "piedra":
			if s.danger_nearby:
				if self.rng.random() < 0.60:
					result["success"] = True
					result["delta_energia"] -= 10.0
					s.danger_nearby = False
					result["event"] = "🪨 ahuyenta la amenaza con una piedra"
				else:
					result["delta_salud"] -= 25.0
					result["delta_energia"] -= 10.0
					result["event"] = "🪨 falla el lanzamiento y es herido"
			else:
				result["event"] = "coge una piedra del suelo"

		# Consecuencias pasivas
		if s.danger_nearby and action not in ("mover", "piedra") and action != "dormir":
			result["delta_salud"] -= 20.0
			result["event"] += " | ⚠️ depredador ataca"

		if s.storm_active and not LOCATIONS[s.location].get("storm_shelter", False):
			result["delta_salud"] -= 15.0
			result["delta_energia"] -= 5.0
			result["event"] += " | 🌩️ tormenta golpea"

		if s.hambre <= 0:
			result["delta_salud"] -= 10.0
			result["event"] += " | 💀 inanición"

		# Aplicar cambios
		s.hambre += result["delta_hambre"]
		s.salud += result["delta_salud"]
		s.energia += result["delta_energia"]

		if result["moved_to"]:
			s.location = result["moved_to"]

		s.last_action = action
		s.tick += 1
		s.clamp()

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
			"item": s.current_item,
			"safe": s.item_is_safe,
			"feedback": s.last_prolog_feedback
		})

		return result

	def get_reward(self, result: dict = None) -> float:
		s = self.state
		if not s.alive:
			return -100.0

		if result is None:
			return 0.0

		# ── Recompensa base por homeostasis (Dolor) ──
		dolor = 0.0
		if s.hambre < 50:
			dolor += (50 - s.hambre) * 0.15
		if s.hambre < 15:
			dolor += 5.0
		if s.hambre <= 0:
			dolor += 10.0

		if s.salud < 70:
			dolor += (70 - s.salud) * 0.10
		if s.salud < 30:
			dolor += 5.0

		if s.energia < 40:
			dolor += (40 - s.energia) * 0.08
		if s.energia < 10:
			dolor += 3.0

		if s.danger_nearby:
			dolor += 5.0

		delta_salud = result.get("delta_salud", 0)
		if delta_salud < 0:
			dolor += abs(delta_salud) * 0.5

		# ── Recompensa por Progreso y Toma de Decisiones Lógicas ──
		progreso = 0.0

		# Comer objeto seguro = recompensa alta
		if result["delta_hambre"] > 10 and delta_salud >= 0:
			progreso += 12.0
		
		# Comer veneno = penalización extra de dolor
		if delta_salud < -20 and "veneno" in result.get("event", ""):
			dolor += 15.0  

		# Recompensa por usar la herramienta Prolog
		if result.get("event", "").startswith("Prolog unifica"):
			progreso += 1.5  # Incentivo directo por consultar a Prolog

		# Huir del peligro
		if s.danger_nearby and result.get("moved_to"):
			progreso += 6.0

		# Mapeo de adyacencia
		if result.get("moved_to"):
			new_loc = result["moved_to"]
			if s.hambre < 40 and LOCATIONS[new_loc].get("food_available", False):
				progreso += 4.0

		paz = 1.0 if (s.hambre > 60 and s.salud > 70 and s.energia > 50 and not s.danger_nearby) else 0.0
		return progreso - dolor + paz
