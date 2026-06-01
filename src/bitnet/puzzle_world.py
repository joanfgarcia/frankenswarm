"""
puzzle_world.py

Entorno de rompecabezas en cuadrícula 3x3 para entrenar capacidades de planificación.
Bit debe navegar, recoger la llave y abrir la puerta.
"""

import random

# Mapeo de coordenadas (row, col) a glifos conceptuales
GRID_GLYPHS = {
	(0, 0): "cueva",
	(0, 1): "bosque",
	(0, 2): "río",
	(1, 0): "tierra",
	(1, 1): "piedra",
	(1, 2): "árbol",
	(2, 0): "fuego",
	(2, 1): "sol",
	(2, 2): "noche",
}

class PuzzleState:
	def __init__(self):
		self.row = 0
		self.col = 0
		self.has_key = False
		self.door_open = False
		self.tick = 0
		self.alive = True
		self.last_action = None
		self.last_success = True
		self.last_event = "comienzo"

	@property
	def location(self) -> str:
		return GRID_GLYPHS[(self.row, self.col)]

	@property
	def emotion_name(self) -> str:
		# Moduladores fijos para que el modelo no se queje de la falta de campos emoción
		if self.door_open:
			return "alegría"
		elif self.tick > 45:
			return "tristeza"  # fatiga temporal
		return "ira"

	@property
	def emotion_id(self) -> int:
		from src.bitnet.glyph_vocabulary import EMOTION_INDEX
		return EMOTION_INDEX[self.emotion_name]

class PuzzleWorld:
	def __init__(self, seed: int = 42):
		self.rng = random.Random(seed)
		self.state = PuzzleState()
		self.history = []

	def reset(self, seed: int | None = None) -> PuzzleState:
		if seed is not None:
			self.rng = random.Random(seed)
		self.state = PuzzleState()
		self.history = []
		return self.state

	def _get_potential(self) -> float:
		s = self.state
		if s.door_open:
			return 3.0
		elif s.has_key:
			dist = abs(s.row - 2) + abs(s.col - 2)
			return 2.0 - float(dist)
		else:
			dist = abs(s.row - 1) + abs(s.col - 2)
			return -float(dist)

	def perceive(self) -> list[str]:
		# Percepción: [Ubicación, EstadoLlave, ÉxitoAnterior, Relleno]
		loc_glyph = self.state.location
		key_glyph = "seguro" if self.state.has_key else "peligro"
		success_glyph = "saciado" if self.state.last_success else "herida"
		
		perception = [loc_glyph, key_glyph, success_glyph, success_glyph]
		return perception

	def act(self, action: str) -> dict:
		s = self.state
		old_potential = self._get_potential()
		result = {
			"success": False,
			"reward": -1.0,  # Penalización de tiempo estándar
			"event": "",
		}

		s.last_action = action
		s.tick += 1

		# Mapeo de acciones
		# comer  → Arriba (row - 1)
		# beber  → Abajo (row + 1)
		# dormir → Izquierda (col - 1)
		# mover  → Derecha (col + 1)
		# ver    → Recoger Llave (en árbol / (1,2))
		# piedra → Abrir Puerta (en noche / (2,2))

		if action == "comer":  # UP
			if s.row > 0:
				s.row -= 1
				result["success"] = True
				result["event"] = "se mueve hacia arriba"
			else:
				result["reward"] = -2.0  # Choque
				result["event"] = "choca contra la pared norte"

		elif action == "beber":  # DOWN
			if s.row < 2:
				s.row += 1
				result["success"] = True
				result["event"] = "se mueve hacia abajo"
			else:
				result["reward"] = -2.0
				result["event"] = "choca contra la pared sur"

		elif action == "dormir":  # LEFT
			if s.col > 0:
				s.col -= 1
				result["success"] = True
				result["event"] = "se mueve hacia la izquierda"
			else:
				result["reward"] = -2.0
				result["event"] = "choca contra la pared oeste"

		elif action == "mover":  # RIGHT
			if s.col < 2:
				s.col += 1
				result["success"] = True
				result["event"] = "se mueve hacia la derecha"
			else:
				result["reward"] = -2.0
				result["event"] = "choca contra la pared este"

		elif action == "ver":  # RECOGER LLAVE
			if s.row == 1 and s.col == 2:  # árbol
				if not s.has_key:
					s.has_key = True
					result["success"] = True
					result["reward"] = 15.0  # Gran recompensa por recoger la llave
					result["event"] = "🔑 recoge la llave del árbol"
				else:
					result["reward"] = -2.0
					result["event"] = "intenta coger la llave pero ya la tiene"
			else:
				result["reward"] = -5.0
				result["event"] = "intenta coger llave pero aquí no hay llave"

		elif action == "piedra":  # ABRIR PUERTA
			if s.row == 2 and s.col == 2:  # noche
				if s.has_key:
					s.door_open = True
					result["success"] = True
					result["reward"] = 100.0  # Éxito absoluto
					result["event"] = "🔓 ¡abre la puerta y escapa del rompecabezas!"
					s.alive = False  # Termina el episodio al salir
				else:
					result["reward"] = -10.0
					result["event"] = "intenta abrir puerta pero está cerrada con llave"
			else:
				result["reward"] = -5.0
				result["event"] = "intenta abrir puerta pero aquí no hay puerta"

		s.last_success = result["success"]
		s.last_event = result["event"]

		# Penalización por límite de ticks
		if s.tick >= 50:
			s.alive = False
			result["reward"] -= 10.0
			result["event"] += " | 💀 se agota el tiempo"

		# Shaping reward calculation
		new_potential = self._get_potential()
		result["shaping_reward"] = 2.0 * (new_potential - old_potential)

		# Registrar en historial
		self.history.append({
			"tick": s.tick,
			"action": action,
			"row": s.row,
			"col": s.col,
			"has_key": s.has_key,
			"door_open": s.door_open,
			"reward": result["reward"],
			"shaping_reward": result["shaping_reward"],
			"event": result["event"],
		})

		return result

	def get_reward(self, result: dict) -> float:
		return float(result.get("reward", -1.0)) + float(result.get("shaping_reward", 0.0))
