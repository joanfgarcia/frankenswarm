from src.bitnet.worlds.puzzle_world import PuzzleWorld


def test_puzzle_world_initialization():
	"""Verifica que el entorno PuzzleWorld se inicializa sin errores."""
	world = PuzzleWorld(seed=42)
	state = world.reset()
	assert state.alive
	assert state.row == 0
	assert state.col == 0
	assert not state.has_key
	assert not state.door_open
	assert state.tick == 0

def test_puzzle_world_movement_boundaries():
	"""Verifica que el agente no pueda salir de los límites de la cuadrícula 3x3."""
	world = PuzzleWorld(seed=42)
	state = world.reset()
	
	# Intentar moverse arriba en (0,0) -> chocar
	res = world.act("comer")
	assert not res["success"]
	assert s_row_col(state) == (0, 0)
	
	# Intentar moverse izquierda en (0,0) -> chocar
	res = world.act("dormir")
	assert not res["success"]
	assert s_row_col(state) == (0, 0)
	
	# Moverse derecha hasta el borde
	world.act("mover") # col = 1
	world.act("mover") # col = 2
	res = world.act("mover") # chocar
	assert not res["success"]
	assert s_row_col(state) == (0, 2)
	
	# Moverse abajo hasta el borde
	world.act("beber") # row = 1
	world.act("beber") # row = 2
	res = world.act("beber") # chocar
	assert not res["success"]
	assert s_row_col(state) == (2, 2)

def test_puzzle_world_pick_key():
	"""Verifica la lógica de coger la llave en la celda del árbol (1,2)."""
	world = PuzzleWorld(seed=42)
	state = world.reset()
	
	# Intentar coger llave en (0,0) -> fallar
	res = world.act("ver")
	assert not res["success"]
	assert not state.has_key
	
	# Ir al árbol: (0,0) -> (0,1) -> (0,2) -> (1,2)
	world.act("mover")
	world.act("mover")
	world.act("beber")
	assert s_row_col(state) == (1, 2)
	
	# Coger la llave -> éxito
	res = world.act("ver")
	assert res["success"]
	assert state.has_key
	assert res["reward"] == 15.0

def test_puzzle_world_open_door():
	"""Verifica la lógica de abrir la puerta en la celda de la noche (2,2)."""
	world = PuzzleWorld(seed=42)
	state = world.reset()
	
	# Ir a la puerta sin llave: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)
	world.act("mover")
	world.act("mover")
	world.act("beber")
	world.act("beber")
	assert s_row_col(state) == (2, 2)
	
	# Intentar abrir la puerta sin llave -> fallar
	res = world.act("piedra")
	assert not res["success"]
	assert not state.door_open
	assert res["reward"] == -10.0
	
	# Resetear y hacer el camino correcto
	state = world.reset()
	# Ir a por la llave: (0,0) -> (0,1) -> (0,2) -> (1,2)
	world.act("mover")
	world.act("mover")
	world.act("beber")
	world.act("ver") # Coger llave
	assert state.has_key
	
	# Ir a la puerta: (1,2) -> (2,2)
	world.act("beber")
	
	# Abrir la puerta con llave -> éxito, termina episodio
	res = world.act("piedra")
	assert res["success"]
	assert state.door_open
	assert not state.alive  # Se completa el episodio
	assert res["reward"] == 100.0

def s_row_col(state):
	return (state.row, state.col)

def test_puzzle_world_shaping_reward():
	"""Verifica que la recompensa moldeada (shaping_reward) funciona correctamente."""
	world = PuzzleWorld(seed=42)
	world.reset()
	
	# De (0,0) a (0,1) -> se acerca a la llave (1,2) -> shaping_reward positivo (+2.0)
	res = world.act("mover")
	assert res["shaping_reward"] == 2.0
	assert world.get_reward(res) == 1.0 # -1.0 tiempo + 2.0 shaping = 1.0
	
	# De (0,1) a (0,0) -> se aleja de la llave -> shaping_reward negativo (-2.0)
	res = world.act("dormir")
	assert res["shaping_reward"] == -2.0
	assert world.get_reward(res) == -3.0 # -1.0 tiempo - 2.0 shaping = -3.0
	
	# Hitting wall at (0,0) -> no cambia potencial -> shaping_reward = 0.0
	res = world.act("comer")
	assert res["shaping_reward"] == 0.0
	assert world.get_reward(res) == -2.0 # -2.0 choque + 0.0 shaping = -2.0
