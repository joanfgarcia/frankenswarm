from src.bitnet.prolog_world import PrologSurvivalWorld


def test_prolog_world_initialization():
	"""Verifica que el entorno PrologSurvivalWorld se inicializa sin errores."""
	world = PrologSurvivalWorld(seed=42)
	state = world.reset()
	assert state.alive
	assert state.location == "cueva"
	assert state.current_item is None
	assert state.last_prolog_feedback is None

def test_prolog_world_query():
	"""Verifica que la acción 'ver' unifique correctamente las consultas lógicas a Prolog."""
	world = PrologSurvivalWorld(seed=42)
	state = world.reset()
	
	# Forzar el estado para simular el encuentro con un objeto desconocido
	state.current_item = "objeto_desconocido"
	
	# Caso 1: Objeto seguro
	state.item_is_safe = True
	res = world.act("ver")
	assert res["success"]
	assert state.last_prolog_feedback == "seguro"
	assert "objeto es seguro" in res["event"]
	
	# Caso 2: Objeto venenoso
	state.item_is_safe = False
	res2 = world.act("ver")
	assert res2["success"]
	assert state.last_prolog_feedback == "peligro"
	assert "objeto es PELIGROSO" in res2["event"]

def test_prolog_world_eat_poison():
	"""Verifica que comer un objeto venenoso penalice la salud."""
	world = PrologSurvivalWorld(seed=42)
	state = world.reset()
	
	state.current_item = "objeto_desconocido"
	state.item_is_safe = False
	
	initial_health = state.salud
	res = world.act("comer")
	assert res["success"]
	assert "veneno" in res["event"]
	assert state.salud < initial_health
	assert state.current_item is None  # Se consume el objeto

def test_prolog_world_eat_safe():
	"""Verifica que comer un objeto seguro aumente el hambre/saciado."""
	world = PrologSurvivalWorld(seed=42)
	state = world.reset()
	
	state.current_item = "objeto_desconocido"
	state.item_is_safe = True
	state.hambre = 20.0
	
	initial_hunger = state.hambre
	res = world.act("comer")
	assert res["success"]
	assert "se sacia" in res["event"]
	assert state.hambre > initial_hunger
	assert state.current_item is None
