import sys
import os
import pytest

# Asegurar que el path del proyecto está en sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bitnet.cooperative_world import CooperativeWorld, CoopAgentState

def test_boredom_initialization():
	world = CooperativeWorld(seed=42)
	state_a, state_b, state_c = world.reset()
	assert state_a.aburrimiento == 0.0
	assert state_b.aburrimiento == 0.0
	assert state_c.aburrimiento == 0.0

def test_boredom_accumulation_dormir():
	world = CooperativeWorld(seed=42)
	state_a, state_b, state_c = world.reset()
	
	# Dormir aumenta el aburrimiento en 6.0
	world.act(state_a, "dormir")
	assert state_a.aburrimiento == 6.0

def test_boredom_accumulation_still():
	world = CooperativeWorld(seed=42)
	state_a, state_b, state_c = world.reset()
	
	# Ver (u otras acciones estáticas) aumenta el aburrimiento en 2.0
	world.act(state_a, "ver")
	assert state_a.aburrimiento == 2.0

def test_boredom_relief_mover():
	world = CooperativeWorld(seed=42)
	state_a, state_b, state_c = world.reset()
	
	# Acumular aburrimiento
	state_a.aburrimiento = 50.0
	state_a.location = "cueva"
	state_a.previous_location = "cueva"
	
	# Moverse a una nueva localización (ej: bosque, que está adyacente a cueva)
	# Forzamos que se mueva a bosque
	res = world.act(state_a, "mover")
	# Si se movió a una nueva casilla, el aburrimiento disminuye en 20.0
	if state_a.location != "cueva":
		assert state_a.aburrimiento == 30.0
	else:
		# Si se quedó en cueva por mala suerte o cansancio, aumenta en 2.0 o 6.0
		assert state_a.aburrimiento > 50.0

def test_boredom_relief_interact():
	world = CooperativeWorld(seed=42)
	state_a, state_b, state_c = world.reset()
	
	# Acumular aburrimiento
	state_a.aburrimiento = 30.0
	state_a.location = "cueva"
	state_b.location = "cueva"
	state_a.mochila_comida = 1
	state_b.hambre = 20.0
	
	# Ejecutar acción de dar (interacción)
	res = world.act(state_a, "dar")
	assert res["success"]
	# La acción de dar alivia el aburrimiento en 15.0
	assert state_a.aburrimiento == 15.0

def test_boredom_reward_penalty():
	world = CooperativeWorld(seed=42)
	state_a, state_b, state_c = world.reset()
	
	# Con aburrimiento <= 50, no hay penalización
	state_a.aburrimiento = 50.0
	res_normal = {"success": True, "delta_hambre": 0, "delta_sed": 0, "delta_salud": 0, "delta_energia": 0}
	reward_normal = world.get_reward(state_a, res_normal)
	
	# Con aburrimiento > 50, se resta una penalización lineal
	state_a.aburrimiento = 100.0
	reward_bored = world.get_reward(state_a, res_normal)
	
	# A aburrimiento=100, la penalización es de -0.5
	assert reward_normal - reward_bored == pytest.approx(0.5)

if __name__ == "__main__":
	sys.exit(pytest.main([__file__]))
