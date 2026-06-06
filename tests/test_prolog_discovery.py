from src.bitnet.cooperative_world import CooperativeWorld, query_prolog


def test_prolog_rules():
	# 1. Test prerequisites
	# 'fuego' requires 'artesanía'
	res1 = query_prolog("prerrequisitos_satisfechos('fuego', ['artesanía'])")
	assert len(res1) > 0

	res2 = query_prolog("prerrequisitos_satisfechos('fuego', [])")
	assert len(res2) == 0

	# 'cocina' requires 'comida', 'agua', 'fuego'
	res3 = query_prolog("prerrequisitos_satisfechos('cocina', ['comida', 'agua', 'fuego'])")
	assert len(res3) > 0

	res4 = query_prolog("prerrequisitos_satisfechos('cocina', ['comida', 'agua'])")
	assert len(res4) == 0

	# 2. Test evaluar_intento
	# fabricar without craftsmanship skill but with resources -> intento_valido
	q1 = "evaluar_intento('fabricar', [], 0, 0, 1, 1, 0, 'bosque', 0.0, 0.0, 0.0, 0.0, 0, 'ninguno', ResultType, Event)"
	res_q1 = query_prolog(q1)[0]
	assert res_q1["ResultType"] == "intento_valido"

	# fabricar without craftsmanship skill and without resources -> error_fisico
	q2 = "evaluar_intento('fabricar', [], 0, 0, 0, 1, 0, 'bosque', 0.0, 0.0, 0.0, 0.0, 0, 'ninguno', ResultType, Event)"
	res_q2 = query_prolog(q2)[0]
	assert res_q2["ResultType"] == "error_fisico"

	# encender fire without craftsmanship skill -> error_requisito (fire requires craftsmanship)
	q3 = "evaluar_intento('encender', [], 0, 0, 2, 0, 0, 'cueva', 0.0, 0.0, 0.0, 0.0, 0, 'ninguno', ResultType, Event)"
	res_q3 = query_prolog(q3)[0]
	assert res_q3["ResultType"] == "error_requisito"

	# encender fire with craftsmanship skill but not fire skill -> intento_valido
	q4 = "evaluar_intento('encender', ['artesanía'], 0, 0, 2, 0, 0, 'cueva', 0.0, 0.0, 0.0, 0.0, 0, 'ninguno', ResultType, Event)"
	res_q4 = query_prolog(q4)[0]
	assert res_q4["ResultType"] == "intento_valido"

def test_world_discovery_flow():
	world = CooperativeWorld(seed=42, permanent_fire_location="cueva", food_decay_rate=0.1)
	world.reset()

	# Get Nico (agent_a), force location to cueva
	agent = world.agent_a
	agent.location = "cueva"
	agent.learned_skills = ["comida", "artesanía"] # max_slots is 3 initially, leaves 1 free slot
	agent.skill_experience = {}
	
	# Verify Nico can observe fire in cueva (permanent fire is active)
	assert world.fire_locations["cueva"] > 0
	
	# Execute 'ver' action under 'alegría' emotion
	agent.hambre = 100
	agent.sed = 100
	agent.salud = 100
	agent.energia = 100
	assert agent.emotion_name == "alegría"

	# Observation loop: observe fire 10 times to discover it
	for _ in range(10):
		res = world.act(agent, "ver")
		assert "observed_skills" in res
		assert "fuego" in res["observed_skills"]

	# Nico should have discovered fire now!
	assert "fuego" in agent.learned_skills
	assert len(agent.learned_skills) == 3

def test_curiosity_reset():
	world = CooperativeWorld(seed=42)
	world.reset()

	agent = world.agent_a
	# Update bosque resource map entry to 0.0
	agent.map_knowledge["bosque"] = {
		"food": 0.0,
		"water": 0.0,
		"last_updated": 0
	}

	# Let world tick reach 36, and query reward to check reset
	world.world_tick = 36
	agent.hambre = 40
	agent.sed = 80
	agent.salud = 100
	agent.energia = 80

	# Get reward -> triggers reset for bosque since world_tick (36) - last_updated (0) = 36 > 35
	world.get_reward(agent, {"success": True})

	# Bosque food mapping should be reset to 5.0 (eligible)
	assert agent.map_knowledge["bosque"]["food"] == 5.0
