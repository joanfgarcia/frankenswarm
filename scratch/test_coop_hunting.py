import unittest
from src.bitnet.cooperative_world import CooperativeWorld, COOP_LOCATION_GLYPHS, WORD_INDEX

class TestCooperativeHunting(unittest.TestCase):
	def test_tribal_expansion(self):
		world = CooperativeWorld(seed=42)
		self.assertEqual(len(world.agents), 3)
		self.assertEqual(world.agents[0].agent_id, "a")
		self.assertEqual(world.agents[1].agent_id, "b")
		self.assertEqual(world.agents[2].agent_id, "c")

	def test_broadcast_screams(self):
		world = CooperativeWorld(seed=42)
		# Colocar agentes en localizaciones distintas
		world.agent_a.location = "bosque"
		world.agent_b.location = "cueva"
		world.agent_c.location = "lago"

		# Limpiar señales recibidas
		for agent in world.agents:
			agent.received_signal = None
			agent.signal_from = None
			agent.nav_target = None

		# Agente A grita, B y C ven
		res_a, res_b, res_c, world_info = world.step("gritar", "ver", "ver", shout_concept_a=WORD_INDEX["comida"])

		self.assertTrue(res_a["shouted"])
		# B y C reciben el grito de A (bosque, comida)
		self.assertEqual(world.agent_b.signal_from, "a")
		self.assertEqual(world.agent_c.signal_from, "a")

		# B y C cambian nav_target a bosque
		self.assertEqual(world.agent_b.nav_target, "bosque")
		self.assertEqual(world.agent_c.nav_target, "bosque")

	def test_worst_state_tom_routing(self):
		world = CooperativeWorld(seed=42)
		state_a, state_b, state_c = world.reset()

		# A no tiene señal de radio
		world.agent_a.received_signal = None

		# Establecer a B en un estado crítico de hambre (15) y C en alegre (60)
		world.agent_a.companion_model["b"]["hambre"] = 15.0
		world.agent_a.companion_model["b"]["location"] = "río"
		world.agent_a.companion_model["b"]["emotion_name"] = "hambre"

		world.agent_a.companion_model["c"]["hambre"] = 60.0
		world.agent_a.companion_model["c"]["location"] = "lago"
		world.agent_a.companion_model["c"]["emotion_name"] = "alegría"

		perc_a = world.perceive(world.agent_a)

		# Perc_a[4] (RcvLoc) y perc_a[5] (RcvWhat) deben corresponder al compañero más crítico (B: río, hambre)
		expected_loc_glyph = WORD_INDEX[COOP_LOCATION_GLYPHS["río"]]
		expected_what_glyph = WORD_INDEX.get("hambre", 0)

		self.assertEqual(perc_a[4], expected_loc_glyph)
		self.assertEqual(perc_a[5], expected_what_glyph)

	def test_cooperative_hunting_success(self):
		world = CooperativeWorld(seed=42)
		state_a, state_b, state_c = world.reset()

		# Spawn prey at río
		world.prey_location = "río"
		world.prey_timer = 5

		# Colocar A y B en río, luchar
		world.agent_a.location = "río"
		world.agent_b.location = "río"
		world.agent_c.location = "cueva"

		# Inicializar hambre baja para probar saciado
		world.agent_a.hambre = 30.0
		world.agent_b.hambre = 30.0

		res_a, res_b, res_c, world_info = world.step("luchar", "luchar", "ver")

		# Éxito: presa eliminada (None)
		self.assertIsNone(world.prey_location)
		self.assertEqual(world_info["coop_bonus_a"], 15.0)
		self.assertEqual(world_info["coop_bonus_b"], 15.0)
		self.assertEqual(world.agent_a.mochila_comida, 1)
		self.assertEqual(world.agent_b.mochila_comida, 1)
		self.assertGreater(world.agent_a.hambre, 70.0)

	def test_cooperative_hunting_failure(self):
		world = CooperativeWorld(seed=42)
		state_a, state_b, state_c = world.reset()

		# Spawn prey at río
		world.prey_location = "río"
		world.prey_timer = 5

		# Solo A en río luchando
		world.agent_a.location = "río"
		world.agent_b.location = "cueva"
		world.agent_c.location = "cueva"

		world.agent_a.salud = 100.0

		res_a, res_b, res_c, world_info = world.step("luchar", "ver", "ver")

		# Falló: presa sigue activa o huyó, A recibe daño de salud
		self.assertTrue("caza fallida" in res_a["event"])
		self.assertLess(world.agent_a.salud, 100.0)

if __name__ == "__main__":
	unittest.main()
