import unittest
from src.bitnet.cooperative_world import CooperativeWorld, COOP_LOCATION_GLYPHS, WORD_INDEX

class TestSkillSpecialization(unittest.TestCase):
	def test_nico_cannot_hunt(self):
		world = CooperativeWorld(seed=42)
		world.reset()

		# Spawn presas en río
		world.prey_location = "río"
		world.prey_timer = 10

		# Colocar Nico (A) y Sofy (B) en el río
		world.agent_a.location = "río"
		world.agent_b.location = "río"
		world.agent_c.location = "cueva"

		# Ambos luchan
		res_a, res_b, res_c, world_info = world.step("luchar", "luchar", "ver")

		# Dado que Nico no puede cazar, el recuento de cazadores es 1 (solo Sofy)
		# Por tanto la caza debe fallar, la presa sigue viva y Nico/Sofy no se sacian
		self.assertIsNotNone(world.prey_location)
		self.assertTrue("caza fallida" in res_b["event"])
		self.assertNotEqual(res_a["event"], "caza cooperativa exitosa de la presa! Comida obtenida.")

	def test_sofy_water_exclusion(self):
		world = CooperativeWorld(seed=42)
		world.reset()

		# Sofy (B) va al lago (recurso de agua disponible)
		world.agent_b.location = "lago"
		# Asegurar que el lago tiene agua
		world.resource_capacities["lago"]["water"] = 5.0

		# 1. Tránsito no debe llenar su mochila automáticamente
		world.step("ver", "ver", "ver")
		self.assertEqual(world.agent_b.mochila_agua, 0)

		# 2. Beber del suelo debe fallar
		world.agent_b.sed = 50.0
		res_a, res_b, res_c, _ = world.step("ver", "beber", "ver")
		self.assertFalse(res_b["success"])
		self.assertTrue("Sofy no sabe extraer agua del suelo" in res_b["event"])
		self.assertLess(world.agent_b.sed, 50.0) # decae por metabolismo

		# 3. Nico (A) extrae agua, va con Sofy y se la da
		world.agent_a.location = "lago"
		world.step("ver", "ver", "ver")
		self.assertEqual(world.agent_a.mochila_agua, 1) # Nico llena mochila

		# Nico y Sofy se juntan en el lago (ya están ahí) y Nico le da agua
		world.agent_b.sed = 40.0
		res_a, res_b, res_c, _ = world.step("dar", "ver", "ver")
		
		# Nico comparte el agua directamente (se consume)
		self.assertTrue(res_a["success"])
		self.assertEqual(world.agent_a.mochila_agua, 0)
		self.assertGreater(world.agent_b.sed, 40.0) # Sofy la consume directamente

	def test_hugo_food_exclusion(self):
		world = CooperativeWorld(seed=42)
		world.reset()

		# Hugo (C) va al bosque (recurso de comida disponible)
		world.agent_c.location = "bosque"
		world.resource_capacities["bosque"]["food"] = 5.0

		# 1. Tránsito no debe llenar su mochila automáticamente
		world.step("ver", "ver", "ver")
		self.assertEqual(world.agent_c.mochila_comida, 0)

		# 2. Comer del suelo debe fallar
		world.agent_c.hambre = 50.0
		res_a, res_b, res_c, _ = world.step("ver", "ver", "comer")
		self.assertFalse(res_c["success"])
		self.assertTrue("Hugo no sabe recolectar comida del suelo" in res_c["event"])
		self.assertLess(world.agent_c.hambre, 50.0) # decae por metabolismo

		# 3. Nico (A) recolecta comida, va al bosque y se la da a Hugo
		world.agent_a.location = "bosque"
		world.step("ver", "ver", "ver")
		self.assertEqual(world.agent_a.mochila_comida, 1) # Nico llena mochila

		# Nico le da comida
		world.agent_c.hambre = 40.0
		res_a, res_b, res_c, _ = world.step("dar", "ver", "ver")
		
		# Nico comparte la comida directamente (se consume)
		self.assertTrue(res_a["success"])
		self.assertEqual(world.agent_a.mochila_comida, 0)
		self.assertGreater(world.agent_c.hambre, 40.0) # Hugo la consume directamente

	def test_allowed_skills(self):
		world = CooperativeWorld(seed=42)
		world.reset()

		# Nico (A) puede recolectar y consumir comida/agua del suelo
		world.agent_a.location = "río"
		world.resource_capacities["río"]["food"] = 5.0
		world.resource_capacities["río"]["water"] = 5.0
		
		# Llena agua
		world.step("ver", "ver", "ver")
		self.assertEqual(world.agent_a.mochila_agua, 1)

		# Consume comida del suelo
		world.agent_a.hambre = 50.0
		res_a, _, _, _ = world.step("comer", "ver", "ver")
		self.assertTrue(res_a["success"])
		self.assertGreater(world.agent_a.hambre, 50.0)

		# Sofy (B) puede recolectar y consumir comida del suelo
		world.agent_b.location = "bosque"
		world.resource_capacities["bosque"]["food"] = 5.0
		world.agent_b.hambre = 50.0
		res_a, res_b, _, _ = world.step("ver", "comer", "ver")
		self.assertTrue(res_b["success"])
		self.assertGreater(world.agent_b.hambre, 50.0)

		# Hugo (C) puede recolectar y consumir agua del suelo
		world.agent_c.location = "lago"
		world.resource_capacities["lago"]["water"] = 5.0
		world.agent_c.sed = 50.0
		res_a, _, res_c, _ = world.step("ver", "ver", "beber")
		self.assertTrue(res_c["success"])
		self.assertGreater(world.agent_c.sed, 50.0)

if __name__ == "__main__":
	unittest.main()
