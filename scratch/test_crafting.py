import unittest
from src.bitnet.cooperative_world import CooperativeWorld, CoopAgentState, COOP_LOCATIONS

class TestCraftingAndSurvival(unittest.TestCase):
	def setUp(self):
		self.world = CooperativeWorld(seed=42)

	def test_metabolic_weight_penalty(self):
		# Agente sin materiales (peso = 1.0)
		self.world.agent_a.mochila_comida = 0
		self.world.agent_a.mochila_agua = 0
		self.world.agent_a.mochila_ramas = 0
		self.world.agent_a.mochila_piedras = 0
		self.world.agent_a.tiene_lanza = False
		self.world.agent_a.hambre = 60.0
		self.world.agent_a.sed = 80.0
		self.world.agent_a.energia = 80.0

		res_light = self.world.act(self.world.agent_a, "ver")
		# Tasa base: hambre decae en -3.5, sed en -7.0, energia en -2.0 (más -1.0 por la acción ver)
		self.assertAlmostEqual(res_light["delta_hambre"], -3.5)
		self.assertAlmostEqual(res_light["delta_sed"], -7.0)
		self.assertAlmostEqual(res_light["delta_energia"], -3.0)

		# Agente cargado a tope:
		# Comida (1) + Agua (1) = +0.20
		# Ramas (3) = +0.45
		# Piedras (3) = +0.75
		# Lanza = +0.15
		# Multiplicador total = 1.0 + 0.20 + 0.45 + 0.75 + 0.15 = 2.55
		self.world.agent_b.mochila_comida = 1
		self.world.agent_b.mochila_agua = 1
		self.world.agent_b.mochila_ramas = 3
		self.world.agent_b.mochila_piedras = 3
		self.world.agent_b.tiene_lanza = True
		self.world.agent_b.hambre = 60.0
		self.world.agent_b.sed = 80.0
		self.world.agent_b.energia = 80.0

		res_heavy = self.world.act(self.world.agent_b, "ver")
		self.assertAlmostEqual(res_heavy["delta_hambre"], -3.5 * 2.55)
		self.assertAlmostEqual(res_heavy["delta_sed"], -7.0 * 2.55)
		self.assertAlmostEqual(res_heavy["delta_energia"], -2.0 * 2.55 - 1.0)

	def test_automatic_resource_gathering_and_fishing(self):
		# Configurar bosque con ramas elegibles y capaciadaes
		self.world.agent_a.location = "bosque"
		self.world.agent_a.learned_skills = ["artesanía"]
		self.world.agent_a.mochila_ramas = 0
		self.world.agent_a.mochila_piedras = 0
		self.world.resource_capacities["bosque"]["branches"] = 5.0
		self.world.resource_capacities["bosque"]["stones"] = 0.0

		# Actuar: debería recolectar 1 rama automáticamente
		self.world.act(self.world.agent_a, "ver")
		self.assertEqual(self.world.agent_a.mochila_ramas, 1)
		self.assertEqual(self.world.resource_capacities["bosque"]["branches"], 4.0)

		# Configurar lago: fish_eligible = True, food_eligible = False
		self.world.agent_b.location = "lago"
		self.world.agent_b.learned_skills = ["pesca"]
		self.world.agent_b.mochila_comida = 0
		self.world.resource_capacities["lago"]["food"] = 5.0

		# Actuar: debería pescar automáticamente
		self.world.act(self.world.agent_b, "ver")
		self.assertEqual(self.world.agent_b.mochila_comida, 1)
		self.assertEqual(self.world.resource_capacities["lago"]["food"], 4.0)

		# Intentar en lago sin habilidad pesca
		self.world.agent_c.location = "lago"
		self.world.agent_c.learned_skills = ["comida"] # habilidad comida normal no sirve para pescar
		self.world.agent_c.mochila_comida = 0
		self.world.act(self.world.agent_c, "ver")
		self.assertEqual(self.world.agent_c.mochila_comida, 0, "Habilidad comida normal no debe permitir pescar")

	def test_crafting_fabricar_lanza(self):
		# Colocar en cueva para evitar recolección automática
		self.world.agent_a.location = "cueva"
		self.world.agent_a.learned_skills = ["artesanía"]
		self.world.agent_a.mochila_ramas = 1
		self.world.agent_a.mochila_piedras = 1
		self.world.agent_a.tiene_lanza = False

		res = self.world.act(self.world.agent_a, "fabricar")
		self.assertTrue(res["success"])
		self.assertTrue(self.world.agent_a.tiene_lanza)
		self.assertEqual(self.world.agent_a.mochila_ramas, 0)
		self.assertEqual(self.world.agent_a.mochila_piedras, 0)

	def test_predator_defense_with_spear(self):
		self.world.agent_a.location = "cueva"
		self.world.agent_a.tiene_lanza = True
		self.world.agent_a.salud = 100.0
		self.world.agent_a.danger_nearby = True

		# Dormir bajo ataque depredador
		res = self.world.act(self.world.agent_a, "dormir")
		# Lanza se rompe, salud queda intacta (100 - 0 delta)
		self.assertFalse(self.world.agent_a.tiene_lanza)
		self.assertEqual(self.world.agent_a.salud, 100.0)
		self.assertIn("repelido por la lanza", res["event"])

	def test_solo_prey_hunting_with_spear(self):
		# Configurar presa en río y fijar timer para evitar despawn
		self.world.prey_location = "río"
		self.world.prey_timer = 5
		self.world.agent_a.location = "río"
		self.world.agent_a.learned_skills = ["caza"]
		self.world.agent_a.tiene_lanza = True
		self.world.agent_a.hambre = 30.0

		# step con lucha de A, y B durmiendo
		res_a, res_b, res_c, world_info = self.world.step("luchar", "dormir", "ver", "ver")

		self.assertFalse(self.world.agent_a.tiene_lanza, "La lanza debe romperse")
		self.assertIsNone(self.world.prey_location, "La presa debe morir/desaparecer")
		self.assertEqual(self.world.agent_a.mochila_comida, 1, "Obtiene comida en mochila")
		self.assertIn("caza individual exitosa", res_a["event"])
		self.assertEqual(world_info["coop_bonus_a"], 15.0)

	def test_fire_lighting_and_decay(self):
		self.world.agent_a.location = "valle"
		self.world.agent_a.learned_skills = ["fuego"]
		self.world.agent_a.mochila_ramas = 2
		self.world.fire_locations["valle"] = 0

		res = self.world.act(self.world.agent_a, "encender")
		self.assertTrue(res["success"])
		self.assertEqual(self.world.fire_locations["valle"], 4, "Hoguera debe iniciarse en 4 ticks")
		self.assertEqual(self.world.agent_a.mochila_ramas, 0)

		# Avanzar tick y comprobar decaimiento
		self.world.step("ver", "ver", "ver", "ver")
		self.assertEqual(self.world.fire_locations["valle"], 3, "Hoguera debe decaer a 3")

	def test_cooking_barbecue_and_stew(self):
		# 1. Comida cocinada (barbacoa) sin habilidad especial de cocina, con fuego activo
		self.world.agent_a.location = "bosque"
		self.world.agent_a.learned_skills = ["comida"]
		self.world.agent_a.mochila_comida = 1
		self.world.fire_locations["bosque"] = 3
		self.world.agent_a.hambre = 20.0
		self.world.agent_a.sed = 80.0

		res_bbq = self.world.act(self.world.agent_a, "comer")
		self.assertTrue(res_bbq["success"])
		# Restaura +60.0 de hambre y delta_sed es 0.0 (peso de mochila comida vacía = 1.10)
		self.assertAlmostEqual(res_bbq["delta_hambre"], 60.0 - 3.5 * 1.1)
		self.assertAlmostEqual(res_bbq["delta_sed"], 0.0 - 7.0 * 1.1)

		# 2. Guiso caliente (comida++) con habilidad de cocina
		self.world.agent_b.location = "bosque"
		self.world.agent_b.learned_skills = ["cocina", "comida"]
		self.world.agent_b.mochila_comida = 1
		self.world.agent_b.mochila_agua = 1
		self.world.fire_locations["bosque"] = 3
		self.world.agent_b.hambre = 20.0
		self.world.agent_b.sed = 20.0

		res_stew = self.world.act(self.world.agent_b, "comer")
		self.assertTrue(res_stew["success"])
		self.assertEqual(self.world.agent_b.mochila_comida, 0)
		self.assertEqual(self.world.agent_b.mochila_agua, 0)
		self.assertIn("guiso caliente", res_stew["event"])
		# Peso antes de comer era 1.20 (comida=1, agua=1)
		self.assertAlmostEqual(res_stew["delta_hambre"], 85.0 - 3.5 * 1.2)
		self.assertAlmostEqual(res_stew["delta_sed"], 50.0 - 7.0 * 1.2)

		# 3. Comida cruda (sin fuego) -> puede intoxicar
		self.world.agent_c.location = "valle"
		self.world.agent_c.mochila_comida = 1
		self.world.fire_locations["valle"] = 0
		self.world.agent_c.hambre = 20.0
		self.world.agent_c.sed = 80.0
		self.world.agent_c.salud = 100.0
		
		# Asegurar repetibilidad del random de intoxicación
		self.world.rng.seed(42) # un seed específico para verificar rama de intoxicación
		res_raw = self.world.act(self.world.agent_c, "comer")
		self.assertTrue(res_raw["success"])
		self.assertAlmostEqual(res_raw["delta_hambre"], 40.0 - 3.5 * 1.1)
		self.assertAlmostEqual(res_raw["delta_sed"], -3.0 - 7.0 * 1.1)

	def test_construction_refugio(self):
		self.world.agent_a.location = "bosque"
		self.world.agent_a.learned_skills = ["construcción"]
		self.world.agent_a.mochila_ramas = 2
		self.world.agent_a.mochila_piedras = 1
		COOP_LOCATIONS["bosque"]["storm_shelter"] = False

		res = self.world.act(self.world.agent_a, "construir")
		self.assertTrue(res["success"])
		self.assertTrue(COOP_LOCATIONS["bosque"]["storm_shelter"], "El bosque debe ser ahora un refugio permanente")
		self.assertEqual(self.world.agent_a.mochila_ramas, 0)
		self.assertEqual(self.world.agent_a.mochila_piedras, 0)

if __name__ == "__main__":
	unittest.main()
