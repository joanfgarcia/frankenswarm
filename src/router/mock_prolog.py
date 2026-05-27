import numpy as np


class MockPrologRouter:
	"""
	Fase 1: Router Simbólico (Mock de Prolog).
	Simula las restricciones lógicas y de triage cognitivo.
	En la versión final, esto generará código ISO Prolog para evaluación formal.
	"""

	def __init__(self, nodes: dict):
		# Referencias a las instancias de los expertos disponibles
		self.nodes = nodes

	def evaluate_and_route(self, embedding: np.ndarray, top_concepts: list[tuple[str, float]]) -> str:
		"""
		Simula las reglas Prolog usando los conceptos más cercanos extraídos por el Traductor.
		"""
		print("  [🧭 Router Prolog] Evaluando vector entrante...")

		# Obtenemos la etiqueta semántica primaria
		primary_concept = top_concepts[0][0].lower() if top_concepts else ""

		# Simulación de Assertz/Retract lógico basado en el contenido latente
		if "código" in primary_concept or "sintaxis" in primary_concept:
			decision = "Experto_Core"
			print(f"  [🧭 Router Prolog] Restricción cumplida: Contenido estructural detectado -> Ruteando a {decision}.")
			return decision

		elif "lógica" in primary_concept or "matemática" in primary_concept:
			decision = "Experto_Oraculo"
			print(f"  [🧭 Router Prolog] Restricción cumplida: Razonamiento abstracto detectado -> Ruteando a {decision}.")
			return decision

		else:
			decision = "Experto_Generalista"
			print(f"  [🧭 Router Prolog] Restricción cumplida: Entropía alta / Contenido general -> Ruteando a {decision}.")
			return decision
