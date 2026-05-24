import numpy as np


class MockBitNetNode:
	"""
	Fase 1: Nodo Experto Falso.
	Simula una red neuronal BitNet 1.58b especializada.
	Recibe un vector, le aplica una ligera perturbación (procesamiento) y lo devuelve.
	"""

	def __init__(self, name: str, specialty: str):
		self.name = name
		self.specialty = specialty

	def forward(self, embedding: np.ndarray) -> np.ndarray:
		print(f"  [🧠 Nodo: {self.name}] Procesando embedding con expertise en: {self.specialty}...")
		# Añade ruido gaussiano mínimo para simular un "procesamiento" matemático
		# que altera el vector de salida.
		processed_vector = embedding + np.random.normal(0, 0.005, embedding.shape)
		return processed_vector
