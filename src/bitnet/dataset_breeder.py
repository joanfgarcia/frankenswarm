import numpy as np

from src.bitnet.translator import SovereignTranslator


class ReferentialDatasetBreeder:
	"""
	Breeder de Datasets Académicos y de Interacción.
	Genera los lotes de "conceptos objetivo" y "estados emocionales" para el juego referencial del Grado 0.
	"""

	def __init__(self, translator: SovereignTranslator):
		self.translator = translator

		# Conceptos físicos objetivos
		self.target_concepts = [
			"gato",
			"perro",
			"casa",
			"árbol",
			"agua",
			"fuego",
			"tierra",
			"aire",
			"sol",
			"luna",
			"peligro",
			"seguridad",
			"búnker",
			"agente",
			"código",
		]

		# Estados afectivos/fisiológicos objetivos
		self.target_emotions = ["miedo", "alegría", "ira", "tristeza", "dolor", "hambre"]

		# Token IDs de Capa 1 para conceptos físicos
		self.target_concept_token_ids = []
		for concept in self.target_concepts:
			tids = self.translator.encode(concept)
			self.target_concept_token_ids.append(tids[0] if tids else 0)

		# Token IDs de Capa 1 para estados emocionales
		self.target_emotion_token_ids = []
		for emotion in self.target_emotions:
			tids = self.translator.encode(emotion)
			self.target_emotion_token_ids.append(tids[0] if tids else 0)

		# Estados homeostáticos objetivos
		self.target_homeostasis = ["neutral", "dolor", "hambre", "urgencia", "seguridad"]

		# Token IDs de Capa 1 para homeostasis
		self.target_homeostasis_token_ids = []
		for homeo in self.target_homeostasis:
			tids = self.translator.encode(homeo)
			self.target_homeostasis_token_ids.append(tids[0] if tids else 0)

		self.num_concepts = len(self.target_concepts)
		self.num_emotions = len(self.target_emotions)
		self.num_homeostasis = len(self.target_homeostasis)

	def generate_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Genera un lote aleatorio para el juego de señalización tridimensional.
		Devuelve:
		- concept_targets: Índices locales de conceptos (batch_size,)
		- concept_token_ids: Token IDs de Capa 1 de conceptos (batch_size,)
		- emotion_targets: Índices locales de emociones (batch_size,)
		- emotion_token_ids: Token IDs de Capa 1 de emociones (batch_size,)
		- homeostasis_targets: Índices locales de homeostasis (batch_size,)
		- homeostasis_token_ids: Token IDs de Capa 1 de homeostasis (batch_size,)
		"""
		concept_targets = np.random.randint(0, self.num_concepts, size=(batch_size,))
		concept_token_ids = np.array([self.target_concept_token_ids[c] for c in concept_targets], dtype=np.int64)

		emotion_targets = np.random.randint(0, self.num_emotions, size=(batch_size,))
		emotion_token_ids = np.array([self.target_emotion_token_ids[e] for e in emotion_targets], dtype=np.int64)

		homeostasis_targets = np.random.randint(0, self.num_homeostasis, size=(batch_size,))
		homeostasis_token_ids = np.array([self.target_homeostasis_token_ids[h] for h in homeostasis_targets], dtype=np.int64)

		return concept_targets, concept_token_ids, emotion_targets, emotion_token_ids, homeostasis_targets, homeostasis_token_ids

	def get_concept_name(self, class_idx: int) -> str:
		"""Devuelve el nombre humano del concepto objetivo."""
		if 0 <= class_idx < self.num_concepts:
			return self.target_concepts[class_idx]
		return "desconocido"

	def get_emotion_name(self, class_idx: int) -> str:
		"""Devuelve el nombre humano del estado afectivo."""
		if 0 <= class_idx < self.num_emotions:
			return self.target_emotions[class_idx]
		return "desconocido"

	def get_homeostasis_name(self, class_idx: int) -> str:
		"""Devuelve el nombre humano del estado homeostático."""
		if 0 <= class_idx < self.num_homeostasis:
			return self.target_homeostasis[class_idx]
		return "desconocido"


class MathDatasetBreeder:
	"""
	Generador de Lotes Aritméticos para el Grado 1.
	Produce operaciones matemáticas A op B = R en el rango [0..10].
	"""

	def __init__(self, translator: SovereignTranslator):
		self.translator = translator

		self.operand_names = [
			"cero", "uno", "dos", "tres", "cuatro",
			"cinco", "seis", "siete", "ocho", "nueve", "diez"
		]
		self.operator_names = ["suma", "resta"]

		# Token IDs de Capa 1
		self.operand_token_ids = []
		for name in self.operand_names:
			tids = self.translator.encode(name)
			self.operand_token_ids.append(tids[0] if tids else 0)

		self.operator_token_ids = []
		for name in self.operator_names:
			tids = self.translator.encode(name)
			self.operator_token_ids.append(tids[0] if tids else 0)

		self.num_operands = len(self.operand_names)
		self.num_operators = len(self.operator_names)

	def generate_batch(self, batch_size: int) -> tuple:
		"""
		Genera un lote de operaciones consistentes.
		Devuelve:
		- operand_a_targets: Índices locales (0..10)
		- operand_a_token_ids: Token IDs del translator
		- operator_targets: Índices locales (0..1)
		- operator_token_ids: Token IDs del translator
		- operand_b_targets: Índices locales (0..10)
		- operand_b_token_ids: Token IDs del translator
		- result_targets: Índices locales (0..10)
		- result_token_ids: Token IDs del translator
		"""
		op_a_list = []
		op_b_list = []
		operator_list = []
		result_list = []

		for _ in range(batch_size):
			op_type = np.random.randint(0, 2)  # 0: suma, 1: resta
			if op_type == 0:
				a = np.random.randint(0, 11)
				b = np.random.randint(0, 11 - a)
				r = a + b
			else:
				a = np.random.randint(0, 11)
				b = np.random.randint(0, a + 1)
				r = a - b

			op_a_list.append(a)
			op_b_list.append(b)
			operator_list.append(op_type)
			result_list.append(r)

		op_a_targets = np.array(op_a_list, dtype=np.int64)
		op_b_targets = np.array(op_b_list, dtype=np.int64)
		operator_targets = np.array(operator_list, dtype=np.int64)
		result_targets = np.array(result_list, dtype=np.int64)

		op_a_token_ids = np.array([self.operand_token_ids[x] for x in op_a_targets], dtype=np.int64)
		op_b_token_ids = np.array([self.operand_token_ids[x] for x in op_b_targets], dtype=np.int64)
		operator_token_ids = np.array([self.operator_token_ids[x] for x in operator_targets], dtype=np.int64)
		result_token_ids = np.array([self.operand_token_ids[x] for x in result_targets], dtype=np.int64)

		return (
			op_a_targets, op_a_token_ids,
			operator_targets, operator_token_ids,
			op_b_targets, op_b_token_ids,
			result_targets, result_token_ids
		)

	def get_operand_name(self, class_idx: int) -> str:
		if 0 <= class_idx < self.num_operands:
			return self.operand_names[class_idx]
		return "desconocido"

	def get_operator_name(self, class_idx: int) -> str:
		if 0 <= class_idx < self.num_operators:
			return self.operator_names[class_idx]
		return "desconocido"

