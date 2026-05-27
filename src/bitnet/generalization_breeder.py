import numpy as np

from src.bitnet.translator import SovereignTranslator


class CompositionalMathDatasetBreeder:
	"""
	Generador de Lotes Aritméticos con Split de Generalización Sistemática.
	Divide las 132 ecuaciones posibles en train/test splits disjuntos,
	asegurando que todos los tokens estén representados en el conjunto de entrenamiento.
	"""

	def __init__(self, translator: SovereignTranslator, split_ratio: float = 0.8, seed: int = 42):
		self.translator = translator
		self.split_ratio = split_ratio
		self.seed = seed

		self.operand_names = [
			"cero", "uno", "dos", "tres", "cuatro",
			"cinco", "seis", "siete", "ocho", "nueve", "diez"
		]
		self.operator_names = ["suma", "resta"]

		# Obtener Token IDs del translator
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

		# Construir y dividir las ecuaciones
		self._build_and_split_equations()

	def _build_and_split_equations(self):
		all_eqs = []
		# Sumas: A + B = R (66 combinaciones)
		for a in range(11):
			for b in range(11 - a):
				all_eqs.append((a, 0, b, a + b))
		# Restas: A - B = R (66 combinaciones)
		for a in range(11):
			for b in range(a + 1):
				all_eqs.append((a, 1, b, a - b))

		# Generador pseudoaleatorio con semilla fija
		rng = np.random.default_rng(self.seed)
		
		# Para garantizar que todos los tokens aparezcan en el conjunto de train,
		# podemos forzar un subconjunto mínimo inicial de ecuaciones y luego barajar el resto.
		# Cada número del 0 al 10 y cada operador debe verse en el train set.
		
		# Intentar particionar aleatoriamente y verificar cobertura de tokens
		# Si falla la cobertura, barajar nuevamente (con diferentes semillas de control)
		temp_seed = self.seed
		while True:
			rng = np.random.default_rng(temp_seed)
			indices = np.arange(len(all_eqs))
			rng.shuffle(indices)

			split_idx = int(len(all_eqs) * self.split_ratio)
			train_indices = indices[:split_idx]
			test_indices = indices[split_idx:]

			train_eqs = [all_eqs[i] for i in train_indices]
			test_eqs = [all_eqs[i] for i in test_indices]

			# Verificar que todos los operandos (0..10) y operadores (0..1) aparezcan en train
			seen_operands = set()
			seen_operators = set()
			for a, op, b, r in train_eqs:
				seen_operands.add(a)
				seen_operands.add(b)
				seen_operands.add(r)
				seen_operators.add(op)

			if len(seen_operands) == 11 and len(seen_operators) == 2:
				self.train_equations = train_eqs
				self.test_equations = test_eqs
				break
			else:
				temp_seed += 1

	def generate_batch(self, batch_size: int, mode: str = "train", custom_eqs: list = None) -> tuple:
		"""
		Genera un lote de operaciones consistentes a partir del split seleccionado.
		Devuelve:
		- op_a_targets: Índices locales (0..10)
		- op_a_token_ids: Token IDs del translator
		- operator_targets: Índices locales (0..1)
		- operator_token_ids: Token IDs del translator
		- op_b_targets: Índices locales (0..10)
		- op_b_token_ids: Token IDs del translator
		- result_targets: Índices locales (0..10)
		- result_token_ids: Token IDs del translator
		"""
		if custom_eqs is not None:
			eqs = custom_eqs
		else:
			eqs = self.train_equations if mode == "train" else self.test_equations
		indices = np.random.randint(0, len(eqs), size=(batch_size,))

		op_a_list = []
		op_b_list = []
		operator_list = []
		result_list = []

		for idx in indices:
			a, op, b, r = eqs[idx]
			op_a_list.append(a)
			operator_list.append(op)
			op_b_list.append(b)
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
