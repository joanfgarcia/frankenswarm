import numpy as np

from src.bitnet.operators import OperatorRegistry
from src.bitnet.translator import SovereignTranslator


class CompositionalMathDatasetBreeder:
	"""
	Generador de Lotes Aritméticos con Split de Generalización Sistemática.
	Divide las ecuaciones posibles en train/test splits disjuntos,
	asegurando que todos los tokens estén representados en el conjunto de entrenamiento.
	"""

	def __init__(self, translator: SovereignTranslator, split_ratio: float = 0.8, seed: int = 42, enabled_operators: list[str] = None):
		self.translator = translator
		self.split_ratio = split_ratio
		self.seed = seed

		self.operand_names = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez"]
		self.operator_names = enabled_operators if enabled_operators is not None else ["suma", "resta"]

		# Obtener Token IDs del translator
		self.operand_token_ids = []
		for name in self.operand_names:
			tids = self.translator.encode(name)
			self.operand_token_ids.append(tids[0] if tids else 0)

		# Mapear ID global del operador a su respectivo Token ID precalculado
		self.operator_token_ids_map = {}
		for name in self.operator_names:
			info = OperatorRegistry.get_operator_info(name)
			if info:
				op_idx = info["idx"]
				tids = self.translator.encode(name)
				self.operator_token_ids_map[op_idx] = tids[0] if tids else 0

		self.num_operands = len(self.operand_names)
		self.num_operators = len(self.operator_names)

		# Construir y dividir las ecuaciones
		self._build_and_split_equations()

	def _build_and_split_equations(self):
		all_eqs = OperatorRegistry.get_equations(self.operator_names)

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

			# Verificar que todos los operandos (0..10) y operadores activos aparezcan en train
			seen_operands = set()
			seen_operators = set()
			for a, op, b, r in train_eqs:
				seen_operands.add(a)
				seen_operands.add(b)
				seen_operands.add(r)
				seen_operators.add(op)

			if len(seen_operands) == len(self.operand_names) and len(seen_operators) == len(self.operator_names):
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
		- operator_targets: Índices locales (0..2) correspondientes al ID global del operador
		- operator_token_ids: Token IDs del translator
		- op_b_targets: Índices locales (0..10)
		- op_b_token_ids: Token IDs del translator
		- result_targets: Índices locales (0..10)
		- result_token_ids: Token IDs del translator
		"""
		eqs = custom_eqs if custom_eqs is not None else self.train_equations if mode == "train" else self.test_equations
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

		# Obtener Token IDs de operadores de forma instantánea a partir del mapa precalculado
		operator_token_ids = np.array([self.operator_token_ids_map[op] for op in operator_targets], dtype=np.int64)

		result_token_ids = np.array([self.operand_token_ids[x] for x in result_targets], dtype=np.int64)

		return (op_a_targets, op_a_token_ids, operator_targets, operator_token_ids, op_b_targets, op_b_token_ids, result_targets, result_token_ids)

	def get_operand_name(self, class_idx: int) -> str:
		if 0 <= class_idx < self.num_operands:
			return self.operand_names[class_idx]
		return "desconocido"

	def get_operator_name(self, class_idx: int) -> str:
		return OperatorRegistry.get_operator_name_by_idx(class_idx)

	@property
	def operator_token_ids(self):
		return self.operator_token_ids_map


class RelationalLogicDatasetBreeder:
	"""
	Generador de Lotes para Lógica Relacional (>, <, igual).
	Divide las 363 ecuaciones posibles en train/test splits disjuntos,
	garantizando cobertura completa de operandos, relaciones y resultados en el train set.
	"""

	def __init__(self, translator: SovereignTranslator, split_ratio: float = 0.9, seed: int = 42):
		self.translator = translator
		self.split_ratio = split_ratio
		self.seed = seed

		self.operand_names = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez"]
		self.relation_names = [">", "<", "igual"]
		self.result_names = ["verdad", "falsedad"]

		# Obtener Token IDs
		self.operand_token_ids = []
		for name in self.operand_names:
			tids = self.translator.encode(name)
			self.operand_token_ids.append(tids[0] if tids else 0)

		self.relation_token_ids = []
		for name in self.relation_names:
			tids = self.translator.encode(name)
			self.relation_token_ids.append(tids[0] if tids else 0)

		self.result_token_ids = []
		for name in self.result_names:
			tids = self.translator.encode(name)
			self.result_token_ids.append(tids[0] if tids else 0)

		self.num_operands = len(self.operand_names)
		self.num_relations = len(self.relation_names)
		self.num_results = len(self.result_names)

		self._build_and_split_equations()

	def _build_and_split_equations(self):
		all_eqs = []
		for a in range(11):
			for b in range(11):
				# > (mayor_que)
				r_gt = 0 if a > b else 1
				all_eqs.append((a, 0, b, r_gt))

				# < (menor_que)
				r_lt = 0 if a < b else 1
				all_eqs.append((a, 1, b, r_lt))

				# igual_a
				r_eq = 0 if a == b else 1
				all_eqs.append((a, 2, b, r_eq))

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

			# Verificar cobertura completa en train
			seen_operands = set()
			seen_relations = set()
			seen_results = set()
			for a, op, b, r in train_eqs:
				seen_operands.add(a)
				seen_operands.add(b)
				seen_relations.add(op)
				seen_results.add(r)

			if len(seen_operands) == 11 and len(seen_relations) == 3 and len(seen_results) == 2:
				self.train_equations = train_eqs
				self.test_equations = test_eqs
				break
			temp_seed += 1

	def generate_batch(self, batch_size: int, mode: str = "train") -> tuple:
		eqs = self.train_equations if mode == "train" else self.test_equations
		indices = np.random.randint(0, len(eqs), size=(batch_size,))

		op_a_list = []
		op_b_list = []
		relation_list = []
		result_list = []

		for idx in indices:
			a, op, b, r = eqs[idx]
			op_a_list.append(a)
			relation_list.append(op)
			op_b_list.append(b)
			result_list.append(r)

		op_a_targets = np.array(op_a_list, dtype=np.int64)
		op_b_targets = np.array(op_b_list, dtype=np.int64)
		relation_targets = np.array(relation_list, dtype=np.int64)
		result_targets = np.array(result_list, dtype=np.int64)

		op_a_token_ids = np.array([self.operand_token_ids[x] for x in op_a_targets], dtype=np.int64)
		op_b_token_ids = np.array([self.operand_token_ids[x] for x in op_b_targets], dtype=np.int64)
		relation_token_ids = np.array([self.relation_token_ids[x] for x in relation_targets], dtype=np.int64)
		result_token_ids = np.array([self.result_token_ids[x] for x in result_targets], dtype=np.int64)

		return (op_a_targets, op_a_token_ids, relation_targets, relation_token_ids, op_b_targets, op_b_token_ids, result_targets, result_token_ids)

	def get_operand_name(self, class_idx: int) -> str:
		return self.operand_names[class_idx]

	def get_relation_name(self, class_idx: int) -> str:
		return self.relation_names[class_idx]

	def get_result_name(self, class_idx: int) -> str:
		return self.result_names[class_idx]
