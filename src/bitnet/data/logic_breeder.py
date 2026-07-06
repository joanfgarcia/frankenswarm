"""
Breeder de Lógica Proposicional con vectores de Emoción y Miedo.

Genera lotes para entrenamiento de inferencia causal (modus ponens)
y detección de contradicciones, incluyendo los vectores afectivos
del ReferentialDatasetBreeder original.

Formato de salida: 4 slots como el resto del sistema
  [concepto_a, operador, concepto_b, resultado]

Pero además genera vectores paralelos de:
  - emotion_ids: estado emocional asociado al contexto
  - fear_weights: peso de pérdida asimétrica (equivocarse en peligro duele más)
"""

import numpy as np

from src.bitnet.operators.operators import OperatorRegistry
from src.bitnet.operators.operators_logic import CAUSAL_RULES, CONCEPT_NAMES
from src.bitnet.translation.translator import SovereignTranslator

# Mapeo concepto → emoción dominante
# Cada concepto evoca una emoción primaria
CONCEPT_EMOTION_MAP = {
	0: "alegría",  # gato → alegría
	1: "alegría",  # perro → alegría
	2: "alegría",  # casa → alegría (seguridad)
	3: "alegría",  # árbol → alegría (naturaleza)
	4: "alegría",  # agua → alegría (vida)
	5: "miedo",  # fuego → miedo
	6: "tristeza",  # tierra → tristeza (ceniza, destrucción)
	7: "alegría",  # aire → alegría (libertad)
	8: "alegría",  # sol → alegría
	9: "tristeza",  # luna → tristeza (soledad)
	10: "miedo",  # peligro → miedo
	11: "alegría",  # seguridad → alegría
}

# Mapeo concepto → nivel de miedo base [0.0 - 1.0]
CONCEPT_FEAR_MAP = {
	0: 0.1,  # gato
	1: 0.1,  # perro
	2: 0.0,  # casa
	3: 0.0,  # árbol
	4: 0.1,  # agua
	5: 0.9,  # fuego
	6: 0.3,  # tierra
	7: 0.0,  # aire
	8: 0.1,  # sol
	9: 0.2,  # luna
	10: 1.0,  # peligro
	11: 0.0,  # seguridad
}


class PropositionalLogicBreeder:
	"""
	Generador de Lotes para Lógica Proposicional con Emoción y Miedo.

	Extiende el esquema relacional (EXP_023/024) con:
	- Conceptos en vez de números
	- Operadores lógicos (implica, contradice) en vez de aritméticos
	- Vectores de emoción y miedo que modulan la función de pérdida

	Parámetros:
		translator: SovereignTranslator para obtener token IDs
		enabled_operators: lista de operadores lógicos activos
		split_ratio: ratio train/test
		seed: semilla para reproducibilidad
		fear_amplifier: multiplicador del vector de miedo en la pérdida
	"""

	def __init__(
		self,
		translator: SovereignTranslator,
		enabled_operators: list[str] | None = None,
		split_ratio: float = 0.90,
		seed: int = 42,
		fear_amplifier: float = 2.0,
	):
		self.translator = translator
		self.split_ratio = split_ratio
		self.seed = seed
		self.fear_amplifier = fear_amplifier

		# Conceptos (operandos del mundo lógico)
		self.concept_names = list(CONCEPT_NAMES)
		self.num_concepts = len(self.concept_names)

		# Resultados booleanos
		self.result_names = ["verdad", "falsedad"]
		self.num_results = len(self.result_names)

		# Emociones disponibles
		self.emotion_names = ["miedo", "alegría", "ira", "tristeza", "dolor", "hambre"]

		# Operadores activos
		if enabled_operators is None:
			enabled_operators = ["implica", "contradice"]
		self.enabled_operators = enabled_operators

		# ── Token IDs ──
		self.concept_token_ids = []
		for name in self.concept_names:
			tids = self.translator.encode(name)
			self.concept_token_ids.append(tids[0] if tids else 0)

		self.result_token_ids = []
		for name in self.result_names:
			tids = self.translator.encode(name)
			self.result_token_ids.append(tids[0] if tids else 0)

		self.emotion_token_ids = []
		for name in self.emotion_names:
			tids = self.translator.encode(name)
			self.emotion_token_ids.append(tids[0] if tids else 0)

		# Token IDs de operadores lógicos
		self.operator_token_ids_map = {}
		for name in self.enabled_operators:
			info = OperatorRegistry.get_operator_info(name)
			if info:
				op_idx = info["idx"]
				tids = self.translator.encode(name)
				self.operator_token_ids_map[op_idx] = tids[0] if tids else 0

		# ── Construir reglas de miedo indexadas por (causa, efecto) ──
		self.fear_rules = {}
		for causa, efecto, fear in CAUSAL_RULES:
			self.fear_rules[(causa, efecto)] = fear

		# ── Build & Split ──
		self._build_and_split_equations()

	def _build_and_split_equations(self):
		"""Genera todas las ecuaciones lógicas y las divide en train/test."""
		all_eqs = OperatorRegistry.get_equations(self.enabled_operators)

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

			# Verificar cobertura: todos los conceptos y resultados en train
			seen_concepts = set()
			seen_operators = set()
			seen_results = set()
			for a, op, b, r in train_eqs:
				seen_concepts.add(a)
				seen_concepts.add(b)
				seen_operators.add(op)
				seen_results.add(r)

			if (
				len(seen_concepts) >= self.num_concepts
				and len(seen_operators) == len(self.enabled_operators)
				and len(seen_results) == self.num_results
			):
				self.train_equations = train_eqs
				self.test_equations = test_eqs
				break
			temp_seed += 1

	def _get_fear_weight(self, concept_a: int, concept_b: int) -> float:
		"""Calcula el peso de miedo para una ecuación dada."""
		# Miedo base del concepto causa
		base_fear = CONCEPT_FEAR_MAP.get(concept_a, 0.0)

		# Miedo adicional de la regla causal específica
		rule_fear = self.fear_rules.get((concept_a, concept_b), 0.0)

		# Combinación: el máximo de ambos, amplificado
		combined = max(base_fear, rule_fear)
		return 1.0 + combined * self.fear_amplifier

	def _get_emotion_id(self, concept_idx: int) -> int:
		"""Obtiene el índice de emoción para un concepto."""
		emotion_name = CONCEPT_EMOTION_MAP.get(concept_idx, "alegría")
		if emotion_name in self.emotion_names:
			return self.emotion_names.index(emotion_name)
		return 1  # default: alegría

	def generate_batch(
		self,
		batch_size: int,
		mode: str = "train",
		custom_eqs: list = None,
	) -> tuple:
		"""
		Genera un lote de ecuaciones lógicas con vectores de emoción y miedo.

		Returns:
			concept_a_targets: Índices locales de concepto A
			concept_a_token_ids: Token IDs de concepto A
			operator_targets: Índices globales de operador
			operator_token_ids: Token IDs de operador
			concept_b_targets: Índices locales de concepto B
			concept_b_token_ids: Token IDs de concepto B
			result_targets: Índices locales de resultado (0=verdad, 1=falsedad)
			result_token_ids: Token IDs de resultado
			emotion_token_ids: Token IDs del estado emocional del contexto
			fear_weights: Pesos de pérdida asimétrica (float array)
		"""
		eqs = custom_eqs if custom_eqs is not None else (self.train_equations if mode == "train" else self.test_equations)
		indices = np.random.randint(0, len(eqs), size=(batch_size,))

		ca_list, cb_list, op_list, r_list = [], [], [], []
		emo_list, fear_list = [], []

		for idx in indices:
			a, op, b, r = eqs[idx]
			ca_list.append(a)
			op_list.append(op)
			cb_list.append(b)
			r_list.append(r)
			emo_list.append(self._get_emotion_id(a))
			fear_list.append(self._get_fear_weight(a, b))

		ca_targets = np.array(ca_list, dtype=np.int64)
		cb_targets = np.array(cb_list, dtype=np.int64)
		op_targets = np.array(op_list, dtype=np.int64)
		r_targets = np.array(r_list, dtype=np.int64)

		ca_tids = np.array([self.concept_token_ids[x] for x in ca_targets], dtype=np.int64)
		cb_tids = np.array([self.concept_token_ids[x] for x in cb_targets], dtype=np.int64)
		op_tids = np.array([self.operator_token_ids_map[op] for op in op_targets], dtype=np.int64)
		r_tids = np.array([self.result_token_ids[x] for x in r_targets], dtype=np.int64)

		emo_tids = np.array([self.emotion_token_ids[e] for e in emo_list], dtype=np.int64)
		fear_weights = np.array(fear_list, dtype=np.float32)

		return (
			ca_targets,
			ca_tids,
			op_targets,
			op_tids,
			cb_targets,
			cb_tids,
			r_targets,
			r_tids,
			emo_tids,
			fear_weights,
		)

	# ── Helpers de debug ──

	def get_concept_name(self, idx: int) -> str:
		if 0 <= idx < self.num_concepts:
			return self.concept_names[idx]
		return "desconocido"

	def get_operator_name(self, idx: int) -> str:
		return OperatorRegistry.get_operator_name_by_idx(idx)

	def get_result_name(self, idx: int) -> str:
		if 0 <= idx < self.num_results:
			return self.result_names[idx]
		return "desconocido"

	def get_emotion_name(self, idx: int) -> str:
		if 0 <= idx < len(self.emotion_names):
			return self.emotion_names[idx]
		return "desconocido"

	# ── Propiedades de compatibilidad ──

	@property
	def operand_token_ids(self):
		"""Compatibilidad con el evaluador genérico."""
		return self.concept_token_ids

	@property
	def relation_token_ids(self):
		"""Compatibilidad con el evaluador genérico."""
		return list(self.operator_token_ids_map.values())
