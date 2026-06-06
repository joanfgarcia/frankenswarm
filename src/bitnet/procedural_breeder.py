"""
Procedural Logic Breeder para Frankenswarm.

Genera un ecosistema de lógica proposicional procedimental compuesto de 60-100 conceptos
conectados a través de reglas causales aleatorias con topología controlada (DAG por capas)
y contradicciones semánticas disjuntas.

Diseñado para soportar:
  - Inferencia transitiva (cadenas de 3 a 5 saltos)
  - Modus tollens (razonamiento hacia atrás)
  - Modulación afectiva y asimetría de pérdida por miedo
  - Integración transparente en el pipeline de PopuLoRA
"""

import numpy as np

from src.bitnet.operators import OperatorRegistry
from src.bitnet.translator import SovereignTranslator


class ProceduralLogicDatasetBreeder:
	"""
	Generador de Lotes Lógicos Procedimentales con Topología Controlada.
	Crea un grafo causal sintético pero estructurado sobre un vocabulario procedural.

	Parámetros:
		translator: SovereignTranslator para obtener token IDs.
		num_concepts: Número total de conceptos a generar (60-100).
		split_ratio: Ratio para dividir las ecuaciones en train/test.
		seed: Semilla aleatoria para la reproducibilidad de la topología.
		fear_amplifier: Multiplicador para ponderar la pérdida según el miedo.
		num_layers: Número de capas en la topología DAG.
		edges_per_node: Número de conexiones salientes por nodo a la siguiente capa.
		num_contradictions: Número de pares de contradicción.
	"""

	def __init__(
		self,
		translator: SovereignTranslator,
		num_concepts: int = 80,
		split_ratio: float = 0.90,
		seed: int = 42,
		fear_amplifier: float = 2.0,
		num_layers: int = 4,
		edges_per_node: int = 2,
		num_contradictions: int = 15,
	):
		self.translator = translator
		self.num_concepts = num_concepts
		self.split_ratio = split_ratio
		self.seed = seed
		self.fear_amplifier = fear_amplifier
		self.num_layers = num_layers
		self.edges_per_node = edges_per_node
		self.num_contradictions = num_contradictions

		# Conceptos representados por marcadores precalculados en el vocabulario (token_1000+)
		self.concept_names = [f"token_{1000 + i}" for i in range(self.num_concepts)]
		self.result_names = ["verdad", "falsedad"]
		self.emotion_names = ["miedo", "alegría", "ira", "tristeza", "dolor", "hambre"]

		# Obtener Token IDs reales del traductor
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

		# Mapear operadores lógicos a Token IDs
		self.enabled_operators = ["implica", "contradice", "cadena", "niega"]
		self.operator_token_ids_map = {}
		self.operator_indices_map = {}
		for name in self.enabled_operators:
			info = OperatorRegistry.get_operator_info(name)
			if info:
				op_idx = info["idx"]
				tids = self.translator.encode(name)
				self.operator_token_ids_map[op_idx] = tids[0] if tids else 0
				self.operator_indices_map[name] = op_idx
			else:
				# Fallback en caso de que no estén registrados en el OperatorRegistry
				# Asignar índices por defecto: implica=3, contradice=4, cadena=5, niega=6
				op_idx = {"implica": 3, "contradice": 4, "cadena": 5, "niega": 6}[name]
				tids = self.translator.encode(name)
				self.operator_token_ids_map[op_idx] = tids[0] if tids else 0
				self.operator_indices_map[name] = op_idx

		# Semilla de NumPy local para la generación procedimental del grafo
		self.rng = np.random.default_rng(self.seed)

		# Construir topología procedimental
		self._generate_topology()

		# Generar y dividir ecuaciones en train/test
		self._build_and_split_equations()

	def _generate_topology(self):
		"""
		Construye la topología del grafo causal y las relaciones lógicas.
		Divide los conceptos en capas para garantizar transitividades largas sin ciclos.
		"""
		# 1. Asignar conceptos a capas
		# Asegurar distribución equitativa de conceptos en las capas
		self.concept_layers = []
		concepts_per_layer = self.num_concepts // self.num_layers
		for i in range(self.num_layers):
			start_idx = i * concepts_per_layer
			# La última capa absorbe los restantes
			end_idx = (i + 1) * concepts_per_layer if i < self.num_layers - 1 else self.num_concepts
			self.concept_layers.append(list(range(start_idx, end_idx)))

		# 2. Generar conexiones Layer i -> Layer i+1
		self.causal_rules = []  # List of (causa, efecto, fear_weight)
		self.causal_adj = {u: [] for u in range(self.num_concepts)}

		for i in range(self.num_layers - 1):
			current_layer = self.concept_layers[i]
			next_layer = self.concept_layers[i + 1]

			for u in current_layer:
				# Elegir aleatoriamente destinos en la siguiente capa
				destinations = self.rng.choice(next_layer, size=min(self.edges_per_node, len(next_layer)), replace=False)
				for v in destinations:
					# Peso de miedo: 10% probabilidad de peligro alto (>0.8), resto moderado
					is_dangerous = self.rng.random() < 0.10
					fear_weight = self.rng.uniform(0.8, 1.0) if is_dangerous else self.rng.uniform(0.1, 0.4)
					self.causal_rules.append((u, v, float(fear_weight)))
					self.causal_adj[u].append((v, float(fear_weight)))

		# Mapear cada concepto a una emoción dominante aleatoriamente
		# Excepto los nodos conectados a reglas de alto miedo, que serán mapeados a "miedo"
		self.concept_emotions = []
		self.concept_base_fear = np.zeros(self.num_concepts)

		for u in range(self.num_concepts):
			# Encontrar el miedo máximo asociado a las reglas en las que participa el nodo
			relevant_rules = [r[2] for r in self.causal_rules if r[0] == u or r[1] == u]
			max_rule_fear = max(relevant_rules) if relevant_rules else 0.0
			self.concept_base_fear[u] = max_rule_fear

			if max_rule_fear > 0.7:
				self.concept_emotions.append("miedo")
			else:
				# Asignar emoción aleatoria de las 5 restantes (excluyendo miedo)
				self.concept_emotions.append(self.rng.choice(self.emotion_names[1:]))

		# 3. Generar contradicciones
		# Seleccionar pares disjuntos aleatoriamente
		self.contradictions = set()
		flat_concepts = list(range(self.num_concepts))
		self.rng.shuffle(flat_concepts)

		for k in range(min(self.num_contradictions, self.num_concepts // 2)):
			u = flat_concepts[2 * k]
			v = flat_concepts[2 * k + 1]
			self.contradictions.add(frozenset({u, v}))

	def _find_all_chains(self) -> list[tuple[int, int, float]]:
		"""
		Busca todas las relaciones de transitividad (cadenas lógicas) en el DAG.
		Retorna una lista de (origen, destino, max_fear) para caminos de longitud >= 2.
		"""
		chains = []
		direct_pairs = {(u, v) for u, v, _ in self.causal_rules}

		def dfs(start: int, current: int, depth: int, max_fear: float):
			if depth >= 2 and (start, current) not in direct_pairs:
				chains.append((start, current, max_fear))

			# Continuar DFS a través de los vecinos
			for next_node, fear in self.causal_adj[current]:
				dfs(start, next_node, depth + 1, max(max_fear, fear))

		for start in range(self.num_concepts):
			for next_node, fear in self.causal_adj[start]:
				dfs(start, next_node, 1, fear)

		# Eliminar duplicados manteniendo el miedo máximo
		unique_chains = {}
		for u, v, fear in chains:
			if (u, v) not in unique_chains or fear > unique_chains[(u, v)]:
				unique_chains[(u, v)] = fear

		return [(u, v, f) for (u, v), f in unique_chains.items()]

	def _build_and_split_equations(self):
		""" Genera todas las ecuaciones y realiza la partición train/test. """
		all_eqs = []
		verdad = 0
		falsedad = 1

		# 1. IMPLICA (Modus Ponens)
		op_implica = self.operator_indices_map["implica"]
		valid_implications = {(u, v) for u, v, _ in self.causal_rules}
		for u, v, _ in self.causal_rules:
			all_eqs.append((u, op_implica, v, verdad))

		# Negativos de implica (muestreo equilibrado)
		for u in range(self.num_concepts):
			neg_count = 0
			# Generar hasta 3 negativos por cada positivo del nodo
			max_negs = max(1, len(self.causal_adj[u]) * 3)
			candidates = list(range(self.num_concepts))
			self.rng.shuffle(candidates)
			for v in candidates:
				if u != v and (u, v) not in valid_implications:
					all_eqs.append((u, op_implica, v, falsedad))
					neg_count += 1
					if neg_count >= max_negs:
						break

		# 2. CONTRADICE
		op_contradice = self.operator_indices_map["contradice"]
		for u in range(self.num_concepts):
			for v in range(u + 1, self.num_concepts):
				is_contradict = frozenset({u, v}) in self.contradictions
				all_eqs.append((u, op_contradice, v, verdad if is_contradict else falsedad))
				all_eqs.append((v, op_contradice, u, verdad if is_contradict else falsedad))
			# Un concepto no se contradice a sí mismo
			all_eqs.append((u, op_contradice, u, falsedad))

		# 3. CADENA (Transitividad)
		op_cadena = self.operator_indices_map["cadena"]
		chains = self._find_all_chains()
		chain_pairs = {(u, v) for u, v, _ in chains}

		for u, v, _ in chains:
			all_eqs.append((u, op_cadena, v, verdad))

		# Negativos de cadena (conceptos sin camino de transitividad)
		# Muestrear proporcionalmente para evitar explosión de datos
		all_starts = list({c[0] for c in chains})
		for u in all_starts:
			neg_count = 0
			candidates = list(range(self.num_concepts))
			self.rng.shuffle(candidates)
			for v in candidates:
				if u != v and (u, v) not in chain_pairs and (u, v) not in valid_implications:
					all_eqs.append((u, op_cadena, v, falsedad))
					neg_count += 1
					if neg_count >= max(2, len([c for c in chains if c[0] == u]) * 2):
						break

		# 4. NIEGA (Modus Tollens)
		op_niega = self.operator_indices_map["niega"]
		# Relación transitiva completa: A -> B (directo o indirecto)
		complete_paths = valid_implications | chain_pairs

		for u, v in complete_paths:
			# Si u implica v, entonces ¬v niega u
			all_eqs.append((v, op_niega, u, verdad))

		# Negativos de niega
		for v in range(self.num_concepts):
			neg_count = 0
			candidates = list(range(self.num_concepts))
			self.rng.shuffle(candidates)
			for u in candidates:
				if u != v and (u, v) not in complete_paths:
					all_eqs.append((v, op_niega, u, falsedad))
					neg_count += 1
					if neg_count >= 3:
						break

		# Realizar partición train/test
		temp_seed = self.seed
		while True:
			rng = np.random.default_rng(temp_seed)
			indices = np.arange(len(all_eqs))
			rng.shuffle(indices)

			split_idx = int(len(all_eqs) * self.split_ratio)
			train_indices = indices[:split_idx]
			test_indices = indices[split_idx:]

			train_eqs = [all_eqs[k] for k in train_indices]
			test_eqs = [all_eqs[k] for k in test_indices]

			# Verificar que el train set tenga cobertura de todos los conceptos y operadores
			seen_concepts = set()
			seen_ops = set()
			seen_results = set()

			for u, op, v, r in train_eqs:
				seen_concepts.add(u)
				seen_concepts.add(v)
				seen_ops.add(op)
				seen_results.add(r)

			if (
				len(seen_concepts) >= self.num_concepts
				and len(seen_ops) == len(self.enabled_operators)
				and len(seen_results) == 2
			):
				self.train_equations = train_eqs
				self.test_equations = test_eqs
				break
			temp_seed += 1

	def _get_fear_weight(self, concept_a: int, concept_b: int) -> float:
		""" Calcula el peso asimétrico del miedo para la función de pérdida. """
		base_fear_a = self.concept_base_fear[concept_a]
		base_fear_b = self.concept_base_fear[concept_b]
		combined_fear = max(base_fear_a, base_fear_b)

		return 1.0 + combined_fear * self.fear_amplifier

	def _get_emotion_id(self, concept_idx: int) -> int:
		""" Retorna el índice local de la emoción para un concepto. """
		emotion_name = self.concept_emotions[concept_idx]
		return self.emotion_names.index(emotion_name)

	def generate_batch(self, batch_size: int, mode: str = "train", custom_eqs: list = None) -> tuple:
		"""
		Genera un lote de tensores de ecuaciones y metadatos afectivos.

		Returns:
			ca_targets: Índices locales de concepto A (batch_size,)
			ca_tids: Token IDs globales de concepto A (batch_size,)
			op_targets: Índices del operador en el registro (batch_size,)
			op_tids: Token IDs del operador (batch_size,)
			cb_targets: Índices locales de concepto B (batch_size,)
			cb_tids: Token IDs globales de concepto B (batch_size,)
			r_targets: Índices del resultado (0=verdad, 1=falsedad) (batch_size,)
			r_tids: Token IDs globales del resultado (batch_size,)
			emo_tids: Token IDs de la emoción dominante (batch_size,)
			fear_weights: Pesos asimétricos de la pérdida (batch_size,)
		"""
		eqs = custom_eqs if custom_eqs is not None else (self.train_equations if mode == "train" else self.test_equations)
		indices = np.random.randint(0, len(eqs), size=(batch_size,))

		ca_list, cb_list, op_list, r_list = [], [], [], []
		emo_list, fear_list = [], []

		for idx in indices:
			u, op, v, r = eqs[idx]
			ca_list.append(u)
			op_list.append(op)
			cb_list.append(v)
			r_list.append(r)
			emo_list.append(self._get_emotion_id(u))
			fear_list.append(self._get_fear_weight(u, v))

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

	def get_concept_name(self, idx: int) -> str:
		if 0 <= idx < self.num_concepts:
			return self.concept_names[idx]
		return "desconocido"

	def get_operator_name(self, idx: int) -> str:
		return OperatorRegistry.get_operator_name_by_idx(idx)

	def get_result_name(self, idx: int) -> str:
		if 0 <= idx < len(self.result_names):
			return self.result_names[idx]
		return "desconocido"

	def get_emotion_name(self, idx: int) -> str:
		if 0 <= idx < len(self.emotion_names):
			return self.emotion_names[idx]
		return "desconocido"

	@property
	def operand_token_ids(self):
		""" Compatibilidad con la suite de evaluación. """
		return self.concept_token_ids

	@property
	def relation_token_ids(self):
		""" Compatibilidad con la suite de evaluación. """
		return list(self.operator_token_ids_map.values())


if __name__ == "__main__":
	# Test de validación
	print("🧪 Inicializando test del ProceduralLogicDatasetBreeder...")
	trans = SovereignTranslator()
	breeder = ProceduralLogicDatasetBreeder(trans, num_concepts=80, split_ratio=0.85, seed=123)

	print(f"✅ Grafo causal generado con {len(breeder.causal_rules)} reglas de implicación directa.")
	print(f"✅ Contradicciones registradas: {len(breeder.contradictions)} pares.")
	print(f"📊 Partición procedural: Train={len(breeder.train_equations)} | Test={len(breeder.test_equations)}")

	# Verificar transitividad
	chains = breeder._find_all_chains()
	print(f"⛓️ Cadenas transitivas encontradas: {len(chains)}")

	# Generar un lote
	print("\n🔄 Generando lote de prueba...")
	batch = breeder.generate_batch(5, mode="train")
	for idx in range(5):
		c_a = breeder.get_concept_name(batch[0][idx])
		op = breeder.get_operator_name(batch[2][idx])
		c_b = breeder.get_concept_name(batch[4][idx])
		res = breeder.get_result_name(batch[6][idx])
		emo = breeder.get_emotion_name(breeder.emotion_token_ids.index(batch[8][idx]))
		fear = batch[9][idx]
		print(f"  [{idx + 1}] {c_a} ({emo}) {op} {c_b} = {res} | Miedo: {fear:.2f}")
