class OperatorRegistry:
	_registry = {}

	@classmethod
	def register(cls, name: str, op_idx: int):
		"""Decorador para registrar un generador de ecuaciones aritméticas."""

		def decorator(func):
			cls._registry[name] = {"generator": func, "idx": op_idx}
			return func

		return decorator

	@classmethod
	def get_equations(cls, enabled_operators: list[str]) -> list[tuple[int, int, int, int]]:
		"""Obtiene todas las ecuaciones combinadas para los operadores activos."""
		all_eqs = []
		for name in enabled_operators:
			if name in cls._registry:
				op_info = cls._registry[name]
				all_eqs.extend(op_info["generator"](op_info["idx"]))
		return all_eqs

	@classmethod
	def get_operator_info(cls, name: str) -> dict | None:
		"""Obtiene la información (idx, generador) de un operador por nombre."""
		return cls._registry.get(name)

	@classmethod
	def get_operator_name_by_idx(cls, idx: int) -> str:
		"""Obtiene el nombre del operador asociado a un índice local."""
		for name, info in cls._registry.items():
			if info["idx"] == idx:
				return name
		return "desconocido"


@OperatorRegistry.register("suma", 0)
def gen_suma(op_idx: int) -> list[tuple[int, int, int, int]]:
	"""Genera las 66 combinaciones únicas de suma A + B = R con R <= 10."""
	eqs = []
	for a in range(11):
		for b in range(11 - a):
			eqs.append((a, op_idx, b, a + b))
	return eqs


@OperatorRegistry.register("resta", 1)
def gen_resta(op_idx: int) -> list[tuple[int, int, int, int]]:
	"""Genera las 66 combinaciones únicas de resta A - B = R con R >= 0 y A, B <= 10."""
	eqs = []
	for a in range(11):
		for b in range(a + 1):
			eqs.append((a, op_idx, b, a - b))
	return eqs


@OperatorRegistry.register("multiplica", 2)
def gen_multiplica(op_idx: int) -> list[tuple[int, int, int, int]]:
	"""Genera las 48 combinaciones únicas de multiplicación A * B = R con R <= 10 y A, B <= 10."""
	eqs = []
	for a in range(11):
		for b in range(11):
			if a * b <= 10:
				eqs.append((a, op_idx, b, a * b))
	return eqs
