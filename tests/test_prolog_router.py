
from src.router.swi_prolog_router import NodeTarget, SiliconTarget, route, route_semantic


def test_prolog_router_semantic_logic():
	"""Valida que la lógica de Prolog devuelva los expertos esperados."""
	# Código enrutado a code_node
	node, _ = route_semantic(domain="code_python", length=500)
	assert node == NodeTarget.CODE

	# Matemáticas enrutado a reason_node
	node, _ = route_semantic(domain="logic_math", length=500)
	assert node == NodeTarget.REASON

	# Síntesis enrutado a synth_node
	node, _ = route_semantic(domain="synth", length=500)
	assert node == NodeTarget.SYNTH

	# General enrutado a default_node
	node, _ = route_semantic(domain="general", length=500)
	assert node == NodeTarget.DEFAULT


def test_prolog_router_hardware_affinity():
	"""Valida que Prolog asigne el silicio adecuado según dominio y rasgos de contexto."""
	# 1. Código corto (< 1000 chars) -> NPU
	_, silicon = route_semantic(domain="code_python", length=500)
	assert silicon == SiliconTarget.NPU

	# 2. Código largo (>= 1000 chars) -> CUDA
	_, silicon = route_semantic(domain="code_python", length=1500)
	assert silicon == SiliconTarget.CUDA

	# 3. Matemáticas / Lógica -> CUDA
	_, silicon = route_semantic(domain="logic_math", length=100)
	assert silicon == SiliconTarget.CUDA

	# 4. Síntesis en background -> Vulkan
	_, silicon = route_semantic(domain="synth", length=100, latency="background")
	assert silicon == SiliconTarget.VULKAN

	# 5. General -> CPU (fallback)
	_, silicon = route_semantic(domain="general", length=100)
	assert silicon == SiliconTarget.CPU


def test_prolog_router_fact_retraction():
	"""Verifica que los hechos dinámicos se limpian correctamente entre consultas."""
	# Consulta 1: Configurar a código corto -> NPU
	_, silicon_1 = route_semantic(domain="code_python", length=500)
	assert silicon_1 == SiliconTarget.NPU

	# Consulta 2: Cambiar a general (si retractall fallara, acumularía hechos y daría resultados inconsistentes)
	node_2, silicon_2 = route_semantic(domain="general", length=100)
	assert node_2 == NodeTarget.DEFAULT
	assert silicon_2 == SiliconTarget.CPU


def test_prolog_router_retro_compatibility():
	"""Valida la compatibilidad de la función route(task) original mediante embeddings."""
	# Entrada de código
	node_code = route("Por favor escribe una función def sumar(a, b): return a + b")
	assert node_code == NodeTarget.CODE

	# Entrada de matemáticas
	node_math = route("Resuelve el cálculo usando lógica matemática")
	assert node_math == NodeTarget.REASON

	# Entrada general (determinista por palabra clave)
	node_gen = route("Hola Frankenswarm, ¿cómo te encuentras hoy?")
	assert node_gen == NodeTarget.DEFAULT

	# Entrada general (semántica por similitud a "saludo cordial")
	node_gen_sem = route("Te envío un saludo cordial")
	assert node_gen_sem == NodeTarget.DEFAULT


def test_routing_survives_a_dead_translator(monkeypatch):
	"""El camino semántico es infraestructura OPCIONAL: si cae, el router sigue.

	Regresión del 27-07-2026: `transformers` quedó inservible en el venv (conflicto
	huggingface-hub entre fastembed y transformers 5.x), el `except` lo tragaba en
	silencio y TODA tarea sin keyword acababa en default_node. Las reglas
	deterministas (RULE 2) tienen que bastar para lo nombrable.
	"""
	import src.router.swi_prolog_router as router

	def _dead_translator():
		raise ImportError("cannot import name 'is_offline_mode' from 'huggingface_hub'")

	monkeypatch.setattr(router, "get_translator", _dead_translator)

	assert router.route("Resuelve el cálculo usando lógica matemática") == NodeTarget.REASON
	assert router.route("Por favor escribe una función def sumar(a, b): return a + b") == NodeTarget.CODE
	assert router.route("Hola Frankenswarm, ¿cómo te encuentras hoy?") == NodeTarget.DEFAULT
