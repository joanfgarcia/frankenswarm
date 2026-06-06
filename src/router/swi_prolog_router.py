"""
Frankenswarm — SWI-Prolog Sovereign Router
====================================================
Declarative rule-based router using SWI-Prolog via pyswip.
This handles dynamic fact asserting and logical enrouting.
"""

from __future__ import annotations

import contextlib
import os
import sys
from enum import StrEnum

# ===========================================================================
# PYSWIP / DYNAMIC LIBRARY TOLERANCE LOADER
# ===========================================================================
# In Linux, if ctypes cannot find swipl, we try a fallback to the standard
# location for Debian-based distros to prevent crash on import.
if sys.platform.startswith("linux"):
	import ctypes.util

	try:
		if not ctypes.util.find_library("swipl"):
			fallback_path = "/usr/lib/x86_64-linux-gnu/libswipl.so"
			if os.path.exists(fallback_path):
				# Pre-load the library into process space so pyswip sees it
				ctypes.CDLL(fallback_path)
	except Exception:
		pass


class NodeTarget(StrEnum):
	CODE = "code_node"
	REASON = "reason_node"
	SYNTH = "synth_node"
	DEFAULT = "default_node"


class SiliconTarget(StrEnum):
	NPU = "npu"
	CUDA = "cuda"
	CPU = "cpu"
	VULKAN = "vulkan"


# Lazy-loaded singleton to avoid loading sentence_transformers on import
_translator = None


def get_translator():
	"""Lazy-loads the VectorTranslator and initializes default concepts."""
	global _translator
	if _translator is None:
		from src.tokenizer.translator import VectorTranslator

		_translator = VectorTranslator()
		# Seed it with default concepts to allow classification
		ontology = [
			"código python",
			"lógica matemática",
			"resumen general",
			"error de sintaxis",
			"saludo cordial",
		]
		_translator.add_concepts(ontology)
	return _translator


def route_semantic(domain: str, length: int, latency: str = "normal") -> tuple[NodeTarget, SiliconTarget]:
	"""
	Routes a query semantically by asserting facts to SWI-Prolog dynamically
	and evaluating the expert and silicon predicates.

	Args:
		domain: The classified concept domain (e.g. 'code_python', 'logic_math', 'synth', 'general').
		length: Length of the task prompt in characters.
		latency: The latency tier ('normal', 'background', etc.)

	Returns:
		A tuple containing the NodeTarget and SiliconTarget.
	"""
	try:
		from pyswip import Prolog
	except ImportError:
		print("[WARNING] pyswip not installed. Fallback to CPU/DEFAULT.")
		return NodeTarget.DEFAULT, SiliconTarget.CPU

	prolog = Prolog()
	rules_path = os.path.join(os.path.dirname(__file__), "router_rules.pl")
	prolog.consult(rules_path)

	# Clean dynamic facts from previous queries to avoid accumulation
	prolog.retractall("query_property(_, _)")

	# Assert current query facts
	prolog.assertz(f"query_property(domain, {domain})")
	prolog.assertz(f"query_property(length, {length})")
	prolog.assertz(f"query_property(latency, {latency})")

	# Query logical target node (expert)
	node_target = NodeTarget.DEFAULT
	results_expert = list(prolog.query("route_expert(Node)"))
	if results_expert:
		target_str = results_expert[0]["Node"]
		with contextlib.suppress(ValueError):
			node_target = NodeTarget(target_str)

	# Query hardware target node (silicon)
	silicon_target = SiliconTarget.CPU
	results_silicon = list(prolog.query("route_silicon(Silicon)"))
	if results_silicon:
		silicon_str = results_silicon[0]["Silicon"]
		with contextlib.suppress(ValueError):
			silicon_target = SiliconTarget(silicon_str)

	return node_target, silicon_target


def route(task: str) -> NodeTarget:
	"""
	Evaluates the Prolog rules to determine the expert node.
	Provides backwards compatibility with the original route(task: str) signature.
	"""
	# Deterministic pre-checks for high-confidence structural keywords
	if any(kw in task for kw in ("def ", "class ", "import ", "return ", "```python")) or any(kw in task for kw in ("SELECT ", "INSERT ", "UPDATE ", "CREATE TABLE")):
		domain = "code_python"
	elif any(kw in task.lower() for kw in ("why ", "explain ", "what is ", "how does ")) or any(kw in task.lower() for kw in ("suma", "resta", "multiplica", ">", "<", "igual", "verdad", "falsedad")):
		domain = "logic_math"
	elif any(kw in task.lower() for kw in ("hola", "buenos días", "buenos dias", "saludo")):
		domain = "general"
	else:
		# Fallback to semantic domain classification

		try:
			translator = get_translator()
			vector = translator.encode(task)
			hints = translator.decode(vector, top_k=1)
			raw_domain = hints[0][0].lower() if hints else "general"
		except Exception:
			raw_domain = "general"

		# Map raw concepts to rules domains
		if "código" in raw_domain or "sintaxis" in raw_domain:
			domain = "code_python"
		elif "lógica" in raw_domain or "matemática" in raw_domain:
			domain = "logic_math"
		elif "resumen" in raw_domain:
			domain = "synth"
		else:
			domain = "general"

	expert, _ = route_semantic(domain, len(task))
	return expert
