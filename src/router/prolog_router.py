"""
Frankenswarm — Sovereign Router
================================
Declarative rule-based router implemented in Python (match/case).
Structurally mirrors Prolog's pattern matching and inference rules.

Rules are data, not code. Adding a new rule = adding one line to ROUTING_RULES.

# TODO: If rule complexity grows beyond ~50 rules, replace this module with:
#       SWI-Prolog via `pyswip` (pip install pyswip).
#       The interface contract (route(task) -> NodeTarget) stays identical.
#       Example migration:
#           from pyswip import Prolog
#           prolog = Prolog()
#           prolog.consult("rules/router.pl")
#           result = list(prolog.query(f"route('{task}', Node)"))[0]["Node"]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class NodeTarget(StrEnum):
	"""Available expert nodes in the swarm."""

	CODE = "code_node"
	REASON = "reason_node"
	SYNTH = "synth_node"
	DEFAULT = "default_node"


@dataclass(frozen=True)
class RoutingRule:
	"""
	A single Prolog-style clause.

	Prolog equivalent:
		route(Input, code_node) :- contains(Input, "def ").
	Python equivalent:
		RoutingRule(predicate=lambda t: "def " in t, target=NodeTarget.CODE)
	"""

	predicate: Callable[[str], bool]
	target: NodeTarget
	description: str  # Human-readable, like a Prolog comment


# ---------------------------------------------------------------------------
# ROUTING RULES — Edit here to add/change routing logic.
# Order matters: first match wins (Prolog-style determinism).
# ---------------------------------------------------------------------------
ROUTING_RULES: list[RoutingRule] = [
	# --- Code detection ---
	RoutingRule(
		predicate=lambda t: any(kw in t for kw in ("def ", "class ", "import ", "return ", "```python")),
		target=NodeTarget.CODE,
		description="Python code patterns → code_node",
	),
	RoutingRule(
		predicate=lambda t: any(kw in t for kw in ("SELECT ", "INSERT ", "UPDATE ", "CREATE TABLE")),
		target=NodeTarget.CODE,
		description="SQL patterns → code_node",
	),
	# --- Reasoning / explanation ---
	RoutingRule(
		predicate=lambda t: t.lower().startswith(("why ", "explain ", "what is ", "how does ")),
		target=NodeTarget.REASON,
		description="Interrogative reasoning prompts → reason_node",
	),
	RoutingRule(
		predicate=lambda t: any(kw in t.lower() for kw in ("because", "therefore", "hypothesis", "analyze")),
		target=NodeTarget.REASON,
		description="Causal / analytical language → reason_node",
	),
	# --- Synthesis ---
	RoutingRule(
		predicate=lambda t: any(kw in t.lower() for kw in ("summarize", "synthesize", "combine", "merge")),
		target=NodeTarget.SYNTH,
		description="Aggregation / synthesis intent → synth_node",
	),
]


def route(task: str) -> NodeTarget:
	"""
	Deterministic router — O(n_rules), zero tokens, zero GPU.

	Prolog mental model:
		route(Task, Node) :- rule1(Task), !, Node = code_node.
		route(Task, Node) :- rule2(Task), !, Node = reason_node.
		route(_, default_node).

	Args:
		task: Raw input string to classify.

	Returns:
		NodeTarget identifying which expert node should handle the task.
	"""
	for rule in ROUTING_RULES:
		if rule.predicate(task):
			return rule.target
	return NodeTarget.DEFAULT


def explain_route(task: str) -> tuple[NodeTarget, str]:
	"""
	Same as route() but returns the matching rule description for traceability.
	Useful for debugging and for understanding why a task was routed somewhere.
	"""
	for rule in ROUTING_RULES:
		if rule.predicate(task):
			return rule.target, rule.description
	return NodeTarget.DEFAULT, "No rule matched — fallback to default_node"
