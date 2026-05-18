"""
Frankenswarm — SWI-Prolog Sovereign Router Prototype
====================================================
Declarative rule-based router using SWI-Prolog via pyswip.
This is a prototype replacing the Python match/case router.
"""

from enum import StrEnum
import os

class NodeTarget(StrEnum):
    CODE    = "code_node"
    REASON  = "reason_node"
    SYNTH   = "synth_node"
    DEFAULT = "default_node"

def route(task: str) -> NodeTarget:
    """
    Evaluates the Prolog rules to determine the expert node.
    """
    try:
        from pyswip import Prolog
    except ImportError:
        # Fallback for prototype if pyswip is not installed
        print("[WARNING] pyswip not installed. Fallback to DEFAULT.")
        return NodeTarget.DEFAULT

    prolog = Prolog()
    rules_path = os.path.join(os.path.dirname(__file__), "router_rules.pl")
    prolog.consult(rules_path)
    
    # Escape quotes in task for safe prolog query
    safe_task = task.replace("'", "\\'")
    query = f"route('{safe_task}', Node)"
    
    results = list(prolog.query(query))
    if results:
        target_str = results[0]["Node"]
        try:
            return NodeTarget(target_str)
        except ValueError:
            pass
            
    return NodeTarget.DEFAULT
