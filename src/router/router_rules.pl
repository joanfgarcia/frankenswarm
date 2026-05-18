% Frankenswarm SWI-Prolog Routing Rules
% Order matters: first match wins.

% --- Helper for substring matching ---
contains(String, Substring) :- sub_string(String, _, _, _, Substring).

% --- Code detection ---
route(Input, code_node) :- contains(Input, "def "), !.
route(Input, code_node) :- contains(Input, "class "), !.
route(Input, code_node) :- contains(Input, "import "), !.
route(Input, code_node) :- contains(Input, "return "), !.
route(Input, code_node) :- contains(Input, "```python"), !.
route(Input, code_node) :- contains(Input, "SELECT "), !.
route(Input, code_node) :- contains(Input, "INSERT "), !.
route(Input, code_node) :- contains(Input, "UPDATE "), !.
route(Input, code_node) :- contains(Input, "CREATE TABLE"), !.

% --- Reasoning / explanation ---
route(Input, reason_node) :- sub_string(Input, 0, _, _, "why "), !.
route(Input, reason_node) :- sub_string(Input, 0, _, _, "explain "), !.
route(Input, reason_node) :- sub_string(Input, 0, _, _, "what is "), !.
route(Input, reason_node) :- sub_string(Input, 0, _, _, "how does "), !.

route(Input, reason_node) :- contains(Input, "because"), !.
route(Input, reason_node) :- contains(Input, "therefore"), !.
route(Input, reason_node) :- contains(Input, "hypothesis"), !.
route(Input, reason_node) :- contains(Input, "analyze"), !.

% --- Synthesis ---
route(Input, synth_node) :- contains(Input, "summarize"), !.
route(Input, synth_node) :- contains(Input, "synthesize"), !.
route(Input, synth_node) :- contains(Input, "combine"), !.
route(Input, synth_node) :- contains(Input, "merge"), !.

% --- Default Fallback ---
route(_, default_node).
