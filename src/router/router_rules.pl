% Frankenswarm SWI-Prolog Routing Rules
:- dynamic query_property/2.

% ==========================================
% HECHOS DINÁMICOS & AUXILIARES
% ==========================================
prop(Key, Value) :- query_property(Key, Value).

% ==========================================
% ENRUTAMIENTO DE EXPERTO LÓGICO
% ==========================================
route_expert(code_node) :- prop(domain, code_python), !.
route_expert(reason_node) :- prop(domain, logic_math), !.
route_expert(synth_node) :- prop(domain, synth), !.
route_expert(default_node).

% ==========================================
% AFINIDAD DE HARDWARE (SILICIO)
% ==========================================
% Código corto (< 1000 caracteres) se enruta a la NPU (bajo consumo)
route_silicon(npu) :- prop(domain, code_python), prop(length, L), L < 1000, !.
% Código largo se enruta a CUDA (dGPU)
route_silicon(cuda) :- prop(domain, code_python), !.
% Razonamiento / Matemáticas se enruta a CUDA
route_silicon(cuda) :- prop(domain, logic_math), !.
% Tareas de síntesis de fondo se enrutan a la iGPU (Vulkan) si es latency=background
route_silicon(vulkan) :- prop(domain, synth), prop(latency, background), !.
% Fallback general a la CPU
route_silicon(cpu).

% ==========================================
% ENRUTAMIENTO ANTERIOR (RETROCOMPATIBILIDAD)
% ==========================================
contains(String, Substring) :- sub_string(String, _, _, _, Substring).

route(Input, code_node) :- contains(Input, "def "), !.
route(Input, code_node) :- contains(Input, "class "), !.
route(Input, code_node) :- contains(Input, "import "), !.
route(Input, code_node) :- contains(Input, "return "), !.
route(Input, code_node) :- contains(Input, "```python"), !.
route(Input, code_node) :- contains(Input, "SELECT "), !.
route(Input, code_node) :- contains(Input, "INSERT "), !.
route(Input, code_node) :- contains(Input, "UPDATE "), !.
route(Input, code_node) :- contains(Input, "CREATE TABLE"), !.

route(Input, reason_node) :- sub_string(Input, 0, _, _, "why "), !.
route(Input, reason_node) :- sub_string(Input, 0, _, _, "explain "), !.
route(Input, reason_node) :- sub_string(Input, 0, _, _, "what is "), !.
route(Input, reason_node) :- sub_string(Input, 0, _, _, "how does "), !.
route(Input, reason_node) :- contains(Input, "because"), !.
route(Input, reason_node) :- contains(Input, "therefore"), !.
route(Input, reason_node) :- contains(Input, "hypothesis"), !.
route(Input, reason_node) :- contains(Input, "analyze"), !.

route(Input, synth_node) :- contains(Input, "summarize"), !.
route(Input, synth_node) :- contains(Input, "synthesize"), !.
route(Input, synth_node) :- contains(Input, "combine"), !.
route(Input, synth_node) :- contains(Input, "merge"), !.

route(_, default_node).

