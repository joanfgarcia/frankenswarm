% Frankenswarm — Base de Conocimiento de Supervivencia
:- dynamic query_property/2.
:- dynamic venenoso/1.
:- dynamic seguro/1.

% hechos de toxicidad
venenoso(hongo_rojo).
venenoso(baya_azul).
venenoso(agua_turbia).

% hechos de seguridad explícita
seguro(hongo_marron).
seguro(baya_roja).
seguro(agua_limpia).

% una cueva es segura si no hay depredador
segura_cueva(cueva) :- \+ query_property(peligro, depredador).

% regla general de consumición segura
comestible(X) :- seguro(X), \+ venenoso(X).
comestible(X) :- \+ venenoso(X).
