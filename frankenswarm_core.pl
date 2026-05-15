% ==========================================
% HECHOS DINÁMICOS (El estado del enjambre)
% ==========================================
% experto(ID, CapacidadNeurons, FitnessActual, Estado)
:- dynamic experto/4.
% linaje(ID_Hijo, ID_Padre).
:- dynamic linaje/2.
% especialidad(ID, Tarea, ScoreConfianza).
:- dynamic especialidad/3.

% --- Población inicial de prueba ---
experto(exp_001, 128, 0.82, congelado).
experto(exp_002, 128, 0.45, mutando).
experto(exp_003, 256, 0.91, congelado). % Ya pasó por Net2Net antes

especialidad(exp_001, programacion_python, 0.88).
especialidad(exp_003, razonamiento_matematico, 0.95).

% ==========================================
% REGLAS DEL CICLO DE VIDA (The Lifecycle)
% ==========================================

% 1. Evaluar Acción basada en Fitness
% Si el fitness es bajo, se mantiene en mutación NEAT.
evaluar_estatus(ID, neat_mutacion) :-
    experto(ID, _, Fitness, mutando),
    Fitness < 0.80.

% Si supera el umbral, se congela para producción.
evaluar_estatus(ID, congelar) :-
    experto(ID, _, Fitness, mutando),
    Fitness >= 0.80.

% 2. Disparador Net2Net (Crecimiento de Capacidad)
% Si un experto congelado estanca su rendimiento, requiere inyección a cero.
requiere_net2net(ID) :-
    experto(ID, Capacidad, Fitness, congelado),
    Fitness < 0.85, % Plateau o degradación ligera ante nuevas tareas
    Capacidad < 1024. % Límite físico impuesto para los 8GB VRAM

% ==========================================
% EL ENRUTADOR SPARSE MoE (The Router)
% ==========================================
% Selecciona el mejor experto para una tarea basándose en su especialidad,
% pero solo si está congelado (listo para producción) y cabe en memoria.

enrutar_tarea(Tarea, ID_Experto_Elegido) :-
    especialidad(ID_Experto_Elegido, Tarea, Confianza),
    experto(ID_Experto_Elegido, Capacidad, _, congelado),
    % Buscamos si hay otro experto mejor que cumpla las condiciones (Maximizando Confianza)
    \+ (
        especialidad(Otro_ID, Tarea, Otra_Confianza),
        experto(Otro_ID, _, _, congelado),
        Otra_Confianza > Confianza
    ).
