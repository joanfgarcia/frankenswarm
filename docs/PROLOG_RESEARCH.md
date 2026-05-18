# Prolog & LLM Integration (Research Notes)

**Reference Link:** [Quantum Prolog: Using LLMs to generate Prolog planners](https://quantumprolog.sgml.net/llm-demo/part1.html)

## 1. El Problema de los LLMs en Planificación
El artículo confirma empíricamente que los LLMs (incluso modelos de 400B de parámetros) son inherentemente defectuosos para tareas de optimización combinatoria y búsqueda de planes. Alucinan, inventan reglas y no pueden mantener un árbol de búsqueda profundo y riguroso de forma autónoma.
- **Conclusión:** Los LLMs deben usarse como **Traductores**, no como **Solucionadores**.

## 2. La Arquitectura Óptima (Bridge Pattern)
La solución que propone el artículo coincide al 100% con nuestra arquitectura en Frankenswarm (`PrologExpert` + `Subprocess` a `swipl`):
- **Capa LLM:** Recibe lenguaje natural y lo traduce EXCLUSIVAMENTE a hechos (estado inicial) y acciones atómicas (reglas).
- **Capa Nativa:** Un motor de Prolog (como SWI-Prolog) recibe ese código y ejecuta la fuerza bruta del *backtracking* para encontrar la solución lógica.

## 3. Ingeniería de Prompts (Directivas Críticas)
Para evitar que el LLM rompa el código o intente resolver el problema por sí mismo, el artículo recomienda las siguientes directivas estrictas para el *System Prompt* del experto en Prolog:

1. **Sin bucles de resolución:** Prohibir explícitamente al LLM escribir el predicado principal de resolución (ej. `main` o bucles iterativos de búsqueda).
2. **Uso de mutaciones:** Obligar a que las acciones modifiquen el estado usando exclusivamente `assertz/1`, `asserta/1` y `retract/1`.
3. **Restricciones de sintaxis:**
   - Prohibir `if-then` o `if-then-else` (complican el AST para el evaluador).
   - Prohibir importaciones de librerías de terceros o Constraint Logic Programming (CLP) para mantener compatibilidad ISO Prolog estricta.
   - Prohibir el uso de predicados folclóricos (ej. `append/3`, `subtract/3`) a menos que se definan explícitamente.

## 4. El Motor Genérico de Planificación
Para que Prolog evalúe las reglas generadas por el LLM, nosotros (el orquestador Python) debemos inyectar en el código final un **motor de evaluación genérico** (una plantilla fija de Prolog) que se encarga de iterar sobre los `action_pred` (generados por la IA) hasta alcanzar un `check_pred` (estado deseado generado por la IA). 

### Esqueleto a Inyectar por Python (Plantilla Base)
```prolog
% Plantilla inyectada automáticamente por PrologExpert
% ... (Las reglas de acción y estado del LLM van aquí arriba) ...

:- initialization(main, main).
% Generador recursivo de planificación genérico
% (El LLM no escribe esto, lo escribimos nosotros como infraestructura)
```

## Impacto en Frankenswarm
Esta lectura valida el diseño del Router y el `PrologExpert`. Confirma que delegar la inferencia semántica al LLM y el cómputo lógico al binario nativo en C es el "State of the Art" actual en agentes lógicos-simbólicos neurosimbólicos.
