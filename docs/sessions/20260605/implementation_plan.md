# Plan de Implementación: Descubrimiento de Habilidades y Ecología de Conservación (Prolog-Driven)

Este plan describe las modificaciones de diseño e infraestructura para permitir que los agentes Nico, Sofy y Hugo sobrevivan y descubran habilidades de forma autónoma delegando el Árbol Tecnológico y la evaluación de acciones al motor Prolog.

---

## 1. Goal Description
El objetivo es implementar un sistema de descubrimiento de habilidades dinámico y modular. En lugar de codificar las restricciones y el aprendizaje en Python, utilizaremos **SWI-Prolog** para:
1. Validar el **Árbol Tecnológico** (Tech Tree).
2. Evaluar las **acciones de los agentes** clasificándolas en: éxito físico, intento válido de aprendizaje (exploración correcta), error físico (falta de recursos) o error de prerrequisitos tecnológicos.
3. Evaluar qué habilidades son elegibles para ser aprendidas mediante **observación** en un lugar determinado.

---

## 2. User Review Required

> [!IMPORTANT]
> **Modulación de Recompensas por Prolog**: 
> La recompensa de PPO en `get_reward` se adaptará al resultado de Prolog:
> - `exito`: Recompensa estándar de la acción.
> - `intento_valido`: Bono de exploración de `+0.2` (que decae a `0.0` a medida que la experiencia de la habilidad se acerca a `10.0` o si los slots de habilidad están llenos) para evitar *reward farming*.
> - `error_requisito`: Penalización de `-0.5` para disuadir a la red de intentar tecnologías para las que no tiene bases.
> - `error_fisico`: Penalización de `-0.3` (estándar para fallos físicos).

---

## 3. Proposed Changes

### Componente 1: Reglas de Prolog (`cooperative_rules.pl`)

#### [MODIFY] [cooperative_rules.pl](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/cooperative_rules.pl)
Añadiremos las siguientes reglas y estructuras:

1. **Definición del Árbol Tecnológico**:
   ```prolog
   prerrequisito('comida', []).
   prerrequisito('agua', []).
   prerrequisito('artesanía', []).
   prerrequisito('caza', ['artesanía']).
   prerrequisito('pesca', ['artesanía']).
   prerrequisito('construcción', ['artesanía']).
   prerrequisito('fuego', ['artesanía']).
   prerrequisito('cocina', ['comida', 'agua', 'fuego']).
   ```

2. **Verificación de Prerrequisitos**:
   ```prolog
   cumple_prereqs([], _) :- !.
   cumple_prereqs([H|T], Skills) :-
       member(H, Skills),
       cumple_prereqs(T, Skills).

   prerrequisitos_satisfechos(Habilidad, AgentSkills) :-
       prerrequisito(Habilidad, Prereqs),
       cumple_prereqs(Prereqs, AgentSkills).
   ```

3. **Consulta de Observación (`que_observar/14`)**:
   Determina qué habilidades puede observar/descubrir un agente en su localización actual:
   ```prolog
   que_observar(Skills, _, FireActive, _, _, _, _, _, _, _, _, _, _, 'fuego') :-
       \+ member('fuego', Skills),
       prerrequisitos_satisfechos('fuego', Skills),
       FireActive =:= 1.

   que_observar(Skills, _, _, GroundFood, FoodEligible, _, _, _, _, _, _, _, _, 'comida') :-
       \+ member('comida', Skills),
       prerrequisitos_satisfechos('comida', Skills),
       GroundFood >= 1.0,
       FoodEligible =:= 1.

   que_observar(Skills, _, _, GroundFood, _, FishEligible, _, _, _, _, _, _, _, 'pesca') :-
       \+ member('pesca', Skills),
       prerrequisitos_satisfechos('pesca', Skills),
       GroundFood >= 1.0,
       FishEligible =:= 1.

   que_observar(Skills, _, _, _, _, _, GroundWater, WaterAvailable, _, _, _, _, _, 'agua') :-
       \+ member('agua', Skills),
       prerrequisitos_satisfechos('agua', Skills),
       GroundWater >= 1.0,
       WaterAvailable =:= 1.

   que_observar(Skills, Location, _, _, _, _, _, _, PreyLocation, _, _, _, _, 'caza') :-
       \+ member('caza', Skills),
       prerrequisitos_satisfechos('caza', Skills),
       Location = PreyLocation.

   que_observar(Skills, _, _, _, _, _, _, _, _, GroundBranches, BranchesEligible, GroundStones, StonesEligible, 'artesanía') :-
       \+ member('artesanía', Skills),
       prerrequisitos_satisfechos('artesanía', Skills),
       GroundBranches >= 1.0, BranchesEligible =:= 1,
       GroundStones >= 1.0, StonesEligible =:= 1.

   que_observar(Skills, _, _, _, _, _, _, _, _, GroundBranches, BranchesEligible, GroundStones, StonesEligible, 'construcción') :-
       \+ member('construcción', Skills),
       prerrequisitos_satisfechos('construcción', Skills),
       GroundBranches >= 1.0, BranchesEligible =:= 1,
       GroundStones >= 1.0, StonesEligible =:= 1.
   ```

4. **Evaluación de Decisiones (`evaluar_intento/16`)**:
   Clasifica los intentos de acción en: `exito`, `intento_valido` (aprendizaje válido), `error_requisito` (tecnológico) o `error_fisico` (recursos):
   ```prolog
   % evaluar_intento(Action, AgentSkills, MochilaComida, MochilaAgua, MochilaRamas, MochilaPiedras, TieneLanza, Location, GroundFood, GroundWater, GroundBranches, GroundStones, FireActive, PreyLocation, ResultType, Event)
   
   evaluar_intento('comer', Skills, MochilaComida, MochilaAgua, _, _, _, _, GroundFood, _, _, _, FireActive, _, ResultType, Event) :-
       ( (MochilaComida > 0, MochilaAgua > 0, member('cocina', Skills), FireActive =:= 1) ->
           ResultType = 'exito', Event = 'cocina guiso caliente exitosamente'
       ; (MochilaComida > 0) ->
           ResultType = 'exito', Event = 'come de la mochila exitosamente'
       ; (GroundFood >= 1.0, (member('comida', Skills) ; member('pesca', Skills))) ->
           ResultType = 'exito', Event = 'come del suelo exitosamente'
       ; (GroundFood >= 1.0, \+ member('comida', Skills), \+ member('pesca', Skills)) ->
           ( prerrequisitos_satisfechos('comida', Skills) ->
               ResultType = 'intento_valido', Event = 'intenta comer/recolectar comida para aprender'
           ;
               ResultType = 'error_requisito', Event = 'intenta comer sin prerrequisito'
           )
       ;
           ResultType = 'error_fisico', Event = 'intenta comer de suelo agotado'
       ),
       !.

   evaluar_intento('beber', Skills, _, MochilaAgua, _, _, _, _, _, GroundWater, _, _, _, _, ResultType, Event) :-
       ( (MochilaAgua > 0) ->
           ResultType = 'exito', Event = 'bebe de la mochila exitosamente'
       ; (GroundWater >= 1.0, member('agua', Skills)) ->
           ResultType = 'exito', Event = 'bebe del suelo exitosamente'
       ; (GroundWater >= 1.0, \+ member('agua', Skills)) ->
           ( prerrequisitos_satisfechos('agua', Skills) ->
               ResultType = 'intento_valido', Event = 'intenta beber para aprender'
           ;
               ResultType = 'error_requisito', Event = 'intenta beber sin prerrequisito'
           )
       ;
           ResultType = 'error_fisico', Event = 'intenta beber de zona seca'
       ),
       !.

   evaluar_intento('fabricar', Skills, _, _, MochilaRamas, MochilaPiedras, TieneLanza, _, _, _, _, _, _, _, ResultType, Event) :-
       ( (\+ member('artesanía', Skills)) ->
           ( prerrequisitos_satisfechos('artesanía', Skills) ->
               ( (MochilaRamas >= 1, MochilaPiedras >= 1) ->
                   ResultType = 'intento_valido', Event = 'intenta fabricar para aprender artesania'
               ;
                   ResultType = 'error_fisico', Event = 'intenta fabricar pero no tiene materiales'
               )
           ;
               ResultType = 'error_requisito', Event = 'intenta fabricar sin prerrequisito'
           )
       ; (TieneLanza =:= 1) ->
           ResultType = 'error_fisico', Event = 'ya tiene una lanza'
       ; (MochilaRamas < 1 ; MochilaPiedras < 1) ->
           ResultType = 'error_fisico', Event = 'faltan materiales para fabricar'
       ;
           ResultType = 'exito', Event = 'fabrica una lanza exitosamente'
       ),
       !.

   evaluar_intento('construir', Skills, _, _, MochilaRamas, MochilaPiedras, _, _, _, _, _, _, _, _, ResultType, Event) :-
       ( (\+ member('construcción', Skills)) ->
           ( prerrequisitos_satisfechos('construcción', Skills) ->
               ( (MochilaRamas >= 2, MochilaPiedras >= 1) ->
                   ResultType = 'intento_valido', Event = 'intenta construir para aprender construccion'
               ;
                   ResultType = 'error_fisico', Event = 'intenta construir pero no tiene materiales'
               )
           ;
               ResultType = 'error_requisito', Event = 'intenta construir sin prerrequisito'
           )
       ; (MochilaRamas < 2 ; MochilaPiedras < 1) ->
           ResultType = 'error_fisico', Event = 'faltan materiales para construir'
       ;
           ResultType = 'exito', Event = 'construye un refugio exitosamente'
       ),
       !.

   evaluar_intento('encender', Skills, _, _, MochilaRamas, _, _, _, _, _, _, _, FireActive, _, ResultType, Event) :-
       ( (\+ member('fuego', Skills)) ->
           ( prerrequisitos_satisfechos('fuego', Skills) ->
               ( (MochilaRamas >= 2) ->
                   ResultType = 'intento_valido', Event = 'intenta encender para aprender fuego'
               ;
                   ResultType = 'error_fisico', Event = 'intenta encender pero no tiene ramas'
               )
           ;
               ResultType = 'error_requisito', Event = 'intenta encender sin prerrequisito'
           )
       ; (FireActive =:= 1) ->
           ResultType = 'error_fisico', Event = 'ya hay una hoguera activa'
       ; (MochilaRamas < 2) ->
           ResultType = 'error_fisico', Event = 'faltan ramas para encender'
       ;
           ResultType = 'exito', Event = 'enciende una hoguera exitosamente'
       ),
       !.

   evaluar_intento(_, _, _, _, _, _, _, _, _, _, _, _, _, _, 'basal', 'acción basal o comunicativa').
   ```

---

## 4. Proposed Changes

### Componente 2: Integración en Python (`cooperative_world.py`)

#### [MODIFY] [cooperative_world.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/cooperative_world.py)

1. **Estado del Agente (`CoopAgentState`)**:
   - Inicializar `skill_experience: dict = field(default_factory=dict)` en el constructor de `CoopAgentState` para almacenar la experiencia acumulada por cada habilidad.

2. **Permisividad de Máscaras (`get_valid_actions_mask`)**:
   - Permitir al agente elegir acciones aunque no posea la habilidad, siempre y cuando cumpla con los requisitos de recursos físicos (ej. ramas para encender, comida para comer, materiales para fabricar). Esto permite que el agente "intente" las acciones y Prolog las evalúe.

3. **Ejecución y Aprendizaje en `act()`**:
   - Al inicio de la resolución de acciones susceptibles de aprendizaje (`comer`, `beber`, `fabricar`, `construir`, `encender`), hacer la consulta Prolog:
     ```python
     q = f"evaluar_intento('{action}', {format_skills(agent.learned_skills)}, {agent.mochila_comida}, {agent.mochila_agua}, {agent.mochila_ramas}, {agent.mochila_piedras}, {1 if agent.tiene_lanza else 0}, '{agent.location}', {caps['food']}, {caps['water']}, {caps['branches']}, {caps['stones']}, {1 if self.fire_locations.get(agent.location, 0) > 0 else 0}, '{self.prey_location}', ResultType, Event)"
     res = query_prolog(q)[0]
     result_type = res["ResultType"]
     ```
   - Si `result_type == 'intento_valido'`:
     - La acción física no progresa (`success = False`).
     - Modulamos la ganancia de experiencia según la emoción actual:
       - `alegría` $\rightarrow$ `+1.0` de experiencia.
       - `ira` / `miedo` / `hambre` $\rightarrow$ `+0.2` de experiencia.
       - `tristeza` / `dolor` $\rightarrow$ `+0.0` de experiencia.
     - Sumamos al acumulado: `agent.skill_experience[skill] = agent.skill_experience.get(skill, 0.0) + exp_gain`.
     - Si `agent.skill_experience[skill] >= 10.0` y hay slots libres, el agente descubre la habilidad permanentemente.
     - Marcamos `result["tried_skill"] = skill` para asignar el bono en `get_reward`.
   - Si `result_type == 'error_requisito'`:
     - La acción física falla.
     - Marcamos `result["tech_error"] = True` para penalizar.
   - Si `result_type == 'error_fisico'`:
     - La acción física falla.
     - Marcamos `result["physical_error"] = True` para penalizar.
   - Si `result_type == 'exito'`:
     - Procede con la lógica física normal (e.g. alimentar, fabricar lanza, etc.).

4. **Observación en `"ver"`**:
   - Si la acción elegida es `"ver"`, consultamos a Prolog qué se puede observar en la localización:
     ```python
     q_obs = f"que_observar({format_skills(agent.learned_skills)}, '{agent.location}', {1 if self.fire_locations.get(agent.location, 0) > 0 else 0}, {caps['food']}, {1 if loc_data['food_eligible'] else 0}, {1 if loc_data.get('fish_eligible') else 0}, {caps['water']}, {1 if loc_data['water_available'] else 0}, '{self.prey_location}', {caps['branches']}, {1 if loc_data.get('branches_eligible') else 0}, {caps['stones']}, {1 if loc_data.get('stones_eligible') else 0}, Habilidad)"
     res_obs = query_prolog(q_obs)
     ```
   - Para cada `Habilidad` encontrada en las respuestas de Prolog, acumulamos experiencia de observación modulada por la emoción (por ejemplo, `+1.0 * emotion_multiplier` por tick).
   - El agente puede aprender una habilidad observando la hoguera permanente en la cueva, o viendo peces en el río, etc.

5. **Modulación de Recompensas (`get_reward`)**:
   - Modificar la función de recompensa en función del tipo de intento registrado en `result`:
     - Si `result.get("tried_skill")` (intento de aprendizaje válido):
       - Bono temporal de `+0.2 * max(0.0, 1.0 - exp_acumulada / 10.0)` para guiar la exploración sin incentivar el farming infinito.
     - Si `result.get("tech_error")`:
       - Penalización de `-0.5` para evitar el spam de tecnologías avanzadas sin bases.
     - Si `result.get("physical_error")`:
       - Penalización de `-0.3`.

---

### Componente 3: Dojo de Entrenamiento (`dojo_populora.py`)

#### [MODIFY] [dojo_populora.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/dojo_populora.py)
Añadiremos los escenarios:
1. `observar_hoguera_cueva`: Si el agente está en la cueva con hoguera activa, no tiene `"fuego"` y su emoción es `"alegría"`, guiarlo a elegir `"ver"` para entrenar la observación del fuego.
2. `intentar_fuego`: Si el agente tiene ramas, tiene `"artesanía"` pero no `"fuego"`, guiarlo a elegir `"encender"` para entrenar el intento válido.
3. `error_tecnologico_fuego`: Si el agente tiene ramas pero no tiene `"artesanía"` ni `"fuego"`, penalizar su acción de `"encender"` guiándolo a `"fabricar"` o `"ver"`.

---

## 5. Verification Plan

### Pruebas Unitarias Automatizadas (`tests/test_prolog_discovery.py`)
Crearemos un nuevo test suite en `tests/test_prolog_discovery.py` para verificar:
1. Que Prolog evalúa correctamente los prerrequisitos del Tech Tree (e.g. no se puede encender fuego sin artesanía).
2. Que Prolog clasifica correctamente las acciones en `exito`, `intento_valido`, `error_requisito` y `error_fisico`.
3. Que la observación en la cueva con fuego activo incrementa la experiencia en la habilidad `"fuego"`.
4. Que al alcanzar 10.0 de experiencia en un slot libre, el agente descubre la habilidad permanentemente.
5. Que el bono de exploración para intentos válidos decae correctamente y no causa bucles de explotación infinita.
