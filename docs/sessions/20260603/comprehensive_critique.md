# Análisis de Rigor Arquitectónico (Temperatura = 0)
## Diagnóstico del Ecosistema Neuro-Simbólico y Evaluación de Modificaciones

Este documento expone una evaluación fría, quirúrgica y honesta del estado actual del sistema FrankenSwarm y de las implicaciones dinámicas y matemáticas de las modificaciones propuestas.

---

## 1. El Estado del Sistema Actual: Diagnóstico de Fragilidad

El acoplamiento de **BitNet (red neuronal de peso ternario 1.58b)** con **Prolog (motor físico de reglas deterministas)** y **PPO (aprendizaje por refuerzo)** presenta fallos de diseño estructurales que comprometen la convergencia y la robustez del sistema.

```
       [ BitNet 1.58-bit ]  <--- (Representación latente no estacionaria)
               |
        (Acción Logits)
               |
               v
       [ Filtro / Máscara ]  <--- (Reglas estáticas de get_valid_actions_mask)
               |
               v
       [ Prolog Engine ]    <--- (Resolución discreta de transiciones)
               |
               v
      [ Feedback de PPO ]   <--- (Recompensa no convergente bajo escasez)
```

### A. El Abismo Neuro-Simbólico y la Máscara de Acción
El sistema delega la física y la lógica de consecuencias a un motor Prolog (`cooperative_rules.pl`). La red neuronal no "entiende" la física; simplemente es enmascarada para no elegir acciones prohibidas.
*   **La Máscara de Acción es un Bastón**: Al enmascarar rígidamente las acciones inválidas en `get_masked_probs`, impedimos que la red experimente el fracaso físico directo durante el día. La red solo aprende sobre un subespacio de probabilidad artificialmente podado.
*   **Inconsistencia de Gradiente**: Cuando las variables cambian y una acción se bloquea de golpe (por ejemplo, `comer` pasa de 1.0 a 0.0 porque el suelo llegó a 0.0), la distribución de probabilidad de la red sufre una discontinuidad abrupta. PPO asume un MDP (Proceso de Decisión de Markov) suave; estas discontinuidades causan oscilaciones salvajes en la ventaja estimada ($A_t$) y en el gradiente de la política.

### B. El Bucle de Recurrencia Temporal (Temporal Resonance)
La introducción de `h_prev` (memoria latente entre ticks) mitigó la amnesia de corto plazo, pero introdujo **inercia de error**:
*   Si un agente comete un error táctico o entra en pánico (gritos repetidos o movimientos erráticos), ese estado mental latente se retroalimenta al siguiente tick.
*   Sin un mecanismo de *olvido latente gating* (como en las LSTM o GRU), el estado oculto se satura rápidamente con ruido dinámico, atrapando a los agentes en bucles de retroalimentación de los cuales el modelo de resonancia de 3 capas no puede escapar fácilmente.

### C. La Ecología del Hambre (El Cuello de Botella de Scarcity)
El análisis de los logs demuestra que el entorno actual es un desierto ecológico:
1.  **Podredumbre instantánea**: La comida decae en 8 ticks a 0.0.
2.  **Cooldown severo**: El cooldown de 60 ticks significa que el mapa está vacío el 88.3% del tiempo.
3.  **Amnesia mental**: El mapa mental (`map_knowledge`) del agente se actualiza a 0.0 cuando visita un nodo vacío. Como el gradiente de recompensa por aproximación (PBRS) se apaga cuando el mapa reporta 0.0 comida, los agentes pierden todo incentivo para volver a buscar comida.
4.  **Cierre genético**: Los fundadores nacen con 2 habilidades y un límite rígido de 2 slots (`max_slots = 2`). Tienen 0 slots libres. No pueden aprender nada nuevo. Están condenados biológicamente a no tener fuego ni cocina.

---

## 2. Evaluación Crítica de las Nuevas Modificaciones

Las propuestas introducen mecánicas profundas de supervivencia, pero abren vectores de fallo específicos en la optimización del aprendizaje por refuerzo:

### A. La Trampa de Explotación de la Experiencia (Reward Farming)
Si intentarlo sin éxito otorga una recompensa positiva (`+0.2`) para guiar la exploración, y la red no puede realizar la acción de todos modos (porque carece de la habilidad):
*   **Vulnerabilidad**: La política aprende rápidamente que la forma más fácil de obtener recompensa acumulada constante es **quedarse quieto e intentar una acción prohibida repetidamente** (por ejemplo, intentar encender fuego sin la habilidad).
*   Si la penalización por fallar es `-0.3` y el bono por intentarlo es `+0.2`, el coste neto es `-0.1`. Si moverse cuesta `-0.2` (desgaste de energía) y el entorno no tiene comida, la red preferirá la inactividad intentando `"encender"` infinitamente antes que morir explorando.
*   **Solución Matemática**: El bono de aprendizaje debe ser estrictamente transitorio y decreciente:
    $$\text{Bono} = 0.2 \times \max\left(0, 1 - \frac{\text{Experiencia Acumulada}}{10.0}\right)$$
    Una vez que la experiencia de esa habilidad llega a 10.0 (o si los slots de habilidad están llenos), el bono debe ser exactamente **0.0**.

### B. No Estacionariedad por Modulación Emocional
El estado emocional ahora controlará la velocidad de aprendizaje (experiencia: `1.0x` en alegría, `0.2x` en ira, `0.0x` en tristeza) y potencialmente las tasas de éxito.
*   **Riesgo de Divergencia de PPO**: PPO asume que la dinámica del entorno ($P(s'|s,a)$) y las recompensas ($R(s,a)$) son estacionarias. Al hacer que el éxito de las acciones y la acumulación de experiencia varíen dinámicamente según la emoción interna (la cual depende de la homeostasis), el MDP se vuelve altamente no estacionario.
*   Un agente puede aprender una política óptima de supervivencia cuando está en `"alegría"`, pero esa misma política fallará catastróficamente al pasar a `"ira"` o `"tristeza"`, ya que el mapa de transición del mundo ha cambiado bajo sus pies. Esto puede impedir que la red converja, dejándola en un estado de caos conductual permanente.

### C. El Árbol Tecnológico (Tech Tree) como Restricción de Exploración
Forzar dependencias rígidas (ej. `"fuego"` requiere `"artesanía"`) restringe aún más el espacio de estados explorables.
*   Si el agente no tiene `"artesanía"`, la acción `"encender"` no le dará experiencia en `"fuego"` ni bonos.
*   Esto es correcto lógicamente, pero reduce la probabilidad de descubrir fuego por puro azar a casi cero. El agente debe descubrir obligatoriamente `"artesanía"` primero. Si la red no asocia que `"artesanía"` es el prerrequisito para el `"fuego"` (porque PPO no planifica a largo plazo con dependencias lógicas abstractas), el árbol tecnológico puede actuar como una barrera insalvable.
*   **Mitigación vía Dojo**: Es **mandatorio** que el Dojo pre-entrene intensamente estas secuencias de transición. El agente debe ser condicionado en el Dojo para aprender `"artesanía"` primero, luego `"fuego"`, y luego `"cocina"`. Si confiamos únicamente en la exploración libre de PPO, la probabilidad de que descubran la secuencia completa es matemáticamente despreciable.

---

## 3. Comparativa Ecológica de Parámetros

| Métrica / Parámetro | Configuración EXP_077 (Falla) | Configuración Modificada (Propuesta) | Justificación Técnica |
| :--- | :--- | :--- | :--- |
| **Decaimiento de Comida** | `0.5` unidades / tick | **`0.1`** unidades / tick | Permite que los recursos duren 40 ticks en el suelo en lugar de 8, ampliando la ventana ecológica útil. |
| **Hoguera en Cueva** | Temporal (apagado rápido) | **Permanente (`fire = 999999`)** | Actúa como un atractor espacial y base permanente para Nico, Sofy y Hugo. |
| **Slots Iniciales (`max_slots`)** | `2` slots (128 width) | **`3`** slots (inicia con 1 slot libre) | Rompe el bloqueo genético. Los fundadores pueden aprender sin depender de neurogénesis tardía. |
| **Homeostasis Default** | `"ira"` (umbral de confort >70) | **`"alegría"` o `"calma"` (umbral confort >45) | Evita que los agentes vivan en constante penalización de aprendizaje (0.2x). |
| **Caducidad de Memoria** | Infinita (amnesia persistente) | **Curiosidad activa (35 ticks)** | Resetea la estimación de comida vieja a `5.0` en el mapa cognitivo para forzar la re-exploración pasiva. |

---

## 4. Dictamen Final y Viabilidad

*   **Ajuste de Podredumbre y Cooldown**: Totalmente necesario. Mantener el entorno en el estado estéril de `EXP_077` era un suicidio de optimización. Un entorno sin gradiente de alimento es un entorno donde PPO solo aprende a morir rápido para minimizar la penalización por hambre a largo plazo.
*   **Hoguera Permanente en la Cueva**: Excelente decisión de diseño. Actúa como un *atractor espacial* (home base). Simplifica la geografía del mapa cognitivo y provee un punto de anclaje claro para la transición de estados.
*   **Descubrimiento Autónomo + Árbol Tecnológico**: Es viable **únicamente** si el Dojo actúa como el motor de transferencia (Behavioral Cloning de la secuencia del árbol) y si el PBRS (Potential-Based Reward Shaping) se incrementa temporalmente cuando el agente progresa en el árbol. De lo contrario, PPO se ahogará en la no estacionariedad emocional.

Nuestra recomendación es proceder con la implementación, pero respetando estrictamente las salvaguardas contra la explotación de recompensas (bono decreciente) y la no estacionariedad emocional (umbral de confort suavizado).
