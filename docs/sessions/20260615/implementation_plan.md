# Plan de Implementación: Currículo Piagetiano Progresivo Completo y Neurogénesis por Etapas (Bit 0-4 años)

Para validar la tesis de soberanía cognitiva "de verdad", reestructuraremos la etapa preescolar utilizando los **corpus completos sin submuestreo** de CHILDES (45,857 oraciones únicas) y de Física NSM (291 plantillas). Clasificaremos dinámicamente todo el volumen de datos en 4 etapas del desarrollo cognitivo basadas en la longitud media de emisión (MLU - Mean Length of Utterance). Cada frontera de edad desencadenará una **neurogénesis en caliente** (crecimiento del modelo y clonación de momentos de optimización).

Además, reestructuraremos los prompts de la Profesora Samantha para realizar una **evaluación semántica y cognitiva real**, en lugar de una comparación mecánica de cadenas de texto exactas.

---

## User Review Required

> [!IMPORTANT]
> **Evaluación Semántica de Samantha**:
> - El `system_prompt` de Samantha se reescribirá para que actúe como una psicóloga del desarrollo infantil. 
> - En lugar de exigir coincidencias exactas de palabras (como comparar 'mucho' con 'mucho'), Samantha evaluará la **coherencia semántica y lógica** de la respuesta de Bit según su edad cognitiva, admitiendo sinónimos, variaciones gramaticales y respuestas lógicas equivalentes (ej. responder 'rápido' o 'alegre' ante una pregunta sobre correr o jugar).
>
> **Prompts de Generación y Validación Apropiados**:
> - Ajustaremos los prompts de evaluación (`AGE_QUESTIONS`) para que sean estímulos naturales de habla dirigida a niños (ej. en lugar de 'tú: cuento uno dos', usaremos 'tú: cuenta conmigo uno dos').
> - Aseguraremos que el vocabulario de las preguntas esté estrictamente contenido en el diccionario base de Bit (3005 palabras) para evitar tokens `<unk>`.

---

## Proposed Changes

### Componente 1: Bucle de Entrenamiento Piaget (`src/bitnet/`)

#### [MODIFY] [train_sovereign_school.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/train_sovereign_school.py)
1. **Algoritmo de Clasificación por MLU**:
   - Implementar `partition_corpus_by_mlu(sequences: list[list[int]])` para clasificar sin descarte el corpus completo:
     - `stage_0_1` (longitud $\le 2$): Primeros balbuceos e imperativos.
     - `stage_1_2` (longitud $= 3$): Dos y tres palabras.
     - `stage_2_3` (longitud en $[4, 5]$): Sintaxis básica y emociones.
     - `stage_3_4` (longitud en $[6, 8]$): Explicaciones causales y condicionales.
2. **Bucle Secuencial por Etapas**:
   - Reestructurar el entrenamiento escolar para iterar por cada sub-etapa de forma progresiva:
     - **0-1 Año**: Datos `stage_0_1`. Duración: 8 épocas. Inicializa a `hidden_dim = 128`.
     - **1-2 Años**: Datos `stage_1_2` + 20% de repaso de `stage_0_1`. Duración: 8 épocas. Crecimiento a `256` dim.
     - **2-3 Años**: Datos `stage_2_3` + 20% de repaso de etapas anteriores. Duración: 8 épocas. Crecimiento a `384` dim.
     - **3-4 Años**: Datos `stage_3_4` + 20% de repaso de etapas anteriores. Duración: 8 épocas. Crecimiento a `512` dim.
3. **Mecanismo de Neurogénesis y Clonación**:
   - En las fronteras de etapa (épocas 8, 16 y 24), pausar y llamar a `net2wider_model` pasando `old_optimizer` y `new_optimizer` para duplicar y reescalar pesos y momentos $m_t$ y $v_t$.
4. **Conjunto de Validación (10%)**:
   - Cada sub-etapa de entrenamiento separará un 10% de sus secuencias asignadas para calcular y reportar `val_loss` en cada época de forma determinista.
5. **Evaluación de Samantha**:
   - Solo se evaluará el examen acumulativo final del hito de 4 años en Samantha al finalizar la sub-fase 3-4 (época 32).
6. **Muestreo Cualitativo Semilla**:
   - Al final de cada época, generar 5 tokens a partir de 3 prompts semilla (`yo sentir`, `fuego estar`, `madre decir`) e imprimirlos en consola para auditar visualmente el progreso de la sintaxis.

---

### Componente 2: Evaluación Cognitiva (`scripts/`)

#### [MODIFY] [evaluate_samantha_age.py](file:///home/joan/Documents/IA/frankenswarm/scripts/evaluate_samantha_age.py)
1. **Rediseño del Evaluador de Samantha**:
   - Modificar `system_prompt` para guiar a Samantha a evaluar semánticamente de forma coherente para cada edad:
```python
system_prompt = (
	"Eres la Profesora Samantha, una experta en psicología infantil y lingüística. "
	"Estás evaluando el habla y desarrollo cognitivo de un niño de 4 a 8 años. "
	"Tu tarea es analizar la respuesta del alumno para cada pregunta y determinar si demuestra "
	"comprensión semántica, lógica física básica y coherencia sintáctica para su edad. "
	"No exijas una coincidencia de palabras exacta; valora positivamente sinónimos, expresiones semánticamente "
	"equivalentes y respuestas con sentido lógico (por ejemplo, si se espera 'mucho' ante 'el perro corre', "
	"respuestas como 'rápido', 'feliz' o 'fuera' son válidas). "
	"Califica cada respuesta de 0 a 10 y detalla tu motivo. "
	"Debes responder ÚNICAMENTE con un objeto JSON válido con este formato:\n"
	"{\n"
	'  "calificaciones": [\n'
	'    {"pregunta": "...", "respuesta": "...", "esperada": "...", "calificacion": 10, "motivo": "..."}\n'
	"  ],\n"
	'  "puntuacion_media": 10.0,\n'
	'  "hito_superado": true\n'
	"}\n"
	"El hito se considera superado si la puntuacion_media es igual o superior a 8.0."
)
```
2. **Prompts e Hitos de Examen Naturales**:
   - Actualizar las preguntas en `AGE_QUESTIONS` para que utilicen lenguaje infantil natural (ej: `el sol brilla` -> esperado `mucho` o `alto`; `si tengo hambre yo` -> esperado `como` o `comida`).

---

## Verification Plan

### Automated Tests
1. Ejecutar el entrenamiento piloto con la evaluación de Samantha modificada:
   ```bash
   systemd-run --user --scope -p MemoryMax=10G env PYTHONPATH=. .venv/bin/python src/bitnet/train_sovereign_school.py --test_mock --reset_state
   ```
2. Verificar que Samantha es capaz de evaluar el JSON correctamente y retornar la puntuación media basada en criterio cognitivo.
