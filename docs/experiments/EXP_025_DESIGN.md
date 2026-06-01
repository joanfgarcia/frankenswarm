# 🧠 EXP_025: Lógica Proposicional con Vectores de Miedo

## Tesis

Si el lenguaje emergió cuando enseñamos vocabulario (EXP_005-006),
y la comparación relacional funcionó con >, <, igual (EXP_023-024),
el siguiente paso es enseñar **implicación causal** con **coste asimétrico**
basado en el vector de miedo.

La hipótesis: la lógica no emerge de la pura computación.
Emerge cuando equivocarse **duele**.

## Niveles del Curriculum

### Nivel 1: Clasificación (ya cubierto parcialmente por EXP_005)
- Input: [concepto] → Output: [categoría]  
- "gato" → "animal", "casa" → "lugar", "fuego" → "peligro"
- **Nuevo**: Añadir vector de miedo. "fuego" + fear=0.8 vs "agua" + fear=0.1

### Nivel 2: Negación y Contradicción (NUEVO)
- Input: [A, relación, B, afirmación] → Output: [verdad/falsedad]
- "gato igual gato verdad" → verdad
- "gato igual perro verdad" → falsedad (contradicción)
- "fuego igual seguridad verdad" → falsedad (contradicción peligrosa, fear=0.9)

### Nivel 3: Modus Ponens — Si A entonces B (NUEVO - EL OBJETIVO)
- Input: [causa, implica, efecto, causa_presente] → Output: [efecto_presente]
- "fuego implica peligro, fuego" → "peligro"
- "agua implica seguridad, agua" → "seguridad"  
- "hambre implica dolor, hambre" → "dolor"

### Nivel 4: Cadenas Lógicas (FUTURO)
- "fuego implica peligro, peligro implica miedo, fuego" → "miedo"
- Requiere estado a lo largo de 3+ tokens de contexto

### Nivel 5: Modus Tollens (FUTURO)
- "fuego implica peligro, NO peligro" → "NO fuego"
- Razonamiento hacia atrás — el más difícil

## Cambios respecto a EXP_024

| Aspecto | EXP_024 | EXP_025 |
|---|---|---|
| Tarea | A > B → verdad/falsedad | A implica B, A → B |
| Vocabulario | números + >, <, igual | conceptos + implica + emociones |
| Output | verdad/falsedad | concepto resultante |
| Vector miedo | No | Sí (modifica loss weight) |
| Tipo de razonamiento | Comparación | Inferencia causal |
| Formato input | [op_a, rel, op_b, 0] | [causa, implica, efecto, causa_presente] |
| Formato output | [op_a, rel, op_b, result] | [causa, implica, efecto, efecto_presente] |
