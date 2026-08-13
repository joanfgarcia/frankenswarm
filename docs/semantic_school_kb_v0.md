# Escuela Semántica — Diseño v0

> KB común para el experimento comparativo G4: glyph-composicional (K-65P) vs
> vectores-libres (inglés). Cada expresión de esta KB alimenta DOS corpus.
> Verificable por construcción: puente K-65P → Prolog (swipl) + bridge.py.
> 
> **75 expresiones validadas** (k65p.validator + bridge, 2026-08-10).

## Arquitectura

```
    KB causal (75 hechos + reglas en K-65P)
          │
          ├─→ Fábrica K-65P ──→ corpus nativo (10K-30K expresiones/etapa)
          │                     verificable: bridge → Prolog → gen_true
          │
          └─→ Curador inglés ──→ corpus v1 (mismas verdades, legible)
                                 evaluable: Samantha (compartido con v1 original)
```

**gen_true**: la continuación generada se convierte a Prolog y se demuestra/refuta
contra la KB compilada. Solo K-65P nativo puede usar esta métrica (el inglés no
tiene puente verificable). Es la ventaja metodológica de G4.

**Teoremas held-out**: consecuencias derivables de la KB que NUNCA aparecen
explícitamente en el corpus. Separan memoria de reasoning. Ejemplo:
- KB dice: `[if [touch X fire] [happen [G something bad] X]]` 
  + `[if [touch X [G fire hot]] [happen [G something bad] X]]` (más específico, no en corpus)
  → ¿genera el modelo la conclusión? → gen_true mide si la CONTINUACIÓN es Prolog-derivable.

## CAPA 1 — PREESCOLAR: hechos atómicos + atribuciones simples

**Existencia (13):** [exist agua], [exist fuego], [exist sol], [exist noche],
[exist árbol], [exist piedra], [exist tierra], [exist río], [exist cueva],
[exist bosque], [exist comida], [exist depredador], [exist yo_palabra]

**Atributos (17):** [good comida], [good agua], [good sol], [good dormir],
[bad fuego], [bad depredador], [bad herida], [bad tormenta],
[hot fuego], [hot sol], [cold agua], [cold noche],
[big sol], [big árbol], [big bosque], [small piedra], [small yo_palabra]

**Atributos con grupo G (4):** [good [G agua cold]], [bad [G fuego hot]],
[good [G comida big]], [bad [G depredador big]]

**Habilidades CAN (6):** [can [do yo_palabra comer]], [can [do yo_palabra beber]],
[can [do yo_palabra dormir]], [can [do yo_palabra mover_accion]],
[can [do depredador comer]], [can [do sol [G ver_accion light]]]

## CAPA 2 — PRIMARIA: acciones, causalidad simple, conectores

**Acciones DO (6):** [do yo_palabra comer comida], [do yo_palabra beber agua],
[do yo_palabra dormir cueva], [do depredador comer [other someone]],
[do sol [G ver_accion light] [above tierra]],
[do tormenta [G mover_accion agua] [above tierra]]

**Causalidad BECAUSE (8):** [because [touch someone fuego] [feel someone bad]],
[because [do someone beber agua] [feel someone good]],
[because [do someone comer comida] [feel someone good]],
[because [do someone dormir] [feel someone good]],
[because [touch someone [G depredador big]] [feel someone bad]],
[because [exist tormenta] [feel someone bad]],
[because [exist sol] [feel someone good]],
[because [exist noche] [can [do someone dormir]]]

**Condicionales IF (4):** [if [touch someone fuego] [happen [G something bad] someone]],
[if [do someone beber agua] [happen [G something good] someone]],
[if [not [do someone comer comida]] [feel someone bad]],
[if [not [do someone beber agua]] [die someone]]

**Conectores LIKE (4):** [like [G fuego hot] [G sol light]],
[like [G agua cold] [G río water]], [like [G noche dark] [G dark very]],
[like [G sol light] [G light very]]

## CAPA 3 — SECUNDARIA: implicaciones anidadas, negación

**IF anidado (4):** [if [touch someone [G fuego hot]] [happen [G something bad] someone]],
[if [touch someone [G agua cold]] [not [happen [G something bad] someone]]],
[if [do someone dormir [for some time]] [feel someone good]],
[if [not [do someone dormir [for some time]]] [feel someone bad]]

**Negación contrastiva (4):** [not [exist depredador]],
[not [good fuego]], [if [not [exist agua]] [die someone]],
[if [not [do someone beber agua]] [die someone]]

**BECAUSE anidado (2):** [because [exist fuego] [because [hot fuego] [feel someone bad]]],
[because [exist depredador] [because [big depredador] [feel someone bad]]]

**LIKE anidado (1):** [like [G [G fuego hot] like sol] [G sol light]]

**IF + BECAUSE combinados (1):** [if [exist tormenta] [can [do someone [move someone [far someone]]]]]

**Cuantificadores con G (3):** [die [G people all]],
[not [die [G people some]]], [not [die [G people some]]]

## Volumen objetivo y protocolo

- **Volumen**: fábrica genera ~5K-10K expresiones/etapa (3 etapas = 15K-30K totales)
  a partir de las combinaciones sintácticas de la KB. Comparable al v1 (~300K muestras
  con repeticiones ×300), pero con VERDAD por construcción en vez de repeticiones ciegas.
- **Protocolo**: DL-006 adaptativo (plateau → examen → avance → Net2WiderNet si suspenso).
  Catas iterativas: se entrena hasta que gen_true >= 0.70, no hasta un contador fijo.
- **Corpus inglés**: el curador produce frases equivalentes a las K-65P usando la misma KB como spec.
  Mismo volumen, mismas etapas. Samantha evalúa.
- **Comparación G4**: K-65P nativo se evalúa con gen_true (juez Prolog, teoremas held-out).
  Inglés se evalúa con Samantha (como v1). Métrica común: épocas-hasta-hito, params,
  margen sobre baseline propio.

## Próximo paso: implementación

1. `scripts/generate_semantic_corpus.py` — fábrica que genera expresiones a partir de la KB
2. `scripts/exam_gen_true.py` — evaluador que usa bridge + swipl para verificar generaciones
3. Recetas `script_job` para la escuela semántica
4. Reentrenamiento de v1-inglés bajo DL-006 si la diferencia con K-65P es abismal
