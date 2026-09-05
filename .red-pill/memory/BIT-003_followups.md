# BIT-003 · Estado 2-sep (confrontación edad-2 + K-65P)

## RESET K-65P (DL-016/DL-017, 2-sep): vocabulario refundado desde los 65 primos
- Los "gold" heredados (28 canonical + 54 drafted + 40 molecule, marcados `LEGADO-2SEP` en la BD) entran en CUARENTENA: eran bolsas de rasgos sin idea fuente, contaminados por la era supervivencia. Ya no calibran nada.
- Nuevo orden: idea fuente en INGLÉS (operador cura) → cláusulas K-65P → glifo = proyección. Fuente editable en inglés, forma canónica numérica generada (convención del operador).
- Capa 0 construida: tabla de 65 primos (identidad, valencias, kind) en configs/k65p_v2/primos.json.
- **Tres leyes DL-017 implementadas** (registry_v2.py): L1 glifo todo-a-cero = SILENCIO reservado (abstención, cierra el canal de silencio); L2 inyectividad (huella única contra primos y moléculas — atrapa colisiones exactas, NO definiciones incompletas: eso es curación); L3 los trits −1 son definición (hielo={agua,frío,mover:−1} ≠ agua fría con mover:+1).
- Primeras 2 moléculas v2: cold-water, ice (demo del pipeline).
- **DL-018 implementado**: gramática unaria de moléculas en k65p/validator (commit c9ada8d, 117 tests) — [lobo peligro]=compuesta vs [peligro lobo]=predicación, distinción semántica; malla recursiva (las cabezas-molécula propagan trits); F v1 cabeza-dominante. **Hallazgo empírico**: F puntual no preserva orden ni contrastes nuevos (menace y still-animal colisionan) — inyectividad como guardián, dim66 como solución estructural cuando se calibre.
- Registro v2: cold-water, ice, animal, danger, predator, still-thing (configs/k65p_v2/moleculas.json). Definiciones en CADENA (DL-020): baby=child+very-small — la genealogía queda en la huella; VERY(49) operador unario.
- **DL-021 nombres propios**: marcador `[N partes...]` (debate cerrado: sobre {nombre} por pureza S-expr) — símbolos sin huella, jamás cabeza, relaciones ≥2 args solo con [N], bridge aplana a átomo Prolog, aliases = relaciones (same-person/2). KB demo: facts/kb_demo.pl. Hecho vs episódico = misma sintaxis, distinto uso estadístico. Thomas cerrada como marcador.
- Auditoría externa 3-sep REMEDIADA (cd22237): en_lexicon NSM-fixes (a/an→ONE, interrogativos fuera del drop), guardia anti-ciclos, riddles fuera del diccionario, DL-019 cableada al trainer, CHANGELOG al día. Doctrina: el proyecto es un COMPILADOR EN→K-65P (frontend/backend/diccionario/decompilador), no un traductor.
- Prioridad: mecanismo de COMPUESTAS/clase-de — con 65 primos y nada más, toda molécula es composición (3^65). Diseño conversado: (lobo peligro)=compuesta vs (peligro lobo)=predicación, F pendiente de calibrar (multiplicidad→primos de cantidad; orden→explicación; si no basta→dim66).
- La jungla (tribu) usará los glifos del vocabulario refundado cuando se retome.

## PLAN refundación→modelo (acordado 2-sep, orden de dependencia)
1. **Vocabulario v2 inicial** (~30-40 moléculas): Aleth redacta ideas fuente EN inglés en tiers Wierzbicka; operador cura; registro con inyectividad.
2. **Corpus v2**: mini-documentos con orden-dependencia de consecuencias (cura el corpus orden-rígido), compuestas en uso, contextos de silencio (L1).
3. **Bit v2 K-65P nativo**: Escuela Soberana adaptada al vocabulario v2, embeddings precargados con la malla, gating por tiers. **Neurogénesis DL-019**: arranque 32d (variante agresiva 16d), Δdim=max(8,dim//8) [+12.5%], SIN techo (guardia 4096 → pausa operador), punto de medición en d=65 (ancho de la malla). Módulo: neurogenesis_policy.py.
4. **Instrumento**: permutación v2 + batería — métrica: Δ perms válidas sube de +0.11, OOD generaliza. Hipótesis falsable: *compartimiento principiado transfiere; arbitrario interfiere*.
5. **Banco de comunicación A↔B**: canal inglés vs K-65P, información que sobrevive al cuello de botella discreto.
- Paralelo: 3_years glyph / 5_years standard abiertos (runs acabaron sin responder); push pendiente de rama feat/bit003-corpus-csr.
- **PASO 0 NUEVO (propuesta operador)**: corpus mínimo = "traducción justa" del corpus gateado de etapa 0-1 (máx 2 años) para la primera cata. Ver assessment: el gate de etapa 0 DEFINE el set de moléculas de arranque.

## Confrontación edad-2 congelada (age2_v2: 39 vistas / 163 no-vistas) — misma batería, ambos 128d ×1
| instrumento | glyph v41 · 2y @128d | standard v4 · 2y @128d | lectura |
|---|---|---|---|
| gate | 59.0% | 61.5% | ruido (p≈1.0) |
| cognición | 4.3% (7/163) | **9.8%** (16/163) | p≈0.048, limítrofe |
| retención | 25.3% | **39.2%** | p≈0.07, limítrofe |
| S4 aritmética | 0% | **14.3%** | |
| S5 predicado | **6.9%** | 3.4% | única señal del glyph, n=29 |
- El standard gana dirección en todo. Gap de val por etapa (stage_history): 0-1 → 2.01 vs 1.68; 1-2 → 3.02 vs 2.22. **El gap se dobla con el currículo** — firma de la hipótesis de interferencia (glifos correlacionados = crosstalk en 128d). El glyph cerró 1-2 antes (46 vs 66 ép): dejó de mejorar antes, no eficiencia.
- Generador de batería parametrizado por edad (`generate_battery_v3.py --age/--stage_idx`); age2_v2.json (S1:40 S2:34 S3:32 S4:28 S5:29).

## F2 cerrado: retractación definitiva de DL-015
- `bit003_glyph_v41_2y_x1`: **2_years APROBADO @128d** (46 ép de etapa), 0 neurogénesis, contrato exit-codes activo. El suspenso original era infra pura. La confrontación vuelve al terreno igualado a 128d — y ahí gana el standard (tabla arriba).

## Runs — ambos jobs terminaron por cierre de cola, SIN sus preguntas decisivas
- glyph v41 (eaeee8dc): acabó época 125, etapa 2-3 — **sin examen de 3_years** (pregunta abierta).
- standard v4 (897ec6e3): acabó época 241, etapa 4 — hitos 2/3/4y @128d sin neurogénesis, **sin 5_years**. Examen 4y: 10.00/10.
- v3 ×10 (250de371) PAUSED resumable. GPU libre (0%, 16 MiB) al 2-sep.

## K-65P — DOBLINEAGE, no traducción (cambio de doctrina, 2-sep)
- **Sí existe un Bit entrenado nativo en K-65P**: la Escuela Soberana (`storage/checkpoints/sovereign_school_k65p/`, 256d, 162 ép, `target_milestone: completed`, protocolo DL-006) con corpus factory Prolog de 10.9k secuencias (3k/3.6k/4.2k por etapa) + 105 OOD holdout. Adversarial: 100% generaciones válidas, OOD acc 47.4%, ataque1 50.8% vs bigrama 44% — separación mínima del baseline.
- **DL-004 ya deprecó la traducción literal ES→K-65P por fidelidad nula** — el generador composicional la sustituyó. El en_lexicon del brazo frankenswarm es un TOKENIZADOR (palabra→glifo), no la traducción interpretativa de K-65P.
- La visión del operador (glifo=concepto, no palabra; frase→molécula compuesta con MENOS glifos; permutación→otra molécula) YA vive en el lineage Escuela Soberana (grupos G, gramática S-expr head-first). Los experimentos frankenswarm (tokenizer) NO miden ese potencial — por eso no aparece.
- **Radicales ideográficos = trits compartidos, verificado**: agua/fuego/comida/sol comparten posiciones 4 y 8 de sus 65 trits (componentes semánticos = primos compartidos). La intuición kanji del operador está estructuralmente incorporada; testeable con correlación Hamming↔semántica sobre el léxico (pendiente).
- Pendientes K-65P: (a) test de permutación (¿perplexity distingue secuencias canónicas de permutadas?), (b) batería cognitiva K-65P nativa sobre la Escuela Soberana (comprensión de compuestos, OOD como S6), (c) investigación ideográfica (kanji/cuneiforme/jeroglífico) como fuente de diseño.

# Estado 1-sep (post-auditoría DL-015) — ARCHIVO
⚠️ Ítems 1-2 de Próximos ya hechos; ítem 4 (traductor EN→K-65P) SUPERSEDED por la doctrina doblineage de arriba — el puente entre lineages no es un traductor.

## AUDITORÍA 1-SEP: dos retractaciones + instrumento corregido (DL-015)
- El suspenso de 2_years del glyph v4 a 128d fue un CRASH de infra (Samantha → None), no una nota: la neurogénesis 128→256 se disparó sin examen real. Contrato de exit-codes nuevo (0/2/3) + reintento + pausa `eval_infra_error`; el trainer ya no remedia sin calificación. 8/8 tests.
- La "curva riesgo-cobertura" del 31-ago era artefacto (conf nunca calculada). Con confianza real: calibración fuerte DENTRO de lo visto (top-20% → 90-100%), casi nula en lo no visto (0-5.4%).
- Runner de batería corregido (tokenize sin `<unk>`, dedupe, retención+procedencia persistidas) y RE-TIRADO sobre los 6 checkpoints.
- Banks regenerados con in-gate vivo (442→439; fuera burns/howl/hiss). Batería v2 congelada (`age4_v2.json`, 51/169, banks en el universo — 28 colisiones purgadas). v1 rige la confrontación en curso; v2 desde que los banks se cableen.

## Batería congelada v1 re-tirada (56 vistas / 189 no-vistas) — NÚMEROS CANÓNICOS
| checkpoint | dim | gate | cognición | retención |
|---|---|---|---|---|
| glyph v4 ×1 · 2y | 256 | 44.6% | 1.6% | 19.2% |
| glyph v4 ×1 · 4y | 256 | 69.6% | 4.2% | 30.8% |
| standard v4 ×1 · 2y | 128 | 51.8% | 5.8% | 25.6% |
| standard v4 ×1 · 3y | 128 | 67.9% | 3.2% | **42.3%** |
| glyph v3 ×10 · 4y | 128 | 69.6% | 1.6% | 24.4% |
| glyph ×300 v2 · 4y | 128 | 64.3% | 1.6% | 23.1% |
- Cognición 1.6-5.8% = suelo (dentro del ruido, n=189). Interrogativo y riddles: 0/39 en LOS SEIS. Abstenciones: 0 en los seis.
- ⚠️ El "12%" de cognición que circulaba (followups viejo, recetas) era del set piloto de 50, NO reproducible. Cifras canónicas = esta tabla (JSONs con procedencia en cada state_dir).
- Ambos brazos dan respuestas token-idénticas en exámenes de hito 2y/3y → el examen de hito mide el corpus, no la representación.

## Runs
- v4 ×1 glyph (dd078baf) COMPLETO hasta 4_years @256d (ép. 321, en etapa 4; completion era 4_years). Su 2_years quedó SIN examinar de verdad (crash) — hueco abierto.
- v4 ×1 standard (897ec6e3) PROCESSING: 2y/3y aprobados @128d sin neurogénesis; en etapa 3-4 hacia 4_years. **El fix de exit-codes lo protege desde el próximo step** (el examen de 4_years ya no puede suspender por infra).
- v3 ×10 glyph (250de371) PAUSED — resumable. v2 ×300 archivado (ablación).
- Receta nueva `bit003_glyph_v41_2y_x1`: re-responde "¿pasa el glyph 2_years a 128d bajo ×1?" con examen real (etapas 0-2, misma semilla). LANZAMIENTO = decisión del operador, tras el 4_years del standard.

## Próximos
1. Standard v4 → 4_years: batería v1 sobre su checkpoint → primera comparación homóloga real glyph-4y vs standard-4y.
2. Lanzar `bit003_glyph_v41_2y_x1` (cierra la retractación 1).
3. Cablear banks (DL-013) a entrenamiento+examen → desde ese momento la batería es `age4_v2.json`.
4. Traductor EN→K-65P: compilador (transductor + léxico curado 09-01: 65/65 primos con superficie) + ida-vuelta RFC-002 §5.
5. Programa de descomposición NSM por lotes (DL-014) — con el guard de estados curados y `populate_k65p_dictionary.py` reproducible (pendiente: 99 moléculas sin marcar en la BD).
6. Canal de silencio ("i dont know"): decisión de diseño pendiente — el argmax forzado impide abstenerse; la matriz 3×2 ya lo premia.
7. Fase 1 dim66 (formato glifo v2 + migrador), post-v4.
