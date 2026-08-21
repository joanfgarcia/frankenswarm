# BIT-003 · Auditoría pre-lanzamiento del gateo (2026-08-21, Aleth)

Veredicto: **NO LANZAR** hasta resolver B1-B5. La semántica documentada (DL-008/RFC) es correcta y reproducible; el trainer no la implementa.

## Bloqueantes
- **B1 NameError**: `train_sovereign_school.py:296` usa `ts_dataset` antes de cargarlo (la carga vive en el else de caché-miss, :321-327). Todo run muere al arrancar → nada de este pipeline ha corrido jamás.
- **B2 Censo por ids**: `childes_freq.update(seq)` cuenta token-IDs (:311-313, :378-380) pero `build_stage_logit_mask` hace `word_to_idx.get(w)` esperando palabras → el top-N no desbloquea NADA. Máscaras reales: [107,107,107,107,161,161,179,20095] (RFC esperaba [242,822,3007,...]); clasificación real: 99% del corpus a E7. Fix: censar palabras (idx_to_word) EXCLUYENDO `<pad>`/`<unk>` (si no, `<unk>` entra al top-200: 44.728 apariciones > corte 6.161).
- **B3 Device mismatch**: `loss_mask_for_gate(mask, targets, a_stage_mask)` recibe la máscara en CPU con targets en CUDA (:778, :797, :981, :1012, :1042) → RuntimeError en el primer batch GPU. En CPU no peta (por eso los smokes no lo ven). Fix: mover `stage_mask` a device al cargar etapa.
- **B4 Exámenes imposibles**: `school_exams_en.json` y AGE_QUESTIONS (29-jul, pre-rebuild) esperan respuestas fuera del vocab nuevo: `aleth`, `bunker`, `madrid` (edades 5-8) → a_token=`<unk>`, inentrenables e inemitibles → el protocolo adaptativo se atascaría en E4/E6 (neurogénesis inútil + rc=78) tras días de GPU. Decisión de operador: reponer esas palabras al vocab+glifos o reescribir exámenes.
- **B5 'xxx' en vocab**: marcador CHAT de habla ininteligible, 50.894 apariciones (rank ~27 del censo) → se desbloquea en E0 y se entrena como target. Bit aprendería a decir "xxx". También `www`/`yyy` (273 c/u). Limpiar en el rebuild de CHILDES o vetar.

## Serios (ensucian la tesis)
- **S1 Instrumento inflado**: `get_allowed_vocab_for_age` ahora traga TODO CHILDES-en sin corte → permite 18.364/20.095 palabras a edad 2 vs 822 del gate de training. El evaluador ya no mide la política de producción; es un cambio de instrumento (DL-004) sin DL propio. Alinear con `build_stage_logit_mask` (compartir código).
- **S2 Contracciones partidas**: CHILDES-en reconstruido SIN apóstrofos (`dont`=8.753, `don't`=0) pero el trainer tokeniza TinyStories crudo CON apóstrofos → `don't/it's/i'm` (en vocab, censo 0) caen a min_stage 7; TinyStories arrastrada a etapas tardías por normalización, no por edad real. Además el regex del evaluador (sin `'`) hace inemitibles las 613 formas con apóstrofo. Unificar normalización.
- **S3 E7 permite `<unk>`**: `masks[-1]=zeros` contradice DL-008 ("vetado en todas las etapas"); 0,47% de targets UNK entran en pérdida en E7. Fix: `masks[-1][1]=-inf`.
- **S4 Currículo preescolar = 4 saludos** casi idénticos (con "maureen", nombre de hablante filtrado) y por MLU los 4 caen en curr_3_4 → curr_0_1/1_2/2_3 VACÍOS. Revisar `rebuild_curriculum_en.py` (98 primary vs 4 preschool).
- **S5 Garantía de gate incompleta**: cubre palabras del currículo pero no de exámenes — `moon` (examen edad 2) vetado en E1 → sus ×300 secuencias nunca entrenan la respuesta. Añadir exam_words al gate de su etapa.

## Menores
- epoch_loss dividido por `samples_per_epoch` fijo aunque la etapa tenga menos secuencias (log desinflado; val_loss OK).
- OOM→CPU solo en bucle clásico; el adaptativo no captura OOM. Añadir `--require_gpu` a la receta.
- Cachés huérfanas del 14-ago (tokenized_corpus.json 110MB + stage_cache) — el hash las ignora; borrarlas.
- `nsm_physics_pre_school.json` sigue en ES (solo alimenta el allowed del evaluador).
- Restos ES anecdóticos en CHILDES-en ("mira el panda...").
- Vocab con tokens basura (`'`, `''`, `'boo`, `-`, `--`; 613 con apóstrofo + 124 con guión).
- `evaluate_exam` en el trainer es código muerto (sin call sites); el instrumento real es `evaluate_samantha_age.py` vía `run_samantha_eval` (dims pasadas correctamente ✓).
- `samples_per_epoch`: default 250k; decisión pendiente 400k → pasar `--samples_per_epoch 400000` explícito.

## Verificado OK
Hash de corpus cubre glifos+diálogos+currículo+exámenes+CHILDES+n_stories ✓ · glifos 20095×65, sin filas cero, sin duplicados ✓ · `<pad>`=0/`<unk>`=1 ✓ · clean_vocabulary==words ✓ · cobertura CHILDES 99,53% ✓ · diálogos EN limpios (0 turnos perdidos, 0 OOV, max 44 tokens) ✓ · con semántica por-palabras se reproducen las tablas de DL-008 ✓.
