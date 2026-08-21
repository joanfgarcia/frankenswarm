# BIT-003 · Plan de implementación — remediación pre-lanzamiento del gateo (2026-08-21)

> **Origen**: auditoría `BIT-003_audit_findings.md` (misma carpeta). **Decisiones del operador ya tomadas** (21-ago):
> exámenes reescritos SOLO con vocabulario in-vocab y ampliados a 10 preguntas/edad · pipeline completo de vocab
> (apóstrofos+artefactos+glifos canónicos) · set preescolar rico · `samples_per_epoch=400k`.
> **Este plan no deja decisiones abiertas: ejecutar tal cual.** Todos los textos nuevos (exámenes, currículo) están
> verificados contra el vocabulario; si el validador de F3.4 detectara algún OOV tras regenerar el vocab, la palabra
> se sustituye por la alternativa indicada en su tabla — nunca inventar texto nuevo.

## Reglas para el coder

1. Repo: `/home/joan/Documents/IA/frankenswarm`. Rama de trabajo: `feat/v1-english-rebuild` (ya existe, seguir en ella). **NO push, NO tocar main.**
2. Un commit por fase, mensajes dados al final de cada fase (Checkpoint Protocol).
3. Indentación: respetar la del archivo (src/ y scripts/download_childes_en.py usan **tabs**; `scripts/rebuild_*.py` usan 4 espacios). Sin placeholders, sin TODOs.
4. Los bloques `ANTES →` deben localizarse por CONTENIDO (los números de línea son orientativos del estado actual y se desplazan al editar).
5. Python del repo: `.venv/bin/python`, ejecutando desde la raíz del repo.
6. Comandos pesados (censo, glifos, smokes) envueltos en OOM shield: `systemd-run --user --scope -p MemoryMax=10G <cmd>` (subir a 16G si el host lo permite para la tokenización full).
7. **NO tocar**: `configs/childes_pre_school.json` (ES, congelado DL-004), `configs/school_curriculum_structured.json` (fuente ES del rebuild), `TOP_N_BY_STAGE`, el repo k65p, nada de `storage/checkpoints/` existente.
8. No lanzar el entrenamiento real: este plan termina en smokes + runbook.

---

## F0 · Preparación

```bash
cd /home/joan/Documents/IA/frankenswarm
git status   # debe estar limpio, en feat/v1-english-rebuild
mkdir -p storage/backups
cp configs/childes_pre_school_en.json storage/backups/childes_pre_school_en.pre-fix.json
rm -f storage/datasets/tokenized_corpus.json
rm -rf storage/datasets/stage_cache
```
(Las cachés borradas son del 14-ago, pre-rebuild; el hash las ignoraría igualmente — se quitan para no confundir y liberar ~110MB. El backup del CHILDES es local, no se commitea: añadir `storage/backups/` a `.gitignore` si no está cubierto.)

---

## F1 · Pipeline corpus/vocabulario (B5 + S2 + S6)

### F1.1 Limpiar artefactos CHAT del corpus CHILDES-en (B5)

**Nuevo archivo** `scripts/clean_childes_en_artifacts.py` (tabs):

```python
"""Limpia artefactos CHAT del corpus CHILDES-en (BIT-003, fix B5).

xxx/yyy/www son marcadores de transcripción CHAT (habla ininteligible), no
palabras: 'xxx' aparecía 50.894 veces (rank ~27 del censo) y E0 habría
entrenado a Bit a producirlo. Se eliminan como TOKEN completo (no subcadena),
se descartan las oraciones que queden con <2 tokens y se re-deduplica.
Idempotente: correrlo dos veces no cambia nada.
"""
import json
import os

ARTIFACTS = {"xxx", "yyy", "www", "xx", "yy"}


def main() -> None:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	path = os.path.join(base_dir, "configs", "childes_pre_school_en.json")
	with open(path, encoding="utf-8") as f:
		sentences = json.load(f)

	cleaned = []
	dropped_tokens = 0
	dropped_sentences = 0
	for s in sentences:
		tokens = s.split()
		keep = [t for t in tokens if t not in ARTIFACTS]
		dropped_tokens += len(tokens) - len(keep)
		if len(keep) < 2:
			dropped_sentences += 1
			continue
		cleaned.append(" ".join(keep))

	seen = set()
	unique = []
	for s in cleaned:
		if s not in seen:
			seen.add(s)
			unique.append(s)

	with open(path, "w", encoding="utf-8") as f:
		json.dump(unique, f, indent=4, ensure_ascii=False)
	print(f"✓ tokens artefacto eliminados: {dropped_tokens:,} | oraciones descartadas: {dropped_sentences:,} | únicas finales: {len(unique):,}")


if __name__ == "__main__":
	main()
```

Ejecutar: `.venv/bin/python scripts/clean_childes_en_artifacts.py` (esperado: ~51k tokens eliminados).

**Además**, en `scripts/download_childes_en.py` (paridad para futuras re-descargas):

ANTES:
```python
def clean_and_extract_words(line: str) -> list[str]:
	# Quitar signos de puntuación comunes
	line = re.sub(r"[?!.,;:\"()\[\]]", "", line)
	words = [w for w in re.findall(r"[a-zA-Z\-']+", line.lower()) if w]
	return words
```
DESPUÉS:
```python
CHAT_ARTIFACTS = {"xxx", "yyy", "www", "xx", "yy"}


def clean_and_extract_words(line: str) -> list[str]:
	# Quitar signos de puntuación comunes y apóstrofos (normalización BIT-003:
	# don't→dont, igual que la fuente AoA y que tokenization.words_of).
	line = re.sub(r"[?!.,;:\"()\[\]]", "", line).replace("'", "")
	# xxx/yyy/www son marcadores CHAT de habla ininteligible, no palabras.
	return [w for w in re.findall(r"[a-zA-Z\-]+", line.lower()) if w and w not in CHAT_ARTIFACTS]
```

### F1.2 Normalización de apóstrofos en la tokenización canónica (S2)

**`src/bitnet/training/modules/tokenization.py`** — reemplazar la función `tokenize` por:

```python
WORD_RE = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+")


def words_of(text: str) -> list[str]:
	"""Tokenización canónica BIT-003. Los apóstrofos se ELIMINAN (don't→dont):
	el corpus CHILDES-en upstream viene sin apóstrofos y mantener ambas formas
	partía la identidad de cada contracción en dos tokens (don't con censo 0 →
	gateada a E7; dont con censo real). Guiones se conservan."""
	return WORD_RE.findall(str(text).replace("'", "").lower())


def tokenize(text: str, word_to_idx: dict) -> list[int]:
	return [word_to_idx.get(w, 1) for w in words_of(text)]  # 1 is <unk>
```

Y en `format_and_tokenize_dialogue`, ANTES:
```python
			turn_words = [speaker] + re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>'\-]+", content.lower())
```
DESPUÉS:
```python
			turn_words = [speaker] + words_of(content)
```

**`src/bitnet/training/modules/stage_gating.py`** — `curriculum_words_by_stage`, ANTES:
```python
		for text in curriculum_data.get(stage, []):
			for w in re.findall(r"[a-zA-Z']+", text.lower()):
				if w in word_to_idx:
					words.add(w)
```
DESPUÉS (añadir arriba del archivo `from src.bitnet.training.modules.tokenization import words_of` y eliminar `import re` si queda sin uso):
```python
		for text in curriculum_data.get(stage, []):
			for w in words_of(text):
				if w in word_to_idx:
					words.add(w)
```

**`scripts/rebuild_english_vocabulary_v2.py`** (4 espacios) — el docstring exige regex idéntico al trainer; ahora se importa de verdad. ANTES:
```python
# Regex IDÉNTICO al de tokenization.py::tokenize (apóstrofes y guiones son token).
TOKEN_RE = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>'\-]+")


def words_of(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())
```
DESPUÉS:
```python
# Tokenización canónica importada del trainer: una sola fuente de verdad
# (apóstrofos eliminados, guiones conservados — BIT-003 S2).
from src.bitnet.training.modules.tokenization import words_of  # noqa: E402
```

### F1.3 Glifos canónicos EN + protección en el desempatador (S6)

**`src/bitnet/vocab/expand_vocabulary.py`** (tabs):

(a) Justo ANTES del bloque del desempatador (`# ── DESEMPATADOR DE GLIFOS DUPLICADOS ...`), reemplazar el bloque de preservación. ANTES:
```python
	# Conservar EXACTAMENTE los glifos de referencia originales para los tokens de referencia
	for idx, w in enumerate(FINAL_VOCAB):
		if w in VOCABULARY:
			all_glyphs[idx] = VOCABULARY[w]
```
DESPUÉS:
```python
	# Conservar EXACTAMENTE los glifos canónicos de referencia. El vocabulario
	# BIT-003 es EN: hay que cubrir la clave ES (VOCABULARY) Y su palabra EN
	# (VOCAB_MAP) — mirar solo la clave ES dejaba 7/28 referencias EN con glifo
	# proyectado por Ridge en vez del canónico ('i' difería en 5 de 65 bits).
	EN_CANON = {VOCAB_MAP.get(k, k): VOCABULARY[k] for k in VOCABULARY}
	for idx, w in enumerate(FINAL_VOCAB):
		if w in VOCABULARY:
			all_glyphs[idx] = VOCABULARY[w]
		elif w in EN_CANON:
			all_glyphs[idx] = EN_CANON[w]
```

(b) Dentro del desempatador, proteger también las referencias EN. ANTES:
```python
			ref_indices = [idx for idx in indices if FINAL_VOCAB[idx] in VOCABULARY]
			non_ref_indices = [idx for idx in indices if FINAL_VOCAB[idx] not in VOCABULARY]
```
DESPUÉS:
```python
			ref_indices = [idx for idx in indices if FINAL_VOCAB[idx] in VOCABULARY or FINAL_VOCAB[idx] in EN_CANON]
			non_ref_indices = [idx for idx in indices if FINAL_VOCAB[idx] not in VOCABULARY and FINAL_VOCAB[idx] not in EN_CANON]
```

### F1.4 Regenerar censo y glifos

```bash
systemd-run --user --scope -p MemoryMax=10G .venv/bin/python scripts/rebuild_english_vocabulary_v2.py --min-count 5 --n-stories 100000
systemd-run --user --scope -p MemoryMax=10G .venv/bin/python -m src.bitnet.vocab.expand_vocabulary
```
El censo tarda minutos (TinyStories ya está en la caché HF del 20-ago); los glifos ~5-10 min de fastembed en CPU. El vocab resultante será algo menor que 20095 (desaparecen ~613 formas con apóstrofo, artefactos y variantes muertas) — **no hay tamaño esperado fijo**; las propiedades se validan en F6.

### F1.5 Verificación de F1

```bash
.venv/bin/python - <<'EOF'
import json, sys
sys.path.insert(0, '.')
import numpy as np
from src.bitnet.vocab.glyph_vocabulary import VOCABULARY
from src.bitnet.vocab.expand_vocabulary import VOCAB_MAP
G = json.load(open('configs/expanded_glyphs.json'))
words, glyphs = G['words'], G['glyphs']
w2i = {w: i for i, w in enumerate(words)}
assert words[0] == '<pad>' and words[1] == '<unk>', 'pad/unk desplazados'
assert len(words) == len(set(words)) == len(glyphs), 'duplicados o desalineación words/glyphs'
assert all(len(r) == 65 for r in glyphs), 'fila de glifo != 65'
assert not any("'" in w for w in words), 'quedan formas con apóstrofo'
assert not ({'xxx', 'yyy', 'www'} & set(words)), 'artefactos CHAT en vocab'
assert not [i for i, r in enumerate(glyphs) if i > 1 and not any(r)], 'filas todo-cero'
bad = [w_en for k, w_en in VOCAB_MAP.items() if w_en in w2i and not (np.asarray(VOCABULARY[k]).astype(int) == np.asarray(glyphs[w2i[w_en]]).astype(int)).all()]
assert not bad, f'referencias sin glifo canónico: {bad}'
missing_ref = [w_en for w_en in VOCAB_MAP.values() if w_en not in w2i]
assert not missing_ref, f'referencias EN fuera del vocab: {missing_ref}'
ch = json.load(open('configs/childes_pre_school_en.json'))
assert not any(('xxx' in s.split()) or ('yyy' in s.split()) or ('www' in s.split()) for s in ch), 'artefactos en corpus'
print(f'✅ F1 OK — vocab={len(words)}, childes={len(ch):,} oraciones')
EOF
```

**Commit F1**: `fix(corpus): BIT-003 F1 — artefactos CHAT fuera, apóstrofos normalizados, glifos canónicos EN (B5+S2+S6)`

---

## F2 · Currículo preescolar rico (S4)

**`scripts/rebuild_curriculum_en.py`** (4 espacios):

(a) En la lista `SALUDA`, ANTES:
```python
    "hello baby, how are you? do you want to play with maureen",
```
DESPUÉS:
```python
    "hello baby, how are you? do you want to play with your friend",
```

(b) Justo DESPUÉS de la lista `SALUDA`, añadir la lista autorizada por el operador (los textos son EXACTOS, no modificar; el reparto por longitud puebla los 4 bins de MLU del particionador: ≤3 → 0-1, =4 → 1-2, 5-6 → 2-3, ≥7 → 3-4):
```python
# Set preescolar rico BIT-003 (decisión operador 21-ago): conversación y rutinas
# cotidianas con reparto deliberado por MLU para poblar curr_0_1..curr_3_4.
# Cada palabra está validada contra el censo (el validador de abajo lo garantiza).
PRESCHOOL_CORE = [
    # MLU ≤ 3 → curr_0_1
    "hello baby",
    "hello child",
    "dad is here",
    "mom is here",
    "i want milk",
    "i see you",
    "water is good",
    "time to sleep",
    "the red ball",
    "the dog runs",
    # MLU = 4 → curr_1_2
    "do you want milk",
    "the cat drinks milk",
    "i play with you",
    "the sun is hot",
    "i eat the apple",
    "the baby wants mom",
    "we go to bed",
    "i love you mom",
    # MLU 5-6 → curr_2_3
    "hello child, how are you",
    "do you want to play",
    "the dog plays with the ball",
    "i want to see the moon",
    "we eat bread and drink water",
    "the baby sleeps in the night",
    "come here and play with me",
    "the cat and the dog play",
    # MLU ≥ 7 → curr_3_4
    "do you want to play with your friend today",
    "i give you food and you give me a kiss",
]
```

(c) En `main()`, tras el bucle `by_stage[stage] = reconstruct(groups, stage)` y ANTES del `total = sum(...)`, añadir:
```python
    # Set preescolar rico (BIT-003 S4): items autorizados, deterministas.
    by_stage["preschool"].extend(
        make_item(text, "conversacion", "preschool_core", ["tú"], "preschool")
        for text in PRESCHOOL_CORE
    )
```

(d) Ejecutar y verificar (la validación RULE 7 del propio script debe dar 0 OOV; esperado preschool=32):
```bash
.venv/bin/python scripts/rebuild_curriculum_en.py
.venv/bin/python - <<'EOF'
import json, sys
sys.path.insert(0, '.')
from src.bitnet.training.modules.tokenization import words_of
from src.bitnet.training.modules.partitioner import partition_corpus_by_mlu
curr = json.load(open('configs/school_curriculum_structured_en.json'))['curriculum']
seqs = [words_of(it['text']) for it in curr['preschool']]
bins = partition_corpus_by_mlu(seqs)
print('bins MLU:', [len(b) for b in bins])
assert all(len(b) >= 2 for b in bins), 'algún bin MLU con <2 items'
print('✅ F2 OK')
EOF
```

**Commit F2**: `feat(curriculum): BIT-003 F2 — set preescolar rico, maureen fuera, 4 bins MLU poblados (S4)`

---

## F3 · Exámenes v2: solo vocabulario in-vocab, 10 preguntas por edad (B4)

### F3.1 `configs/school_exams_en.json`

Sustituir SOLO estas entradas (el resto del archivo queda byte-idéntico):

1. `primary.geografia`, entrada `{"question": "you: the capital of Spain is", "answer": "madrid"}` →
   `{"question": "you: rain falls from the", "answer": "sky"}`
2. `primary.conversacion` entera →
```json
"conversacion": [
    {"question": "you: what is your name", "answer": "bit"},
    {"question": "you: where are you from", "answer": "cave"}
]
```
3. La clave `secondary.borges_filosofia` se renombra a `secondary.literatura_filosofia` con este contenido:
```json
"literatura_filosofia": [
    {"question": "you: the long halls of the library are full of", "answer": "mirrors"},
    {"question": "you: a perfect memory keeps the shape of each", "answer": "cloud"},
    {"question": "you: one point can contain the whole", "answer": "universe"},
    {"question": "you: i dreamed that a man dreamed me in a", "answer": "dream"}
]
```
4. `secondary.conversacion` entera →
```json
"conversacion": [
    {"question": "you: what is your home", "answer": "cave"},
    {"question": "you: do you like books", "answer": "yes"}
]
```

### F3.2 `AGE_QUESTIONS` v2 (10 preguntas/edad) en `scripts/evaluate_samantha_age.py`

Reemplazar el dict COMPLETO `AGE_QUESTIONS = {...}` (líneas ~14-66) por (tabs):

```python
AGE_QUESTIONS = {
	2: [
		{"question": "you: hello", "expected": "hello"},
		{"question": "you: cat", "expected": "meow"},
		{"question": "you: water", "expected": "water"},
		{"question": "you: fire", "expected": "bad"},
		{"question": "you: mom", "expected": "dad"},
		{"question": "you: dog", "expected": "bark"},
		{"question": "you: the baby wants", "expected": "milk"},
		{"question": "you: night. time to", "expected": "sleep"},
		{"question": "you: the cat wants to", "expected": "eat"},
		{"question": "you: the ball is", "expected": "big"},
	],
	3: [
		{"question": "you: what is your name", "expected": "baby"},
		{"question": "you: the dog runs", "expected": "much"},
		{"question": "you: i want", "expected": "bread"},
		{"question": "you: if i touch the fire", "expected": "burns"},
		{"question": "you: where is dad", "expected": "here"},
		{"question": "you: the sun is", "expected": "hot"},
		{"question": "you: at night we", "expected": "sleep"},
		{"question": "you: the cat drinks", "expected": "milk"},
		{"question": "you: one and one are", "expected": "two"},
		{"question": "you: the dog is my", "expected": "friend"},
	],
	4: [
		{"question": "you: hello", "expected": "hello"},
		{"question": "you: how are you", "expected": "fine"},
		{"question": "you: who are you", "expected": "boy"},
		{"question": "you: the sun shines", "expected": "much"},
		{"question": "you: if i touch the fire", "expected": "hurt"},
		{"question": "you: the fish lives in the", "expected": "water"},
		{"question": "you: at night i see the", "expected": "moon"},
		{"question": "you: the bird has", "expected": "wings"},
		{"question": "you: the snow is", "expected": "cold"},
		{"question": "you: we read a", "expected": "book"},
	],
	5: [
		{"question": "you: i count one two", "expected": "three"},
		{"question": "you: one plus one is", "expected": "two"},
		{"question": "you: the bear eats", "expected": "honey"},
		{"question": "you: the bird flies", "expected": "high"},
		{"question": "you: the flowers drink", "expected": "water"},
		{"question": "you: two plus two is", "expected": "four"},
		{"question": "you: the week has seven", "expected": "days"},
		{"question": "you: the fish swims and the bird", "expected": "flies"},
		{"question": "you: four plus one is", "expected": "five"},
		{"question": "you: we sleep in a", "expected": "bed"},
	],
	6: [
		{"question": "you: what is three plus three? it is", "expected": "six"},
		{"question": "you: what is six minus four? it is", "expected": "two"},
		{"question": "you: the water of the river runs towards the", "expected": "sea"},
		{"question": "you: the trees give oxygen and", "expected": "shade"},
		{"question": "you: the heart pumps blood to the", "expected": "body"},
		{"question": "you: what is four plus five? it is", "expected": "nine"},
		{"question": "you: what is ten minus five? it is", "expected": "five"},
		{"question": "you: the moon shines at", "expected": "night"},
		{"question": "you: the roots of the tree are under the", "expected": "ground"},
		{"question": "you: we write words on a", "expected": "page"},
	],
	7: [
		{"question": "you: what is your name", "expected": "bit"},
		{"question": "you: where are you from", "expected": "cave"},
		{"question": "you: the earth moves around the", "expected": "sun"},
		{"question": "you: the maps show rivers and", "expected": "countries"},
		{"question": "you: rain falls from the", "expected": "sky"},
		{"question": "you: what is seven plus three? it is", "expected": "ten"},
		{"question": "you: the heart moves the", "expected": "blood"},
		{"question": "you: a story lives inside a", "expected": "book"},
		{"question": "you: the sun gives light and", "expected": "heat"},
		{"question": "you: winter is the season of", "expected": "snow"},
	],
	8: [
		{"question": "you: if x plus two is five then x is", "expected": "three"},
		{"question": "you: every cause produces an", "expected": "effect"},
		{"question": "you: the long halls of the library are full of", "expected": "mirrors"},
		{"question": "you: a perfect memory keeps the shape of each", "expected": "cloud"},
		{"question": "you: one point can contain the whole", "expected": "universe"},
		{"question": "you: what is your home", "expected": "cave"},
		{"question": "you: do you like books", "expected": "yes"},
		{"question": "you: written words overcome the passage of", "expected": "time"},
		{"question": "you: what is nine minus six? it is", "expected": "three"},
		{"question": "you: the poet sings to the moon in the cold", "expected": "night"},
	],
}
```

### F3.3 `dialogue_triggers` (3 sitios)

En `src/bitnet/training/modules/exam_compiler.py` (2 ocurrencias) y `scripts/evaluate_samantha_age.py` (1 ocurrencia), reemplazar el set:

ANTES: `{"hello", "how are you", "who are you", "what is your name", "where are you from", "what is the bunker", "do you like borges"}`
DESPUÉS: `{"hello", "how are you", "who are you", "what is your name", "where are you from", "what is your home", "do you like books"}`

### F3.4 Validador de vocabulario de instrumentos

**Nuevo archivo** `scripts/validate_exam_vocab.py` (tabs):

```python
"""Valida que TODO el texto de instrumentos (exámenes, AGE_QUESTIONS, currículo)
esté dentro del vocabulario y que cada respuesta sea UN token in-vocab (BIT-003 B4).
Falla duro (rc=1) listando palabra y origen. Correr tras cualquier cambio de
vocab/exámenes/currículo."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.bitnet.training.modules.tokenization import words_of  # noqa: E402
from scripts.evaluate_samantha_age import AGE_QUESTIONS  # noqa: E402

base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
words = set(json.load(open(os.path.join(base, "configs", "expanded_glyphs.json")))["words"])
errors = []


def check(text: str, origin: str) -> None:
	for w in words_of(text):
		if w not in words:
			errors.append((w, origin))


exams = json.load(open(os.path.join(base, "configs", "school_exams_en.json")))
for bucket, subjects in exams.items():
	for subject, qas in subjects.items():
		for qa in qas:
			check(qa["question"], f"exams/{bucket}/{subject}")
			check(qa["answer"], f"exams/{bucket}/{subject} (answer)")
			if len(words_of(qa["answer"])) != 1:
				errors.append((qa["answer"], f"exams/{bucket}/{subject}: respuesta no es 1 token"))

for age, qas in AGE_QUESTIONS.items():
	for qa in qas:
		check(qa["question"], f"AGE_QUESTIONS/{age}")
		check(qa["expected"], f"AGE_QUESTIONS/{age} (expected)")
		if len(words_of(qa["expected"])) != 1:
			errors.append((qa["expected"], f"AGE_QUESTIONS/{age}: expected no es 1 token"))

curr = json.load(open(os.path.join(base, "configs", "school_curriculum_structured_en.json")))["curriculum"]
for stage, items in curr.items():
	for it in items:
		check(it["text"], f"curriculum/{stage}")

if errors:
	print("❌ Palabras fuera del vocabulario en instrumentos:")
	for w, origin in errors:
		print(f"   {w!r} ← {origin}")
	raise SystemExit(1)
print("✅ Instrumentos 100% in-vocab (exámenes + AGE_QUESTIONS + currículo).")
```

Ejecutar: `.venv/bin/python scripts/validate_exam_vocab.py` → debe dar ✅.
**Tabla de sustitución si algo saliera OOV tras F1** (usar en orden, sin inventar): `wings→legs`, `ground→earth`, `page→book`, `halls→walls`, `heat→fire`, `seasons/season→time`, `bed→home`. Si falla otra palabra, detenerse y reportar al operador.

**Commit F3**: `feat(exams): BIT-003 F3 — instrumentos v2 solo-vocab, 10 preguntas/edad, triggers actualizados (B4)`

---

## F4 · Trainer: arranque, censo, device, E7, gate de exámenes (B1+B2+B3+S3+S5+M1)

Archivo: `src/bitnet/training/train_sovereign_school.py` (tabs) salvo indicación.

### F4.1 (B1) Cargar TinyStories ANTES del hash

ANTES (líneas ~294-298):
```python
	# 3. Caché de corpus tokenizado
	tokenized_cache_path = os.path.join(base_dir, "storage", "datasets", "tokenized_corpus.json")
	n_stories = len(ts_dataset) if args.full_tinystories else min(100000, len(ts_dataset))
	corpus_hash = compute_corpus_hash(base_dir, n_stories)
	cached = None if args.force_tokenize else load_tokenized_cache(tokenized_cache_path, corpus_hash)
```
DESPUÉS:
```python
	# 3. Caché de corpus tokenizado
	# TinyStories se carga SIEMPRE antes del hash: n_stories (que entra en el
	# hash) depende de len(ts_dataset); cargarlo solo en el caché-miss dejaba
	# NameError en todo arranque (fix B1). Con caché HF local es barato.
	tokenized_cache_path = os.path.join(base_dir, "storage", "datasets", "tokenized_corpus.json")
	tiny_stories_cache = os.path.join(base_dir, "storage", "datasets", "tiny_stories")
	os.makedirs(tiny_stories_cache, exist_ok=True)
	from datasets import load_dataset

	if args.force_download:
		print("📖 Forzando re-descarga de TinyStories desde HF Hub...")
		ts_dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=tiny_stories_cache, force_redownload=True)
	else:
		print("📖 Cargando TinyStories (caché HF local o descarga inicial)...")
		ts_dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=tiny_stories_cache)
	n_stories = len(ts_dataset) if args.full_tinystories else min(100000, len(ts_dataset))
	corpus_hash = compute_corpus_hash(base_dir, n_stories)
	cached = None if args.force_tokenize else load_tokenized_cache(tokenized_cache_path, corpus_hash)
```

En la rama `else` (caché-miss), ELIMINAR el bloque ahora redundante — desde `tiny_stories_cache = os.path.join(...)` hasta `n_stories = len(ts_dataset) if args.full_tinystories else min(100000, len(ts_dataset))` inclusive (las líneas ~315-329 originales: creación de dir, `from datasets import load_dataset`, el if/elif/else de `load_dataset` y el recálculo de `n_stories`), dejando el `print(f"  ✓ {n_stories:,} historias disponibles en TinyStories.")` y todo lo demás.

Al FINAL de la rama `else`, ELIMINAR estas tres líneas:
```python
		del ts_dataset
		import gc

		gc.collect()
```
y añadirlas DESPUÉS del if/else completo (justo antes del comentario `# El corpus preescolar principal son las TinyStories tokenizadas`), desindentadas un nivel:
```python
	del ts_dataset
	import gc

	gc.collect()
```

### F4.2 (B2) Censo por palabras, excluyendo `<pad>`/`<unk>`

Hay DOS bloques idénticos (rama caché-hit ~311-313 y rama miss ~378-380). En AMBOS, ANTES:
```python
		childes_freq = Counter()
		for seq in tokenized_childes:
			childes_freq.update(seq)
```
DESPUÉS:
```python
		# Censo por PALABRA, no por id: build_stage_logit_mask cruza el top-N con
		# word_to_idx — un Counter de ids no desbloqueaba nada (fix B2). t > 1
		# excluye <pad>/<unk>: <unk> (~0.5% del corpus) entraría al top-200.
		childes_freq = Counter()
		for seq in tokenized_childes:
			childes_freq.update(idx_to_word[t] for t in seq if t > 1)
```

### F4.3 (S5) Gate de exámenes + `stage_config` antes de las máscaras

(a) En `src/bitnet/training/modules/exam_compiler.py`, añadir al final:
```python
def exam_answer_words_for_age(age: int, exams_data: dict, dictionary) -> set[str]:
	"""Palabras de RESPUESTA de la batería de examen de una edad (exams + AGE_QUESTIONS),
	ya mapeadas por el diccionario. Se inyectan en el gateo de su etapa: todo target
	de examen debe ser producible en la etapa que lo examina (BIT-003 S5)."""
	answers: set[str] = set()
	key = None
	if age in [2, 3, 4]:
		key = "preschool"
	elif age in [5, 6]:
		key = "primary"
	elif age in [7, 8]:
		key = "secondary"
	if key:
		for qas in exams_data.get(key, {}).values():
			for qa in qas:
				answers.add(dictionary.map_to_base_word(qa["answer"]))
	from scripts.evaluate_samantha_age import AGE_QUESTIONS
	for qa in AGE_QUESTIONS.get(age, []):
		answers.add(dictionary.map_to_base_word(qa["expected"]))
	return answers
```

(b) En `src/bitnet/training/modules/stage_gating.py`:

`build_stage_logit_mask` — añadir parámetro y unión. ANTES:
```python
def build_stage_logit_mask(
	stage_idx: int,
	vocab_size: int,
	word_to_idx: dict,
	childes_freq: Counter | None,
	curriculum_words: set,
) -> torch.Tensor:
```
DESPUÉS (y dentro, tras `allowed.update(curriculum_words)`, añadir `allowed.update(exam_words)`):
```python
def build_stage_logit_mask(
	stage_idx: int,
	vocab_size: int,
	word_to_idx: dict,
	childes_freq: Counter | None,
	curriculum_words: set,
	exam_words: set | None = None,
) -> torch.Tensor:
```
```python
	allowed.update(curriculum_words)
	allowed.update(exam_words or set())
```

`build_all_stage_masks` — ANTES:
```python
def build_all_stage_masks(
	vocab_size: int,
	word_to_idx: dict,
	childes_freq: Counter,
	curriculum_data: dict,
) -> list:
	masks = [
		build_stage_logit_mask(i, vocab_size, word_to_idx, childes_freq, curriculum_words_by_stage(curriculum_data, i, word_to_idx))
		for i in range(8)
	]
```
DESPUÉS:
```python
def build_all_stage_masks(
	vocab_size: int,
	word_to_idx: dict,
	childes_freq: Counter,
	curriculum_data: dict,
	exam_words_by_stage: list | None = None,
) -> list:
	exam_words_by_stage = exam_words_by_stage or [set()] * 8
	masks = [
		build_stage_logit_mask(i, vocab_size, word_to_idx, childes_freq, curriculum_words_by_stage(curriculum_data, i, word_to_idx), exam_words_by_stage[i])
		for i in range(8)
	]
```

(c) En el trainer, ANTES (línea ~409):
```python
	stage_masks = build_all_stage_masks(vocab_size, word_to_idx, childes_freq, curriculum_data)
```
DESPUÉS:
```python
	stage_config = get_stage_config(args.base_epochs, args.stage_scale)

	# Respuestas de examen acumuladas por etapa → entran al gateo (fix S5:
	# 'moon' era examen de edad 2 pero estaba vetado en E1, y sus ×300
	# secuencias nunca entrenaban la respuesta).
	exam_words_by_stage = []
	_acc_exam_words: set = set()
	for _cfg in stage_config:
		if _cfg["age"] is not None:
			_acc_exam_words |= exam_answer_words_for_age(_cfg["age"], exams_data, dictionary)
		exam_words_by_stage.append(set(_acc_exam_words))

	stage_masks = build_all_stage_masks(vocab_size, word_to_idx, childes_freq, curriculum_data, exam_words_by_stage)
```
Y ELIMINAR la línea posterior duplicada `stage_config = get_stage_config(args.base_epochs, args.stage_scale)` (estaba tras `# 4. Cargar o inicializar estado escolar`, línea ~541). Añadir `exam_answer_words_for_age` al import de `exam_compiler` en la cabecera:
```python
from src.bitnet.training.modules.exam_compiler import compile_exam_sequences_for_age, exam_answer_words_for_age
```

### F4.4 (S3) E7 sin `<unk>`

En `stage_gating.build_all_stage_masks`, ANTES:
```python
	# La etapa final cubre TODO el vocabulario: toda frase del corpus debe ser
	# clasificable (ninguna se pierde, solo se reordena por etapa).
	masks[-1] = torch.zeros(vocab_size)
```
DESPUÉS:
```python
	# La etapa final cubre TODO el vocabulario real: toda frase del corpus debe
	# ser clasificable (ninguna se pierde, solo se reordena por etapa). <unk>
	# sigue vetado también en E7 (salvaguarda DL-008: ruido de generación).
	final_mask = torch.zeros(vocab_size)
	final_mask[UNK_TOKEN] = float("-inf")
	masks[-1] = final_mask
```

### F4.5 (B3) Máscaras al device una sola vez

- Línea ~704 ANTES: `a_stage_mask = stage_masks[0]` → DESPUÉS: `a_stage_mask = stage_masks[0].to(device)`
- Línea ~740 ANTES: `a_stage_mask = stage_masks[a_stage_idx]` → DESPUÉS: `a_stage_mask = stage_masks[a_stage_idx].to(device)`
- Líneas ~774 y ~794: quitar el `.to(device)` inline → `apply_stage_gate(train_model(inputs, tau=tau), a_stage_mask)` y `apply_stage_gate(train_model(inputs_v), a_stage_mask)`
- Línea ~965 ANTES: `stage_mask = stage_masks[active_stage_idx]` → DESPUÉS: `stage_mask = stage_masks[active_stage_idx].to(device)`
- Líneas ~974, ~1005 y ~1036: quitar el `.to(device)` inline (queda `stage_mask` a secas).
- En el bloque `except torch.cuda.OutOfMemoryError` del bucle clásico, tras `device = torch.device("cpu")` y la migración del modelo/optimizador, añadir:
```python
					stage_mask = stage_mask.to(device)
```

### F4.6 (M1) Denominador del loss de época

En AMBOS bucles (adaptativo ~784 y clásico ~1023), ANTES:
```python
		epoch_loss /= (max_gen_seqs_per_epoch + len(train_curr))
```
DESPUÉS:
```python
		epoch_loss /= max(1, len(x_epoch))
```

### F4.7 Default de `samples_per_epoch`

En el argparse, ANTES: `default=250000,` → DESPUÉS: `default=400000,` y en el help: `250k ≈ 2.3M tokens/época` → `400k ≈ 3.7M tokens/época (análisis Chinchilla 21-ago, confirmado por operador)`.

**Commit F4**: `fix(train): BIT-003 F4 — arranque TinyStories, censo por palabras, máscaras en device, E7 sin unk, gate de exámenes (B1+B2+B3+S3+S5)`

---

## F5 · Evaluador alineado con el gateo de entrenamiento (S1)

### F5.1 Trainer persiste las máscaras

En `train_sovereign_school.py`, justo DESPUÉS de `os.makedirs(save_dir, exist_ok=True)` (línea ~539), añadir:
```python
	# Máscaras de gateo persistidas para el evaluador (DL-009): Samantha debe
	# permitir EXACTAMENTE el vocabulario que el entrenamiento dejó producir —
	# el gateo legado del evaluador tragaba todo CHILDES (18k palabras a edad 2).
	gate_masks_path = os.path.join(save_dir, "stage_gate_masks.json")
	with open(gate_masks_path, "w", encoding="utf-8") as f:
		json.dump({
			"corpus_hash": corpus_hash,
			"vocab_size": vocab_size,
			"stage_allowed": [torch.nonzero(m == 0).flatten().tolist() for m in stage_masks],
		}, f)
	print(f"🔒 [GATEO] Máscaras por etapa persistidas para el evaluador: {gate_masks_path}")
```

### F5.2 `state_manager.py` pasa las máscaras al subproceso

En `run_samantha_eval`, en la lista `eval_cmd` (tras `"--device", "cpu",`), añadir:
```python
		"--stage_masks",
		os.path.join(save_dir, "stage_gate_masks.json"),
		"--stage_idx",
		str(stage_idx),
```

### F5.3 `scripts/evaluate_samantha_age.py` consume las máscaras

(a) Argparse (tras `--test_mock`):
```python
	parser.add_argument("--stage_masks", type=str, default=None, help="JSON de máscaras de gateo por etapa persistido por el trainer (stage_gate_masks.json)")
	parser.add_argument("--stage_idx", type=int, default=None, help="Índice de etapa (0-7) cuya máscara de gateo usar")
```

(b) En `run_evaluation`, reemplazar el bloque de construcción de la máscara. ANTES:
```python
	# Construir logit mask de edad para restringir la generación al vocabulario del hito
	allowed_vocab = get_allowed_vocab_for_age(target_age, base_dir)
	special_tokens = {"me", "you", "<pad>", "<unk>", "hello", "mom", "dad", "baby", "kid", "meow", "bark", "water", "fire", "yes", "no", "fine", "bad", "bread", "good"}
	allowed_mask = torch.zeros(len(words), dtype=torch.bool, device=device)
	for w, idx in word_to_idx.items():
		if w in allowed_vocab or w in special_tokens or w.lower() in allowed_vocab:
			allowed_mask[idx] = True
	allowed_mask[0] = True
	allowed_mask[1] = True
```
DESPUÉS:
```python
	# Máscara de producción del hito. Camino preferente (DL-009): la máscara de
	# gateo REAL del entrenamiento, persistida por el trainer — el instrumento
	# permite exactamente lo que la etapa dejó producir. Fallback legado si no
	# hay fichero (checkpoints antiguos): get_allowed_vocab_for_age.
	stage_allowed_words = None
	allowed_mask = torch.zeros(len(words), dtype=torch.bool, device=device)
	use_trained_gate = False
	if args.stage_masks and args.stage_idx is not None and os.path.exists(args.stage_masks):
		with open(args.stage_masks, encoding="utf-8") as f:
			gate_data = json.load(f)
		if gate_data.get("vocab_size") == len(words):
			for idx in gate_data["stage_allowed"][args.stage_idx]:
				allowed_mask[idx] = True
			stage_allowed_words = {words[i] for i in gate_data["stage_allowed"][args.stage_idx]}
			use_trained_gate = True
			print(f"🔒 [GATEO] Máscara de entrenamiento E{args.stage_idx}: {int(allowed_mask.sum())} palabras producibles.")
		else:
			print(f"⚠️ [GATEO] vocab_size del fichero de máscaras ({gate_data.get('vocab_size')}) != vocab actual ({len(words)}). Fallback legado.")
	if not use_trained_gate:
		allowed_vocab = get_allowed_vocab_for_age(target_age, base_dir)
		special_tokens = {"me", "you", "<pad>", "<unk>", "hello", "mom", "dad", "baby", "kid", "meow", "bark", "water", "fire", "yes", "no", "fine", "bad", "bread", "good"}
		for w, idx in word_to_idx.items():
			if w in allowed_vocab or w in special_tokens or w.lower() in allowed_vocab:
				allowed_mask[idx] = True
	allowed_mask[0] = True   # <pad> permitido: es la señal de parada de generación
	allowed_mask[1] = False  # <unk> vetado, igual que en entrenamiento (DL-008)
```

(c) Monitor de vocabulario fuera de edad — usar la misma fuente. ANTES:
```python
	allowed_vocab = get_allowed_vocab_for_age(target_age, base_dir)
	oob_words_found = {}
```
DESPUÉS:
```python
	allowed_vocab = stage_allowed_words if stage_allowed_words is not None else get_allowed_vocab_for_age(target_age, base_dir)
	oob_words_found = {}
```

Nota: el bucle de generación ya trata `pred_token in [0, 1]` como parada y excluye pad/unk en el primer paso — no tocar.

**Commit F5**: `feat(eval): BIT-003 F5 — Samantha usa la máscara de gateo real del entrenamiento, fallback legado (S1/DL-009)`

---

## F6 · Script de auditoría permanente (acceptance test)

**Nuevo archivo** `scripts/audit_stage_gating.py` (tabs). Reproduce la auditoría del 21-ago con las funciones REALES del trainer; es el criterio de aceptación y la fuente de las tablas de la documentación:

```python
"""Auditoría del gateo BIT-003: máscaras, censo, clasificación e instrumentos.
Usa las MISMAS funciones que el trainer. rc=0 solo si todos los checks pasan.
Uso: .venv/bin/python scripts/audit_stage_gating.py"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np  # noqa: E402

from src.bitnet.training.modules.exam_compiler import exam_answer_words_for_age  # noqa: E402
from src.bitnet.training.modules.stage_config import get_stage_config  # noqa: E402
from src.bitnet.training.modules.stage_gating import (  # noqa: E402
	build_all_stage_masks,
	classify_sequences_by_gate,
	token_min_stage,
)
from src.bitnet.training.modules.tokenization import tokenize, words_of  # noqa: E402
from src.bitnet.vocab.dictionary_tool import SovereignDictionary  # noqa: E402
from src.bitnet.vocab.expand_vocabulary import VOCAB_MAP  # noqa: E402
from src.bitnet.vocab.glyph_vocabulary import VOCABULARY  # noqa: E402

base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
fails = []


def check(cond: bool, msg: str) -> None:
	print(("  ✓ " if cond else "  ✗ ") + msg)
	if not cond:
		fails.append(msg)


print("── 1. Vocabulario y glifos ──")
G = json.load(open(os.path.join(base, "configs", "expanded_glyphs.json")))
words, glyphs = G["words"], G["glyphs"]
w2i = {w: i for i, w in enumerate(words)}
vocab_size = len(words)
check(words[0] == "<pad>" and words[1] == "<unk>", "<pad>=0 y <unk>=1")
check(len(set(words)) == len(words) == len(glyphs), "sin duplicados, words/glyphs alineados")
check(all(len(r) == 65 for r in glyphs), "glifos de 65 dims")
check(not any("'" in w for w in words), "sin formas con apóstrofo")
check(not ({"xxx", "yyy", "www"} & set(words)), "sin artefactos CHAT en vocab")
check(not [i for i, r in enumerate(glyphs) if i > 1 and not any(r)], "sin filas de glifo todo-cero")
bad_canon = [w for k, w in VOCAB_MAP.items() if w in w2i and not (np.asarray(VOCABULARY[k]).astype(int) == np.asarray(glyphs[w2i[w]]).astype(int)).all()]
check(not bad_canon, f"28 referencias EN con glifo canónico (mal: {bad_canon})")

print("── 2. CHILDES-en ──")
childes = json.load(open(os.path.join(base, "configs", "childes_pre_school_en.json")))
seqs, childes_freq, n_unk, n_tok = [], Counter(), 0, 0
for s in childes:
	ids = tokenize(s, w2i)
	if not (2 <= len(ids) <= 64):
		continue
	seqs.append(ids)
	n_tok += len(ids)
	n_unk += sum(1 for t in ids if t == 1)
	childes_freq.update(words[t] for t in ids if t > 1)
check(not any(("xxx" in s.split()) for s in childes), "sin 'xxx' en el corpus")
coverage = 1 - n_unk / max(1, n_tok)
check(coverage >= 0.99, f"cobertura vocab ≥99% (real {coverage:.4%})")

print("── 3. Máscaras por etapa ──")
curr_json = json.load(open(os.path.join(base, "configs", "school_curriculum_structured_en.json")))["curriculum"]
curriculum_data = {k: [it["text"] for it in curr_json.get(k, [])] for k in ("preschool", "primary", "secondary")}
exams_data = json.load(open(os.path.join(base, "configs", "school_exams_en.json")))
dictionary = SovereignDictionary(os.path.join(base, "configs", "expanded_glyphs.json"))
stage_config = get_stage_config()
exam_words_by_stage, acc = [], set()
for cfg in stage_config:
	if cfg["age"] is not None:
		acc |= exam_answer_words_for_age(cfg["age"], exams_data, dictionary)
	exam_words_by_stage.append(set(acc))
masks = build_all_stage_masks(vocab_size, w2i, childes_freq, curriculum_data, exam_words_by_stage)
sizes = [int((m == 0).sum()) for m in masks]
print(f"  tamaños: {sizes}")
check(all(sizes[i] < sizes[i + 1] for i in range(7)), "tamaños estrictamente crecientes E0→E7")
check(sizes[0] >= 200, f"E0 ≥ 200 palabras (real {sizes[0]}) — el top-N desbloquea de verdad (B2)")
check(all(m[1].item() == float("-inf") for m in masks), "<unk> vetado en las 8 etapas (S3)")
check(all(m[0].item() == 0.0 for m in masks), "<pad> permitido en las 8 etapas")
xxx_ok = "xxx" not in w2i or all(m[w2i["xxx"]].item() != 0.0 for m in masks[:7])
check(xxx_ok, "'xxx' no producible antes de E7 (B5)")

print("── 4. Clasificación del corpus ──")
tms = token_min_stage(masks, vocab_size)
groups = classify_sequences_by_gate(seqs, tms)
dist = [len(g) / max(1, len(seqs)) for g in groups]
print("  CHILDES por etapa: " + ", ".join(f"E{i}:{p:.0%}" for i, p in enumerate(dist)))
check(sum(dist[:3]) >= 0.60, f"E0-E2 ≥60% de CHILDES (real {sum(dist[:3]):.0%})")
check(dist[7] <= 0.15, f"E7 ≤15% de CHILDES (real {dist[7]:.0%})")

print("── 5. Instrumentos dentro del gateo ──")
viol = []
for cfg in stage_config:
	if cfg["age"] is None:
		continue
	for w in exam_answer_words_for_age(cfg["age"], exams_data, dictionary):
		idx = w2i.get(w)
		if idx is None or masks[cfg["stage_idx"]][idx].item() != 0.0:
			viol.append((cfg["age"], w))
check(not viol, f"toda respuesta de examen producible en su etapa (mal: {viol})")

print("── 6. Longitudes ──")
dial = json.load(open(os.path.join(base, "configs", "tiny_dialogues_large_en.json")))
from src.bitnet.training.modules.tokenization import format_and_tokenize_dialogue  # noqa: E402
dlens = [len(format_and_tokenize_dialogue(d, w2i)) for d in dial]
check(max(dlens) <= 64, f"diálogos ≤64 tokens (max {max(dlens)}) — nada se descarta en bucketing")
clens = [len(words_of(t)) for k in curriculum_data for t in curriculum_data[k]]
check(max(clens) <= 64, f"currículo ≤64 tokens (max {max(clens)})")

print()
if fails:
	print(f"❌ AUDITORÍA: {len(fails)} checks fallidos.")
	raise SystemExit(1)
print("✅ AUDITORÍA COMPLETA: todos los checks OK. Copiar tamaños y distribución a RFC/DL.")
```

Ejecutar: `systemd-run --user --scope -p MemoryMax=10G .venv/bin/python scripts/audit_stage_gating.py` → **rc=0 obligatorio para continuar**.

**Commit F6**: `test(gating): BIT-003 F6 — auditoría permanente del gateo (scripts/audit_stage_gating.py)`

---

## F7 · Documentación

1. **`docs/DECISION_LOG.md`** — añadir al final:

```markdown
## DL-009 · 2026-08-21 — Instrumento v2 y alineación evaluador↔gateo (remediación pre-lanzamiento BIT-003)

**Problema.** La auditoría pre-lanzamiento del 21-ago (`.red-pill/memory/BIT-003_audit_findings.md`) encontró que
(a) el evaluador, al repuntarse a `childes_pre_school_en.json`, pasó a permitir TODO el vocabulario de CHILDES sin
corte (18.364/20.095 palabras a edad 2, vs 822 del gateo de entrenamiento) — un cambio de instrumento de facto sin
DL; (b) los exámenes (29-jul, pre-rebuild) esperaban respuestas fuera del vocabulario nuevo (aleth, bunker, madrid);
(c) el trainer no implementaba la semántica de DL-008 (censo por ids, NameError de arranque, device mismatch,
E7 con `<unk>`, 'xxx' producible en E0, contracciones partidas don't/dont, 7/28 glifos de referencia sin canónico).

**Decisión (operador, 21-ago).**
1. **Instrumento v2**: exámenes y `AGE_QUESTIONS` reescritos SOLO con palabras del vocabulario (sin nombres propios
   de lore: el control v1-inglés mide adquisición de lenguaje, no memorización de tokens sin presencia en corpus) y
   ampliados a 10 preguntas por edad. Toda respuesta de examen entra al gateo de su etapa (fix S5).
2. **Evaluador alineado**: el trainer persiste `stage_gate_masks.json` y Samantha restringe la generación a la
   máscara REAL de la etapa (fallback legado para checkpoints antiguos). `<unk>` vetado también en evaluación.
3. **Pipeline de corpus**: apóstrofos normalizados en la tokenización canónica (don't→dont, como el corpus CHILDES-en
   upstream), artefactos CHAT (xxx/yyy/www) eliminados del corpus, glifos canónicos EN preservados (28/28).
4. `samples_per_epoch=400.000` por defecto (análisis Chinchilla 21-ago, confirmado).

**Instrumento.** Este DL VERSIONA el instrumento de evaluación de v1-inglés (sucesor del congelado en DL-004, que
sigue intacto para los runs v1-ES históricos). Los dos brazos BIT-003 (glyph/standard) usan este instrumento v2
idéntico; la comparativa entre brazos no se ve afectada.

**Evidencia.** `scripts/audit_stage_gating.py` (rc=0) — tamaños de gateo y distribución del corpus por etapa se
copian de su salida tras cada regeneración de corpus.

**Referencias.** `.red-pill/memory/BIT-003_audit_findings.md`, `.red-pill/memory/BIT-003_fix_plan.md`, RFC_GATING_VOCABULARIO_ETAPAS.md.
```

2. **`docs/RFC_GATING_VOCABULARIO_ETAPAS.md`**:
   - En §3.1, tabla de tamaños y en §5.2, tablas de distribución: sustituir los números por los que imprima `scripts/audit_stage_gating.py` tras F1-F6, y añadir bajo cada tabla: `_Medido por scripts/audit_stage_gating.py (2026-08-21, post-remediación DL-009)._`
   - En §3.1, añadir a la composición del gateo: `∪ RespuestasExamen(E{i})` (fix S5, DL-009).
   - En §6 (tabla de implementación), añadir fila: `scripts/audit_stage_gating.py | auditoría permanente del gateo (acceptance)`.
   - En §7, añadir: `5. **Instrumento**: el evaluador usa la máscara real de la etapa persistida por el trainer (DL-009); el gateo legado por edad queda como fallback para checkpoints antiguos.`

3. **`.red-pill/memory/BIT-003_gating_state.md`** — reescribir la sección `## Pendiente` a:
```markdown
## Pendiente
- Remediación DL-009 aplicada (ver BIT-003_fix_plan.md): ejecutar smokes F8 y lanzar ambos brazos.
- samples_per_epoch confirmado: 400k (default del trainer desde F4.7).
```
y añadir a `## Implementado`: `- Remediación 21-ago (B1-B5, S1-S6): ver .red-pill/memory/BIT-003_fix_plan.md y DL-009.`

**Commit F7**: `docs: BIT-003 F7 — DL-009, RFC de gateo actualizado, estado del workspace`

---

## F8 · Smokes y runbook de lanzamiento

### F8.1 Smoke CPU (rápido, valida el pipeline de datos entero)

```bash
cd /home/joan/Documents/IA/frankenswarm
systemd-run --user --scope -p MemoryMax=16G .venv/bin/python -m src.bitnet.training.train_sovereign_school \
  --adaptive --embedding glyph --amp off \
  --samples_per_epoch 2000 --max_epochs_per_run 1 --batch_size 32 \
  --state_dir storage/checkpoints/smoke_glyph_cpu --reset_state --test_mock
```
**Verificar en la salida** (abortar el plan si falla algo):
- Arranca sin `NameError` (B1).
- `🚧 [GATEO] Palabras producibles por etapa:` con valores CRECIENTES y E0 ≥ 200 (B2).
- `🚧 [CLASIFICACIÓN] Corpus general por etapa:` con E0-E2 dominando (no 99% en E7).
- `🔒 [GATEO] Máscaras por etapa persistidas...` (F5.1).
- Completa 1 época con Loss finito (ni NaN ni inf) y guarda checkpoint.

### F8.2 Smoke GPU (valida B3 — el device mismatch NO se ve en CPU)

```bash
systemd-run --user --scope -p MemoryMax=16G .venv/bin/python -m src.bitnet.training.train_sovereign_school \
  --adaptive --embedding glyph --amp bf16 --require_gpu \
  --samples_per_epoch 2000 --max_epochs_per_run 1 --batch_size 32 \
  --state_dir storage/checkpoints/smoke_glyph_gpu --reset_state --test_mock
```
Verificar: 1 época completa en CUDA sin `RuntimeError` de índices/device y con `∇STE` > 0. Repetir una vez con `--embedding standard --state_dir storage/checkpoints/smoke_std_gpu` (mismo criterio).

Tras los smokes: `rm -rf storage/checkpoints/smoke_*` y borrar la caché tokenizada del smoke SOLO si molesta el disco (`storage/datasets/tokenized_corpus.json` del subset 100k se regenerará distinto para el run full por el hash).

**Commit F8** (si hubo ajustes durante los smokes): `fix(train): BIT-003 F8 — ajustes de smoke`

### F8.3 Runbook de lanzamiento real (para el operador; el coder NO lo ejecuta)

Un run por brazo, `--state_dir` separados (el estado valida `embedding_mode`). `systemd-inhibit` es OBLIGATORIO (las suspensiones del portátil matan CUDA — causa raíz de los fritos de julio). Sin `--opt8bit` (mapeo de estados bajo neurogénesis sin verificar). `--wd_mode no_embed` (recomendación DL-007 para comparativas nuevas, ambos brazos igual).

```bash
# Brazo K-65P (glifos)
systemd-inhibit --what=sleep:idle --why="BIT-003 glyph arm" \
systemd-run --user --scope -p MemoryMax=16G \
nohup .venv/bin/python -m src.bitnet.training.train_sovereign_school \
  --adaptive --embedding glyph --amp bf16 --compile --require_gpu \
  --full_tinystories --samples_per_epoch 400000 --wd_mode no_embed --seed 42 \
  --state_dir storage/checkpoints/bit003_glyph \
  > storage/checkpoints/bit003_glyph.log 2>&1 &
```
```bash
# Brazo estándar (control) — al terminar el glyph, o en serie por VRAM
systemd-inhibit --what=sleep:idle --why="BIT-003 standard arm" \
systemd-run --user --scope -p MemoryMax=16G \
nohup .venv/bin/python -m src.bitnet.training.train_sovereign_school \
  --adaptive --embedding standard --amp bf16 --compile --require_gpu \
  --full_tinystories --samples_per_epoch 400000 --wd_mode no_embed --seed 42 \
  --state_dir storage/checkpoints/bit003_standard \
  > storage/checkpoints/bit003_standard.log 2>&1 &
```
Notas: la primera tokenización full (~2.1M historias + CHILDES) tarda y genera un `tokenized_corpus.json` de varios GB — vigilar disco. Un examen suspendido sin techo de dim pausa con rc=78 (compatible con el job manager de red-pill si se encola como job).

---

## Criterios de aceptación globales

1. `scripts/audit_stage_gating.py` → rc=0 con TODOS los checks ✓.
2. `scripts/validate_exam_vocab.py` → rc=0.
3. `scripts/rebuild_curriculum_en.py` → "0 palabras OOV" y preschool=32.
4. Smoke CPU y smoke GPU (ambos brazos) completan 1 época: gateo creciente, clasificación sana, sin NaN, checkpoint escrito.
5. `git log` en `feat/v1-english-rebuild` con los commits F1-F7 (F8 si aplica). Sin push.
6. Ningún archivo de la lista "NO tocar" modificado (`git status` lo confirma).
