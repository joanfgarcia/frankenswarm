"""Auditoría del gateo BIT-003: máscaras, censo, clasificación e instrumentos.
Usa las MISMAS funciones que el trainer. rc=0 solo si todos los checks pasan.
Sus tablas (tamaños de gateo, distribución por etapa) son la fuente de la
documentación (RFC_GATING_VOCABULARIO_ETAPAS, DL-008/DL-009).
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
from src.bitnet.training.modules.tokenization import format_and_tokenize_dialogue, tokenize, words_of  # noqa: E402
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
check(len({tuple(g) for g in glyphs}) == len(words), "glifos únicos por palabra")
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
check(not any(set(s.split()) & {"xxx", "yyy", "www"} for s in childes), "sin 'xxx/yyy/www' en el corpus")
coverage = 1 - n_unk / max(1, n_tok)
check(coverage >= 0.99, f"cobertura vocab >=99% (real {coverage:.4%})")

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
print(f"  tamaños de gateo por etapa: {sizes}")
check(all(sizes[i] < sizes[i + 1] for i in range(7)), "tamaños estrictamente crecientes E0→E7")
check(sizes[0] >= 200, f"E0 >= 200 palabras (real {sizes[0]}) — el top-N desbloquea de verdad (B2)")
check(all(m[1].item() == float("-inf") for m in masks), "<unk> vetado en las 8 etapas (S3)")
check(all(m[0].item() == 0.0 for m in masks), "<pad> permitido en las 8 etapas")
xxx_ok = "xxx" not in w2i or all(m[w2i["xxx"]].item() != 0.0 for m in masks[:7])
check(xxx_ok, "'xxx' no producible antes de E7 (B5)")

print("── 4. Clasificación del corpus ──")
tms = token_min_stage(masks, vocab_size)
groups = classify_sequences_by_gate(seqs, tms)
dist = [len(g) / max(1, len(seqs)) for g in groups]
print("  CHILDES por etapa: " + ", ".join(f"E{i}:{p:.1%}" for i, p in enumerate(dist)))
check(sum(dist[:3]) >= 0.60, f"E0-E2 >=60% de CHILDES (real {sum(dist[:3]):.0%})")
check(dist[7] <= 0.15, f"E7 <=15% de CHILDES (real {dist[7]:.1%})")

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
dlens = [len(format_and_tokenize_dialogue(d, w2i)) for d in dial]
check(max(dlens) <= 64, f"diálogos <=64 tokens (max {max(dlens)}) — nada se descarta en bucketing")
clens = [len(words_of(t)) for k in curriculum_data for t in curriculum_data[k]]
check(max(clens) <= 64, f"currículo <=64 tokens (max {max(clens)})")

print()
if fails:
	print(f"❌ AUDITORÍA: {len(fails)} checks fallidos.")
	raise SystemExit(1)
print("✅ AUDITORÍA COMPLETA: todos los checks OK. Copiar tamaños y distribución a RFC/DL.")
