"""Extractor de vocabulario semántico — palabras por etapa de v1 con sus glifos.

Parte del currículo estructurado de v1 (school_curriculum_structured_en.json,
ya anotado con category/intent/primes) y cruza cada palabra con el diccionario
real (expanded_glyphs.json: 12.143 palabras + glifos ternarios Ridge).

Produce, por etapa: el vocabulario significativo con su glifo, las palabras
OOV (spanglish/ruido sin glifo), y los intents/primes anotados.

Doctrina (RULE 8 de CONVENTIONS.md): el vocabulario del corpus semántico sale
del diccionario de v1, no de moléculas inventadas.

Uso: python scripts/extract_semantic_vocabulary.py
"""

import json, re, sys
from collections import Counter
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]

CURRICULUM = base_dir / "configs" / "school_curriculum_structured_en.json"
GLYPHS = base_dir / "configs" / "expanded_glyphs.json"
OUT = base_dir / "storage" / "curriculum" / "semantic_vocab.json"

STOPWORDS = set("""the a an and or but if then of to in on at for with by from as is are was
were be been being do does did have has had i you he she it we they me him her them my your
his its our their this that these those not no yes so very more than all some one two there
here what who when where why how can could will would should may might must up down out over
under again now just only also too am are is""".split())


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1]+", text.lower())


def main() -> None:
    sc = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    cur = sc.get("curriculum", {})

    glyphs_data = json.loads(GLYPHS.read_text(encoding="utf-8"))
    words_list = glyphs_data["words"]
    glyphs_list = glyphs_data["glyphs"]
    word_to_glyph = {w: g for w, g in zip(words_list, glyphs_list, strict=False)}
    vocab_set = set(word_to_glyph)

    result = {"meta": {"source_curriculum": CURRICULUM.name, "source_glyphs": GLYPHS.name,
        "theta": glyphs_data.get("theta")}, "stages": {}}

    for stage in ["preschool", "primary", "secondary"]:
        items = cur.get(stage, [])
        freq = Counter()
        intents = Counter()
        primes = Counter()
        for it in items:
            txt = it.get("text", "") if isinstance(it, dict) else str(it)
            freq.update(words_of(txt))
            intents[it.get("intent", "código_intent")] += 1
            for p in it.get("primes", []):
                primes[p] += 1

        vocab = []
        oov = []
        for w, c in freq.most_common():
            if w in STOPWORDS or len(w) < 2:
                continue
            if w in vocab_set:
                vocab.append({"word": w, "freq": c, "glyph": word_to_glyph[w]})
            else:
                oov.append({"word": w, "freq": c})

        result["stages"][stage] = {
            "n_sentences": len(items),
            "vocab": vocab,
            "n_vocab": len(vocab),
            "oov": oov,
            "n_oov": len(oov),
            "intents": dict(intents.most_common()),
            "primes": dict(primes.most_common()),
        }
        print(f"{stage:10s}: {len(vocab):4d} palabras con glifo | {len(oov):4d} OOV | {len(items)} frases | {len(intents)} intents")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVocabulario semántico guardado en {OUT}")


if __name__ == "__main__":
    main()
