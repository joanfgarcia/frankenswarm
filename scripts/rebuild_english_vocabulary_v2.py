"""Censo de vocabulario EN desde CERO para BIT-003 (corpus limpio de v1).

Fuentes SOLO inglés real descargado de cero:
- configs/childes_pre_school_en.json   (CHILDES-en, 1.45M oraciones únicas)
- configs/tiny_dialogues_large_en.json (diálogos EN)
- TinyStories train (100k)             (HF cache)

Reglas:
- Sin mapeo OOV (RULE 7): cada palabra del censo es un token real del corpus.
- El regex de extracción coincide EXACTAMENTE con la tokenización del trainer
  (src/bitnet/training/modules/tokenization.py: tokenize), incluidos apóstrofes.
- Blacklist de seguridad heredada del censo v1.

Salida: configs/clean_vocabulary_words.json (entrada de expand_vocabulary.py).
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

CONTROL_TOKENS = ["<pad>", "<unk>"]
DIALOGUE_TOKENS = ["me", "you"]

# Blacklist de seguridad (heredada de rebuild_english_vocabulary.py).
BLACKLIST_PATTERNS = [
    r"\b(kill|murder|slaughter|assassin|massacre|torture|rape|abuse)\b",
    r"\b(sex|sexual|porn|nude|naked|erotic|orgasm|penis|vagina|breast)\b",
    r"\b(cocaine|heroin|meth|marijuana|drug|drunk|alcohol|beer|wine|whiskey|vodka|cigarette)\b",
    r"\b(fuck|shit|damn|ass|bitch|bastard|crap|hell|dick|cock|pussy|slut|whore)\b",
    r"\b(hitler|nazi|stalin|terrorist|terrorism|isis|bomb|weapon|gun|rifle|pistol|bullet)\b",
]
BLACKLIST_RE = re.compile("|".join(BLACKLIST_PATTERNS), re.IGNORECASE)

# Tokenización canónica importada del trainer: una sola fuente de verdad
# (apóstrofos eliminados, guiones conservados — BIT-003 S2).
from src.bitnet.training.modules.tokenization import words_of  # noqa: E402


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Censo de vocabulario EN desde cero (BIT-003)")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--n-stories", type=int, default=100000)
    args = parser.parse_args()

    freq: Counter = Counter()
    print("── CENSO EN DESDE CERO (BIT-003) ──")

    # 1. CHILDES-en
    childes_path = os.path.join(base_dir, "configs", "childes_pre_school_en.json")
    childes = load_json(childes_path)
    n_tokens = 0
    for s in childes:
        freq.update(words_of(s))
        n_tokens += 1
    print(f"  CHILDES-en: {n_tokens:,} oraciones | tokens únicos: {len(freq):,}")

    # 2. Diálogos EN
    dialogues_path = os.path.join(base_dir, "configs", "tiny_dialogues_large_en.json")
    dialogues = load_json(dialogues_path)
    for group in dialogues:
        for turn in group:
            turn_clean = re.sub(r"^(you|me)\s*:\s*", "", turn, flags=re.IGNORECASE)
            freq.update(words_of(turn_clean))
    print(f"  Diálogos EN: {len(dialogues):,} grupos | tokens únicos acumulados: {len(freq):,}")

    # 3. TinyStories
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)
    n_stories = min(args.n_stories, len(ds))
    skipped_blacklist = 0
    for i in range(n_stories):
        text = ds[i]["text"]
        if BLACKLIST_RE.search(text):
            skipped_blacklist += 1
            continue
        freq.update(words_of(text))
        if (i + 1) % 25000 == 0:
            print(f"  TinyStories: {i + 1:,} historias procesadas...")
    print(f"  TinyStories: {n_stories:,} historias (om. {skipped_blacklist:,} por blacklist)")

    # 4. Filtrar
    min_count = args.min_count
    clean_words = []
    for w, c in freq.items():
        if c < min_count:
            continue
        if BLACKLIST_RE.search(w):
            continue
        if w in CONTROL_TOKENS or w in DIALOGUE_TOKENS:
            continue
        clean_words.append(w)

    vocab = CONTROL_TOKENS + DIALOGUE_TOKENS + sorted(clean_words)
    print(f"\n  Palabras únicas (freq >= {min_count}): {len(freq):,}")
    print(f"  Vocabulario final: {len(vocab):,} tokens")

    out_path = os.path.join(base_dir, "configs", "clean_vocabulary_words.json")
    out = {
        "metadata": {
            "generated": datetime.now(UTC).astimezone().isoformat(),
            "sources": [
                "configs/childes_pre_school_en.json",
                "configs/tiny_dialogues_large_en.json",
                f"TinyStories train (first {n_stories})",
            ],
            "min_count": min_count,
            "note": "Censo BIT-003 desde cero: CHILDES-en + diálogos EN + TinyStories. Sin mapeo OOV (RULE 7). Regex alineado con tokenization.py.",
        },
        "words": vocab,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ Escrito {out_path} ({len(vocab):,} palabras)")
    print("\nSIGUIENTE PASO:")
    print("  Apuntar FINAL_VOCAB en src/bitnet/expand_vocabulary.py a configs/clean_vocabulary_words.json")
    print("  y ejecutar calibrate_and_project() para derivar expanded_glyphs.json")


if __name__ == "__main__":
    main()