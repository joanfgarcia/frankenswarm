import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime

base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(base_dir)

CONTROL_TOKENS = ["<pad>", "<unk>"]
DIALOGUE_TOKENS = ["me", "you"] # English equivalent of "yo", "tú"

# Safety Blacklist
BLACKLIST_PATTERNS = [
    r"\b(kill|murder|slaughter|assassin|massacre|torture|rape|abuse)\b",
    r"\b(sex|sexual|porn|nude|naked|erotic|orgasm|penis|vagina|breast)\b",
    r"\b(cocaine|heroin|meth|marijuana|drug|drunk|alcohol|beer|wine|whiskey|vodka|cigarette)\b",
    r"\b(fuck|shit|damn|ass|bitch|bastard|crap|hell|dick|cock|pussy|slut|whore)\b",
    r"\b(hitler|nazi|stalin|terrorist|terrorism|isis|bomb|weapon|gun|rifle|pistol|bullet)\b",
]
BLACKLIST_RE = re.compile("|".join(BLACKLIST_PATTERNS), re.IGNORECASE)

def words_of(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z\-]+", str(text).lower())

def main():
    freq = Counter()
    print("── ENGLISH CENSUS FOR TINYSTORIES + DIALOGUES ──")
    
    # 1. Load English Dialogues
    dialogues_path = os.path.join(base_dir, "configs/tiny_dialogues_large_en.json")
    if os.path.exists(dialogues_path):
        print(f"Reading {dialogues_path}...")
        with open(dialogues_path, encoding="utf-8") as f:
            dialogues = json.load(f)
            for d in dialogues:
                for turn in d:
                    # Strip speaker prefix "you: " or "me: "
                    turn_clean = re.sub(r"^(you|me)\s*:\s*", "", turn, flags=re.IGNORECASE)
                    freq.update(words_of(turn_clean))
        print(f"  Dialogues total tokens so far: {sum(freq.values()):,}")

    # 2. Load TinyStories from HF Cache
    print("Loading TinyStories from local cache...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)
    
    # Process first 100,000 stories
    n_stories = min(100000, len(ds))
    print(f"Processing {n_stories:,} stories for vocabulary...")
    for i in range(n_stories):
        text = ds[i]["text"]
        # Skip stories with blacklisted content
        if BLACKLIST_RE.search(text):
            continue
        freq.update(words_of(text))
        if (i+1) % 25000 == 0:
            print(f"  Processed {i+1:,} stories...")
            
    print(f"Total tokens processed: {sum(freq.values()):,}")
    print(f"Unique words before filtering: {len(freq):,}")

    # 3. Filter
    min_count = 3
    clean_words = []
    for w, count in freq.items():
        if count < min_count:
            continue
        if BLACKLIST_RE.match(w):
            continue
        if w in DIALOGUE_TOKENS or w in CONTROL_TOKENS:
            continue
        clean_words.append(w)
        
    vocab = CONTROL_TOKENS + DIALOGUE_TOKENS + sorted(clean_words)
    print(f"Unique words after filtering (freq >= {min_count}, safe): {len(vocab):,}")

    # 4. Save to clean_vocabulary_words.json
    out_path = os.path.join(base_dir, "configs/clean_vocabulary_words.json")
    out = {
        "metadata": {
            "generated": datetime.now(UTC).astimezone().isoformat(),
            "sources": ["tiny_dialogues_large_en.json", f"TinyStories train (first {n_stories})"],
            "min_count": min_count,
            "note": "English clean vocabulary for School v3. Maps to expanded_glyphs.json.",
        },
        "words": vocab,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Written {out_path} ({len(vocab):,} words)")

if __name__ == "__main__":
    main()
