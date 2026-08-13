"""Audit TinyStories vocabulary → glyph mapping.

Census script that:
1. Extracts all unique words from TinyStories
2. Projects each to a 65-trit glyph via fastembed + Ridge
3. Deduplicates glyphs (counts how many words collapse to same concept)
4. Flags potentially inappropriate words (violence, sexual, etc.)
5. Produces a clean glyph table for School v3

Usage:
    PYTHONPATH=. .venv/bin/python scripts/audit_tinystories_glyphs.py [--max-stories N]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime

import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Words that should NEVER enter a child's vocabulary
BLACKLIST_PATTERNS = [
    # Violence
    r"\b(kill|murder|slaughter|assassin|massacre|torture|rape|abuse)\b",
    # Sexual
    r"\b(sex|sexual|porn|nude|naked|erotic|orgasm|penis|vagina|breast)\b",
    # Drugs/alcohol (adult context)
    r"\b(cocaine|heroin|meth|marijuana|drug|drunk|alcohol|beer|wine|whiskey|vodka|cigarette)\b",
    # Profanity
    r"\b(fuck|shit|damn|ass|bitch|bastard|crap|hell|dick|cock|pussy|slut|whore)\b",
    # Political/controversial figures (not concepts a child needs)
    r"\b(hitler|nazi|stalin|terrorist|terrorism|isis|bomb|weapon|gun|rifle|pistol|bullet)\b",
]

# Compile once
BLACKLIST_RE = re.compile("|".join(BLACKLIST_PATTERNS), re.IGNORECASE)

# English stopwords — functional words with minimal semantic content
ENGLISH_STOPWORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what",
    "and", "but", "or", "nor", "not", "no",
    "if", "then", "else", "when", "where", "how", "why",
    "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "up", "out", "off", "over", "under", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "against", "just", "also", "very", "too", "so",
    "as", "than", "both", "each", "all", "any", "few", "more",
    "most", "other", "some", "such", "only", "own", "same",
    "s", "t", "d", "ll", "re", "ve", "m",  # contractions
])


def extract_words(text):
    """Extract lowercase alphabetic words from text."""
    return re.findall(r"[a-zA-Z]{2,}", text.lower())


def build_ridge_projector():
    """Build the same Ridge projector used by expand_vocabulary.py."""
    from fastembed import TextEmbedding

    from src.bitnet.vocab.expand_vocabulary import VOCAB_MAP
    from src.bitnet.vocab.glyph_vocabulary import VOCABULARY

    model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    ref_keys = list(VOCABULARY.keys())
    ref_natural = [VOCAB_MAP.get(k, k) for k in ref_keys]
    ref_embs = np.array(list(model.embed(ref_natural)), dtype=np.float32)
    ref_glyphs = np.stack([VOCABULARY[k] for k in ref_keys], axis=0).astype(np.float32)

    E = ref_embs
    Y = ref_glyphs
    alpha = 1.0
    A = np.linalg.solve(E @ E.T + alpha * np.eye(len(ref_keys)), Y)

    # Calibrate theta
    preds = (E @ E.T) @ A
    best_theta, best_acc = 0.25, 0
    for theta in np.arange(0.05, 0.6, 0.01):
        p = np.zeros_like(preds, dtype=np.int8)
        p[preds > theta] = 1
        p[preds < -theta] = -1
        acc = np.mean(p == Y)
        if acc > best_acc:
            best_acc = acc
            best_theta = theta

    print(f"Ridge projector ready. theta={best_theta:.3f}, reconstruction accuracy={best_acc*100:.1f}%")
    return model, E, A, best_theta


def project_words_to_glyphs(words, model, E, A, theta, batch_size=512):
    """Project a list of words to 65-trit glyph vectors."""
    all_glyphs = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        embs = np.array(list(model.embed(batch)), dtype=np.float32)
        cont = (embs @ E.T) @ A
        g = np.zeros_like(cont, dtype=np.int8)
        g[cont > theta] = 1
        g[cont < -theta] = -1
        all_glyphs.append(g)
        if (i + batch_size) % 5000 < batch_size:
            print(f"  Projected {min(i + batch_size, len(words)):,}/{len(words):,} words...")
    return np.vstack(all_glyphs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stories", type=int, default=None, help="Limit stories to process (for testing)")
    args = ap.parse_args()

    # 1. Load TinyStories
    print("Loading TinyStories...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)
    n_stories = len(ds) if args.max_stories is None else min(args.max_stories, len(ds))
    print(f"Processing {n_stories:,} of {len(ds):,} stories")

    # 2. Extract vocabulary with frequency counts
    print("\nExtracting vocabulary...")
    word_counts = Counter()
    flagged_stories = 0
    for i in range(n_stories):
        text = ds[i]["text"]
        words = extract_words(text)
        word_counts.update(words)
        if BLACKLIST_RE.search(text):
            flagged_stories += 1
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i+1:,} stories, {len(word_counts):,} unique words so far...")

    total_tokens = sum(word_counts.values())
    print("\nVocabulary census:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Unique words: {len(word_counts):,}")
    print(f"  Stories with flagged content: {flagged_stories:,} ({flagged_stories/n_stories*100:.2f}%)")

    # 3. Filter
    content_words = {}
    stopwords_removed = 0
    blacklisted = []
    for word, count in word_counts.items():
        if word in ENGLISH_STOPWORDS:
            stopwords_removed += 1
            continue
        if BLACKLIST_RE.match(word):
            blacklisted.append((word, count))
            continue
        content_words[word] = count

    print("\nFiltering:")
    print(f"  Stopwords removed: {stopwords_removed}")
    print(f"  Blacklisted words: {len(blacklisted)}")
    if blacklisted:
        print(f"  Blacklisted samples: {blacklisted[:20]}")
    print(f"  Content words remaining: {len(content_words):,}")

    # 4. Project to glyphs
    print("\nProjecting to glyphs...")
    word_list = sorted(content_words.keys())
    model, E, A, theta = build_ridge_projector()
    glyphs = project_words_to_glyphs(word_list, model, E, A, theta)

    # 5. Deduplicate
    glyph_to_words = {}
    for i, word in enumerate(word_list):
        g_tuple = tuple(glyphs[i].tolist())
        if g_tuple not in glyph_to_words:
            glyph_to_words[g_tuple] = []
        glyph_to_words[g_tuple].append((word, content_words[word]))

    unique_glyphs = len(glyph_to_words)
    compression = len(word_list) / unique_glyphs

    print("\nGlyph census:")
    print(f"  Content words: {len(word_list):,}")
    print(f"  Unique glyphs: {unique_glyphs:,}")
    print(f"  Compression ratio: {compression:.2f}x")

    # 6. Show largest collision groups (words sharing same glyph)
    collisions = sorted(glyph_to_words.values(), key=lambda ws: len(ws), reverse=True)
    print("\nTop 20 glyph collision groups (synonyms → same concept):")
    for group in collisions[:20]:
        words_str = ", ".join(f"{w}({c})" for w, c in sorted(group, key=lambda x: -x[1])[:8])
        if len(group) > 8:
            words_str += f", ... (+{len(group)-8} more)"
        print(f"  [{len(group):3d} words] {words_str}")

    # 7. Frequency distribution of glyphs
    glyph_freqs = []
    for _g, words in glyph_to_words.items():
        total = sum(c for _, c in words)
        glyph_freqs.append(total)
    glyph_freqs.sort(reverse=True)

    print("\nGlyph frequency distribution:")
    print(f"  Top 10 most frequent glyphs cover: {sum(glyph_freqs[:10])/total_tokens*100:.1f}% of tokens")
    print(f"  Top 100: {sum(glyph_freqs[:100])/total_tokens*100:.1f}%")
    print(f"  Top 1000: {sum(glyph_freqs[:1000])/total_tokens*100:.1f}%")
    print(f"  Glyphs appearing >100 times: {sum(1 for f in glyph_freqs if f > 100):,}")
    print(f"  Glyphs appearing >1000 times: {sum(1 for f in glyph_freqs if f > 1000):,}")

    # 8. Save results
    out_dir = os.path.join(base_dir, "configs")
    audit_path = os.path.join(out_dir, "tinystories_glyph_audit.json")
    audit_data = {
        "metadata": {
            "generated": datetime.now(UTC).isoformat(),
            "source": "roneneldan/TinyStories",
            "stories_processed": n_stories,
            "total_tokens": total_tokens,
            "unique_words": len(word_counts),
            "content_words": len(content_words),
            "unique_glyphs": unique_glyphs,
            "compression_ratio": round(compression, 2),
            "theta": float(theta),
            "stopwords_removed": stopwords_removed,
            "blacklisted_count": len(blacklisted),
            "flagged_stories": flagged_stories,
        },
        "blacklisted": [(w, c) for w, c in blacklisted],
        "top_collisions": [
            {"words": [w for w, _ in group], "total_freq": sum(c for _, c in group)}
            for group in collisions[:50]
        ],
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"\nAudit saved to: {audit_path}")


if __name__ == "__main__":
    main()
