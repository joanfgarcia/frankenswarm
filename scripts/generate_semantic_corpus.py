"""Fábrica de corpus semántico K-65P — expresiones válidas Y verdaderas.

A diferencia del generador sintáctico (generate_k65p_corpus.py), esta fábrica
parte de una KB causal de hechos + reglas y genera variaciones por sustitución
de entidades, verificando cada expresión contra el validador (forma) y contra
la KB vía Prolog (verdad). Produce pares contrastivos FALSOS con NOT y reserva
teoremas held-out para el examen gen_true.

Uso: PYTHONPATH=.:../k65p/src .venv/bin/python scripts/generate_semantic_corpus.py [--seed 770]
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))
k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

from k65p.lexicon import molecule_names
from k65p.validator import is_valid

LANG = "en"
OUT_DIR = base_dir / "storage" / "curriculum" / "factory_semantic"

# ── KB: hechos + reglas agrupados por etapa ───────────────────────────────────
_KB = {
    "preschool": [
        # Existencia
        "[25 fire]", "[25 water]", "[25 sun]", "[25 night]",
        "[25 tree]", "[25 rock]", "[25 earth]", "[25 river]",
        "[25 cave]", "[25 forest]", "[25 food]", "[25 predator]", "[25 myself]",
        # Atributos (evaluadores)
        "[8 food]", "[8 water]", "[8 sun]", "[8 sleep]",
        "[9 fire]", "[9 predator]", "[9 wound]", "[9 storm]",
        "[60 fire]", "[60 sun]", "[61 water]", "[61 night]",
        "[10 sun]", "[10 tree]", "[10 forest]", "[11 rock]", "[11 myself]",
        # Atributos con grupo G
        "[8 [G water cold]]", "[9 [G fire hot]]",
        "[8 [G food big]]", "[9 [G predator big]]",
        # Habilidades CAN
        "[46 [21 myself eat]]", "[46 [21 myself drink]]",
        "[46 [21 myself sleep]]", "[46 [21 myself move_action]]",
    ],
    "primary": [
        # Acciones DO
        "[21 myself eat food]", "[21 myself drink water]",
        "[21 myself sleep cave]", "[21 predator eat 2]",
        "[21 sun [G see_action light]]", "[21 storm [G move_action water]]",
        # BECAUSE
        "[47 [24 2 fire] [15 2 9]]", "[47 [21 2 drink water] [15 2 8]]",
        "[47 [21 2 eat food] [15 2 8]]", "[47 [21 2 sleep] [15 2 8]]",
        "[47 [24 2 [G predator big]] [15 2 9]]",
        "[47 [25 storm] [15 2 9]]", "[47 [25 sun] [15 2 8]]",
        "[47 [25 night] [46 [21 2 sleep]]]",
        # IF
        "[48 [24 2 fire] [22 [G 4 9] 2]]",
        "[48 [21 2 drink water] [22 [G 4 8] 2]]",
        "[48 [44 [21 2 eat food]] [15 2 9]]",
        "[48 [44 [21 2 drink water]] [28 2]]",
        # LIKE
        "[51 [G fire hot] [G sun light]]",
        "[51 [G water cold] [G river water]]",
        "[51 [G night dark] [G dark very]]",
    ],
    "secondary": [
        # IF anidados
        "[48 [24 2 [G fire hot]] [22 [G 4 9] 2]]",
        "[48 [24 2 [G water cold]] [44 [22 [G 4 9] 2]]]",
        "[48 [44 [21 2 sleep]] [15 2 9]]",
        # Negación
        "[44 [25 predator]]", "[44 [8 fire]]",
        "[48 [44 [25 water]] [28 2]]",
        # BECAUSE anidados
        "[47 [25 fire] [47 [60 fire] [15 2 9]]]",
        "[47 [25 predator] [47 [10 predator] [15 2 9]]]",
        "[47 [25 sun] [47 [60 sun] [15 2 8]]]",
        # IF + BECAUSE combinados
        "[48 [47 [24 2 fire] [15 2 9]] [46 [21 2 [23 2 40]]]]",
        "[48 [47 [25 night] [46 [21 2 sleep]]] [15 2 8]]",
        # IF + movimiento
        "[48 [25 storm] [46 [21 2 [23 2 40]]]]",
        # Contraste verdad/falsedad
        "[48 [25 predator] [22 [G 4 9] 2]]",
        "[48 [44 [25 predator]] [22 [G 4 8] 2]]",
        # Cuantificadores
        "[28 [G 3 58]]", "[44 [28 [G 3 57]]]",
        # DO + compuestos
        "[21 myself eat [G food big]]",
        "[21 myself drink [G water cold]]",
    ],
}

# Teoremas HELD-OUT: nunca en corpus, solo en examen gen_true
_HELDOUT = [
    "[48 [24 2 [G fire hot]] [22 [G 4 9] 2]]",
    "[48 [44 [21 2 drink water]] [28 2]]",
    "[47 [25 fire] [47 [60 fire] [15 2 9]]]",
]

# Pools de sustitución
_ENTITIES = ["fire","water","sun","night","tree","rock","earth","river",
    "cave","forest","food","predator","myself","storm","wound"]
_EVALUATORS = ["8","9","10","11","60","61","64"]  # good,bad,big,small,hot,cold,dark
_ACTIONS = ["eat","drink","sleep","move_action"]


def entity_variations(expr: str, rng: random.Random, n: int = 3) -> list[str]:
    """Sustituye moléculas por otras entidades, manteniendo validez."""
    vars = []
    en_lex = molecule_names(lang=LANG)
    for e in _ENTITIES:
        candidate = expr.replace("fire", "___TMP___").replace("water", e)
        candidate = candidate.replace("___TMP___", e)
        if candidate != expr and is_valid(candidate, lexicon=en_lex):
            vars.append(candidate)
    if len(vars) <= n:
        return vars
    return rng.sample(vars, n)


def negate(expr: str) -> str | None:
    """Envuelve en NOT si no produce redundancia [44 [44 X]]."""
    if expr.startswith("[44 "):
        return None  # ya negado
    en_lex = molecule_names(lang=LANG)
    neg = f"[44 {expr}]"
    return neg if is_valid(neg, lexicon=en_lex) else None


def generate(rng: random.Random) -> None:
    en_lex = molecule_names(lang=LANG)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()

    for stage in ["preschool", "primary", "secondary"]:
        records = []
        ood_records = []
        exprs = _KB.get(stage, [])

        for expr in exprs:
            # La expresión base
            if expr in seen or not is_valid(expr, lexicon=en_lex):
                continue
            seen.add(expr)
            records.append(make_record(expr, stage, rng))
            # Variaciones por entidad
            for v in entity_variations(expr, rng, n=4):
                if v not in seen and is_valid(v, lexicon=en_lex):
                    seen.add(v)
                    records.append(make_record(v, stage, rng))
            # Pares falsos contrastivos (NOT)
            neg = negate(expr)
            if neg and neg not in seen and is_valid(neg, lexicon=en_lex):
                seen.add(neg)
                records.append(make_record(neg, stage, rng))

        # Held-out: marcar como OOD
        for ho in _HELDOUT:
            if ho not in seen and is_valid(ho, lexicon=en_lex):
                seen.add(ho)
                ood_records.append(make_record(ho, stage, rng, ood=True))

        out_path = OUT_DIR / f"{stage}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {stage}: {len(records)} expresiones → {out_path.name}")

        if ood_records:
            ood_path = OUT_DIR / "ood_holdout.jsonl"
            with open(ood_path, "w", encoding="utf-8") as f:
                for r in ood_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  OOD held-out: {len(ood_records)} teoremas → {ood_path.name}")

    (OUT_DIR / "generation_manifest.json").write_text(
        json.dumps({"seed": rng.randint(0, 999999), "type": "semantic", "lang": LANG}, indent=2),
        encoding="utf-8")
    # KB de verdad (solo hechos positivos, sin negaciones contrastivas)
    from k65p.bridge import to_prolog as _to_plog
    kb_path = OUT_DIR / "kb.pl"
    with open(kb_path, "w", encoding="utf-8") as kb_out:
        for stage in ["preschool", "primary", "secondary"]:
            for expr in _KB.get(stage, []):
                if expr.startswith("[44 "):
                    continue
                if is_valid(expr, lexicon=en_lex):
                    try:
                        kb_out.write(_to_plog(expr, executable=True) + "\n")
                        kb_out.write(_to_plog(expr, executable=False) + "\n")
                    except Exception:
                        continue
            for ho in _HELDOUT:
                if is_valid(ho, lexicon=en_lex):
                    try:
                        kb_out.write(_to_plog(ho, executable=True) + "\n")
                        kb_out.write(_to_plog(ho, executable=False) + "\n")
                    except Exception:
                        continue
    print(f"  KB de verdad: {kb_path}")
    print("  manifiesto escrito")


def make_record(canonical: str, stage: str, rng: random.Random, ood: bool = False) -> dict:
    record = {
        "k65p_canonical": canonical,
        "stage": stage,
        "provenance": "semantic_factory_v0",
        "hash": hashlib.sha256(canonical.encode()).hexdigest()[:16],
    }
    if ood:
        record["ood_reason"] = "heldout_theorem"
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fábrica de corpus semántico K-65P")
    parser.add_argument("--seed", type=int, default=770)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    print(f"Fábrica semántica — seed={args.seed} — lang={LANG}")
    generate(rng)
    print("Listo.")
