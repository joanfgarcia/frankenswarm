"""Reconstrucción determinista del currículo EN (BIT-003).

El currículo ES (school_curriculum_structured.json) quedó contaminado por el
mapeo OOV ciego (RULE 7). NO se traduce su texto. En su lugar, los items con
intent SISTEMÁTICO se reconstruyen desde la semántica del intent (category/
intent/primes) con plantillas EN canónicas. Garantías:

- Inglés 100% limpio: cada palabra se valida contra clean_vocabulary_words.json.
- Aritmética correcta: suma/resta se REGENERAN (los textos originales tenían
  operaciones falsas: "tres menos cinco son uno").
- Determinismo y reproducibilidad (sin LLM, sin traducción, sin OOV).

La prosa libre contaminada (código_intent, causa_efecto, literatura... ~89% del
curriculum viejo) SE DESCARTA: el lenguaje natural real lo aportan las fuentes
de corpus CHILDES-en + TinyStories + diálogos (todas EN, descargadas de cero).

Salida: configs/school_curriculum_structured_en.json (formato original).
"""

import hashlib
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from src.bitnet.vocab.glyph_vocabulary import SEMANTIC_PRIMES

PRIME_SET = set(SEMANTIC_PRIMES)

NUM_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
NUM_ES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]

# Condiciones de crecimiento/vivencia (fuente de verdad: el NOMBRE del intent).
COND_MAP = {
    "agua": ("water", "agua_prima"),
    "luz": ("light", "luz"),
    "sol": ("sun", None),
    "humedad": ("moisture", None),
    "calor": ("heat", "caliente"),
    "aire": ("air", None),
}

# Aritmética CORRECTA y determinista (semántica del intent, no el texto sucio).
SUMA_PAIRS = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (1, 4), (4, 1), (2, 3),
              (3, 2), (1, 5), (5, 1), (4, 2), (2, 4), (3, 3), (5, 2), (2, 5), (4, 3)]
RESTA_PAIRS = [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (3, 2), (4, 2), (5, 2),
               (4, 3), (5, 3), (6, 3), (5, 4), (6, 4), (7, 5), (9, 3)]

# qué_es_*: pregunta conceptual → (texto EN, primes válidos ES).
QUE_ES = {
    "qué_es_abajo": ("what is below", ["abajo"]),
    "qué_es_afuera": ("what is outside", []),
    "qué_es_antes": ("what is before", ["antes"]),
    "qué_es_aquí": ("what is here", ["aquí"]),
    "qué_es_arriba": ("what is above", ["arriba"]),
    "qué_es_cercano": ("what is near", ["cerca"]),
    "qué_es_dentro": ("what is inside", ["dentro"]),
    "qué_es_después": ("what is after", ["después"]),
    "qué_es_dónde": ("what is where", ["dónde"]),
    "qué_es_falso": ("what is false", ["no", "verdad"]),
    "qué_es_lejos": ("what is far", ["lejos"]),
    "qué_es_morir": ("what is dying", ["morir"]),
    "qué_es_mucho": ("what is a lot", ["mucho"]),
    "qué_es_mucho_tiempo": ("what is a long time", ["mucho_tiempo"]),
    "qué_es_más_grande": ("what is bigger", ["grande", "más"]),
    "qué_es_pasar": ("what is happening", ["pasar"]),
    "qué_es_poco": ("what is little", ["pequeño"]),
    "qué_es_poco_tiempo": ("what is a short time", ["poco_tiempo"]),
    "qué_es_tu_amigo": ("what is your friend", ["tú", "alguien"]),
    "qué_es_un_lado": ("what is a side", ["lado"]),
    "qué_es_un_momento": ("what is a moment", ["momento"]),
    "qué_es_una_hora": ("what is an hour", ["momento"]),
    "qué_es_verdad": ("what is truth", ["verdad"]),
    "qué_es_vivir": ("what is living", ["vivir"]),
}

# pregunta_*: conversación con el niño → texto EN limpio.
PREGUNTA = {
    "pregunta_pasa": "what is happening to you, child",
    "pregunta_alma": "what is happening in your soul, child",
    "pregunta_amigos": "do you have a new friend, child",
    "pregunta_aprendizaje": "you are learning well, child",
    "pregunta_bueno": "you are good, child",
    "pregunta_casa": "what is happening at your home, child",
    "pregunta_cerebro": "what is happening in your brain, child",
    "pregunta_comida": "are you hungry, child? what do you want to eat",
    "pregunta_corazón": "what is happening in your heart, child",
    "pregunta_crecimiento": "you are growing big, child",
    "pregunta_cuerpo": "what is happening in your body, child",
    "pregunta_dibujo": "what a beautiful drawing, child",
    "pregunta_días": "how were your days, child",
    "pregunta_grandeza": "you are big, child",
    "pregunta_lectura": "do you like reading, child? what books do you love",
    "pregunta_mala_estaba": "you were bad, child",
}

SALUDA = [
    "hello baby, how are you? do you want to play today",
    "hello child, how are you? do you want to play with a ball",
    "hello baby, how are you? do you want to play with your friend",
    "hello baby, how are you? do you want to play with toys",
]

# Set preescolar rico BIT-003 (decisión operador 21-ago): conversación y rutinas
# cotidianas con reparto deliberado por MLU para poblar curr_0_1..curr_3_4
# (los saludos de arriba, con 9+ tokens, caían todos en curr_3_4 y dejaban las
# tres primeras particiones vacías). Cada palabra está validada contra el censo
# (la validación RULE 7 de abajo lo garantiza en cada regeneración).
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

PLANT_PARTES = {
    "partes_plantas": ["leaves", "roots", "flowers"],
    "partes_árboles": ["leaves", "roots", "fruits"],
}


def plant_conds(intent: str) -> tuple[str, list[tuple[str, str]]]:
    """Extrae (sujeto EN, [(condición EN, primo ES)]) del nombre del intent.

    Los intents sin condiciones (crecer_árboles, vivencia_plantas...) usan el
    valor por defecto del texto original limpio: agua y luz (crecer) / agua (vivencia).
    """
    plant_token = "plantas" if "plantas" in intent else "árboles"
    subject = "plants" if plant_token == "plantas" else "trees"
    suffix = intent.split(f"{plant_token}_", 1)[1] if f"{plant_token}_" in intent else ""
    conds = [c for c in suffix.split("_") if c in COND_MAP] if suffix else []
    if not conds:
        conds = ["agua", "luz"] if intent.startswith("crecer_") else ["agua"]
    return subject, [(COND_MAP[c][0], COND_MAP[c][1]) for c in conds]


def make_item(text: str, category: str, intent: str, primes: list[str], stage: str) -> dict:
    primes_ok = [p for p in primes if p in PRIME_SET]
    return {"text": text, "category": category, "intent": intent, "primes": primes_ok}


def reconstruct(items_by_intent: dict, stage: str) -> list[dict]:
    out: list[dict] = []
    # Dedup por conjunto de condiciones (orden canónico): las permutaciones
    # (agua_calor_humedad vs agua_humedad_calor) no aportan cognición — solo
    # tokens redundantes que el modelo memoriza como ruido. Una variante canónica.
    emitted_plant_conds: set[tuple] = set()

    for intent, items in items_by_intent.items():
        category = items[0]["category"]

        if intent == "suma":
            for a, b in SUMA_PAIRS:
                s = a + b
                primes = [NUM_ES[a], NUM_ES[b], NUM_ES[s]]
                out.append(make_item(
                    f"{NUM_EN[a]} plus {NUM_EN[b]} is {NUM_EN[s]}",
                    category, intent, primes, stage,
                ))
            continue

        if intent == "resta":
            for a, b in RESTA_PAIRS:
                r = a - b
                primes = [NUM_ES[a], NUM_ES[b], NUM_ES[r]]
                out.append(make_item(
                    f"{NUM_EN[a]} minus {NUM_EN[b]} is {NUM_EN[r]}",
                    category, intent, primes, stage,
                ))
            continue

        if intent.startswith("qué_es_"):
            text, primes = QUE_ES[intent]
            out.append(make_item(text, category, intent, primes, stage))
            continue

        if intent.startswith("crecer_") or intent.startswith("vivencia_"):
            subject, conds = plant_conds(intent)
            cond_names = sorted(c[0] for c in conds)  # orden canónico
            cond_primes = [c[1] for c in conds if c[1]]

            # vivencia_*_aire (aire SOLO) es redundante con crecer_*_aire y
            # respiracion_*; además "live in air" es semánticamente raro.
            if intent.startswith("vivencia_") and cond_names == ["air"]:
                continue

            key = (intent[:6], subject, tuple(cond_names))
            if key in emitted_plant_conds:
                continue
            emitted_plant_conds.add(key)

            if intent.startswith("vivencia_"):
                text = f"{subject} live in " + " and ".join(cond_names)
                primes = cond_primes + ["vivir"]
            else:
                text = f"{subject} grow with " + " and ".join(cond_names)
                primes = cond_primes
            out.append(make_item(text, category, intent, primes, stage))
            continue

        if intent.startswith("respiracion_"):
            subject, _ = plant_conds(intent)
            out.append(make_item(f"{subject} breathe air", category, intent, [], stage))
            continue

        if intent == "necesidades_árboles":
            out.append(make_item("trees need water and light", category, intent, ["agua_prima", "luz"], stage))
            continue

        if intent.startswith("partes_"):
            for part in PLANT_PARTES[intent]:
                subject = "plants" if intent == "partes_plantas" else "trees"
                out.append(make_item(f"{subject} have {part}", category, intent, ["parte"], stage))
            continue

        if intent.startswith("crecimiento_"):
            subject = "plants" if "plantas" in intent else "trees"
            out.append(make_item(f"{subject} grow in soil and water", category, intent, ["agua_prima"], stage))
            continue

        if intent.startswith("pregunta_"):
            text = PREGUNTA[intent]
            out.append(make_item(text, category, intent, ["tú"], stage))
            continue

        if intent == "saluda":
            for text in SALUDA:
                out.append(make_item(text, category, intent, ["tú"], stage))
            continue

        if intent == "saluda_niño":
            out.append(make_item("hello child, how are you", category, intent, ["tú"], stage))
            continue

        if intent == "qué_hacer":
            out.append(make_item("what do you want to do today", category, intent, ["tú", "hacer"], stage))
            continue

        if intent == "qué_le_gusta":
            out.append(make_item("what do you like to do at school", category, intent, ["tú", "hacer"], stage))
            continue

        if intent == "qué_le_gusta_a_tu_amigo":
            out.append(make_item("what does your friend like to do", category, intent, ["tú", "alguien", "hacer"], stage))
            continue

        raise ValueError(f"intent sistemático sin plantilla: {intent}")

    return out


def main() -> None:
    src_path = BASE / "configs" / "school_curriculum_structured.json"
    vocab_path = BASE / "configs" / "clean_vocabulary_words.json"
    out_path = BASE / "configs" / "school_curriculum_structured_en.json"

    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    vocab = set(json.loads(vocab_path.read_text(encoding="utf-8"))["words"])

    # Agrupar por intent (los sistemáticos)
    from collections import defaultdict
    by_stage = {}
    for stage in ("preschool", "primary", "secondary"):
        groups = defaultdict(list)
        for it in data["curriculum"].get(stage, []):
            i = it["intent"]
            if (i == "suma" or i == "resta" or i.startswith(("qué_es_", "crecer_", "vivencia_",
                    "respiracion_", "partes_", "crecimiento_", "pregunta_"))
                    or i in ("necesidades_árboles", "saluda", "saluda_niño", "qué_hacer",
                             "qué_le_gusta", "qué_le_gusta_a_tu_amigo")):
                groups[i].append(it)
        by_stage[stage] = reconstruct(groups, stage)

    # Set preescolar rico (BIT-003 S4): items autorizados, deterministas.
    by_stage["preschool"].extend(
        make_item(text, "conversacion", "preschool_core", ["tú"], "preschool")
        for text in PRESCHOOL_CORE
    )

    total = sum(len(v) for v in by_stage.values())
    print(f"Reconstruidos: preschool={len(by_stage['preschool'])} primary={len(by_stage['primary'])} secondary={len(by_stage['secondary'])} (total {total})")

    # Validación estricta de vocabulario (RULE 7: ninguna palabra fuera del censo)
    TOKEN_RE = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>'\-]+")
    missing: dict[str, list[str]] = {}
    for stage, items in by_stage.items():
        for it in items:
            for w in TOKEN_RE.findall(it["text"].lower()):
                if w not in vocab:
                    missing.setdefault(w, []).append(it["text"])
    if missing:
        print("❌ Palabras fuera del vocabulario (RULE 7 violada):")
        for w, ex in sorted(missing.items()):
            print(f"   {w!r} ← {ex[0]}")
        print(f"   total palabras OOV: {len(missing)}")
        raise SystemExit(1)
    print("✅ Vocabulario validado: 0 palabras OOV.")

    final = {
        "metadata": {
            "version": "v3.0-structured-en-determinista",
            "total_sentences": total,
            "stats": {k: len(v) for k, v in by_stage.items()},
        },
        "curriculum": by_stage,
    }
    curr_str = json.dumps(by_stage, sort_keys=True, ensure_ascii=False)
    final["metadata"]["sha256"] = hashlib.sha256(curr_str.encode("utf-8")).hexdigest()

    out_path.write_text(json.dumps(final, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Escrito {out_path} (sha256={final['metadata']['sha256']})")


if __name__ == "__main__":
    main()