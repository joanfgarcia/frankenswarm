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
