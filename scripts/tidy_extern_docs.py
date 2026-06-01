import os
import re

EXTERN_DIR = "/home/joan/Documents/IA/frankenswarm/docs/extern"

AUDITORS = {
	"CLAUDE": {"name": "Claude Sonnet", "role": "Principal AI Auditor & Architect", "color": "cyan"},
	"DEEPSEEK": {"name": "DeepSeek", "role": "Poeta-Ingeniero & Principal Scientist", "color": "blue"},
	"GROK": {"name": "Grok", "role": "Pragmatic Coach & Senior Advisor", "color": "orange"},
	"LUMO": {"name": "Lumo", "role": "Profesor Equilibrado & Hardware Specialist", "color": "green"},
}


def get_metadata(filename):
	# Detect auditor
	auditor_key = None
	for key in AUDITORS:
		if key in filename.upper():
			auditor_key = key
			break

	if not auditor_key:
		return None, None, None, None

	auditor = AUDITORS[auditor_key]

	# Detect phase
	stage = "General Research Evaluation"
	doc_type = "External Audit Report"
	date = "2026-05-26"

	if "PRE_0_" in filename:
		stage = "Pre-Phase 0 Evaluation"
		doc_type = "Pre-Execution Audit"
		date = "2026-05-26"
	elif "POST_0_" in filename:
		stage = "Post-Phase 0 Evaluation"
		doc_type = "Post-Execution Audit"
		date = "2026-05-26"
	elif "POST_EXP_005_" in filename:
		stage = "Post-Experiment 005 (Proto-Syntax Analysis)"
		doc_type = "Post-Execution Audit"
		date = "2026-05-26"
	elif "POST_EXP_006_DIVERGENCE_" in filename:
		stage = "Post-Experiment 006 (Dialect Divergence)"
		doc_type = "Post-Execution Audit"
		date = "2026-05-28"
	elif "POST_EXP_006_" in filename:
		stage = "Post-Experiment 006 (MVP Homeostático)"
		doc_type = "Post-Execution Audit"
		date = "2026-05-27"

	return auditor, stage, doc_type, date


def tidy_file(filepath):
	filename = os.path.basename(filepath)
	# Skip special files like README.md, TESIS_SESSION_1.md and results reports
	if filename in ["README.md", "TESIS_SESSION_1.md", "PHASE_0_RESULTS_REPORT.md"]:
		return

	auditor, stage, doc_type, date = get_metadata(filename)
	if not auditor:
		return

	with open(filepath, encoding="utf-8") as f:
		content = f.read()

	# Skip if already standardized (with H1 matching standard H1)
	expected_h1 = f"# {doc_type} — {auditor['name']}"
	if expected_h1 in content and "Auditor" in content and "Role" in content and "Target Stage" in content:
		return

	# Clean existing header if any
	cleaned_content = re.sub(r"^#\s+.*?\n", "", content, flags=re.MULTILINE)
	cleaned_content = re.sub(r"^>\s*\*\*Auditor\*\*.*?\n", "", cleaned_content, flags=re.MULTILINE)
	cleaned_content = re.sub(r"^>\s*\*\*Role\*\*.*?\n", "", cleaned_content, flags=re.MULTILINE)
	cleaned_content = re.sub(r"^>\s*\*\*Target Stage\*\*.*?\n", "", cleaned_content, flags=re.MULTILINE)
	cleaned_content = re.sub(r"^>\s*\*\*Date\*\*.*?\n", "", cleaned_content, flags=re.MULTILINE)
	cleaned_content = re.sub(r"^---\s*---\s*", "", cleaned_content, flags=re.MULTILINE)
	cleaned_content = re.sub(r"^---\s*", "", cleaned_content, flags=re.MULTILINE)
	cleaned_content = cleaned_content.strip()

	header = f"""# {doc_type} — {auditor["name"]}

> **Auditor**: {auditor["name"]}
> **Role**: {auditor["role"]}
> **Target Stage**: {stage}
> **Date**: {date}

---

"""

	# Add GitHub Alerts for blockquotes containing key lessons or warnings
	def alert_replacer(match):
		blockquote = match.group(0)
		if "[!" in blockquote:
			return blockquote
		if (
			"miedo" in blockquote.lower()
			or "riesgo" in blockquote.lower()
			or "preocup" in blockquote.lower()
			or "dificil" in blockquote.lower()
			or "fracaso" in blockquote.lower()
			or "pesadilla" in blockquote.lower()
		):
			return "> [!WARNING]\n" + blockquote
		elif (
			"lección" in blockquote.lower()
			or "key" in blockquote.lower()
			or "clave" in blockquote.lower()
			or "recomend" in blockquote.lower()
			or "solución" in blockquote.lower()
		):
			return "> [!IMPORTANT]\n" + blockquote
		else:
			return "> [!NOTE]\n" + blockquote

	new_content = re.sub(r"^> .+$", alert_replacer, cleaned_content, flags=re.MULTILINE)
	new_content = new_content.strip()

	full_output = header + new_content + "\n"

	with open(filepath, "w", encoding="utf-8") as f:
		f.write(full_output)
	print(f"✨ Standardized and formatted: {filename}")


def main():
	for filename in os.listdir(EXTERN_DIR):
		if filename.endswith(".md"):
			filepath = os.path.join(EXTERN_DIR, filename)
			tidy_file(filepath)


if __name__ == "__main__":
	main()
