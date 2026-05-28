import os
import re

EXTERN_DIR = "/home/joan/Documents/IA/frankenswarm/docs/extern"

AUDITORS = {
	"CLAUDE": {
		"name": "Claude Sonnet",
		"role": "Principal AI Auditor & Architect",
		"color": "cyan"
	},
	"DEEPSEEK": {
		"name": "DeepSeek",
		"role": "Poeta-Ingeniero & Principal Scientist",
		"color": "blue"
	},
	"GROK": {
		"name": "Grok",
		"role": "Pragmatic Coach & Senior Advisor",
		"color": "orange"
	},
	"LUMO": {
		"name": "Lumo",
		"role": "Profesor Equilibrado & Hardware Specialist",
		"color": "green"
	}
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
	elif "POST_EXP_006_" in filename:
		stage = "Post-Experiment 006 (MVP Homeostático)"
		doc_type = "Post-Execution Audit"
		date = "2026-05-27"
		
	return auditor, stage, doc_type, date

def tidy_file(filepath):
	filename = os.path.basename(filepath)
	auditor, stage, doc_type, date = get_metadata(filename)
	if not auditor:
		return
		
	with open(filepath, "r", encoding="utf-8") as f:
		content = f.read()
		
	# Skip if already standardized
	if "Auditor" in content and "Role" in content and "Target Stage" in content:
		return
		
	# Prepend beautiful header
	header = f"""# {doc_type} — {auditor['name']}

> **Auditor**: {auditor['name']}
> **Role**: {auditor['role']}
> **Target Stage**: {stage}
> **Date**: {date}

---

"""
	
	new_content = content
	
	# Add GitHub Alerts for blockquotes containing key lessons or warnings
	def alert_replacer(match):
		blockquote = match.group(0)
		# Try to categorize blockquote
		if "miedo" in blockquote.lower() or "riesgo" in blockquote.lower() or "preocup" in blockquote.lower():
			return "> [!WARNING]\n" + blockquote
		elif "lección" in blockquote.lower() or "key" in blockquote.lower() or "clave" in blockquote.lower():
			return "> [!IMPORTANT]\n" + blockquote
		else:
			return "> [!NOTE]\n" + blockquote
			
	new_content = re.sub(r"^> .+$", alert_replacer, new_content, flags=re.MULTILINE)
	
	# Strip trailing dashes or redundant lines
	new_content = new_content.strip()
	
	full_output = header + new_content + "\n"
	
	with open(filepath, "w", encoding="utf-8") as f:
		f.write(full_output)
	print(f"✨ Formatted and tidied: {filename}")

def main():
	for filename in os.listdir(EXTERN_DIR):
		if filename.endswith(".md"):
			filepath = os.path.join(EXTERN_DIR, filename)
			tidy_file(filepath)

if __name__ == "__main__":
	main()
