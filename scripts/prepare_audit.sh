#!/usr/bin/env bash

# Frankenswarm Certification Preparation Script
# Combines all relevant code and documentation into a single digest file for LLM Auditors.

OUTPUT_FILE="FRANKENSWARM_DIGEST.txt"

echo "Creating Frankenswarm Certification Digest..."
> "$OUTPUT_FILE"

append_file() {
	local file=$1
	if [ -f "$file" ]; then
		echo "Appending $file..."
		echo -e "\n\n================================================================" >> "$OUTPUT_FILE"
		echo "FILE: $file" >> "$OUTPUT_FILE"
		echo "================================================================" >> "$OUTPUT_FILE"
		cat "$file" >> "$OUTPUT_FILE"
	fi
}

# Documentation & Configs at root
append_file "README.md"
append_file "CHANGELOG.md"
append_file "CONVENTIONS.md"
append_file "pyproject.toml"

# Core Prolog & Python Bridge
append_file "frankenswarm_core.pl"
append_file "pyswip_bridge.py"

# Deep Documentation
# NOTE: docs/extern/ is INTENTIONALLY EXCLUDED from the digest.
# It contains external audit reports and third-party opinions.
# Including them would bias future auditors by exposing prior conclusions,
# compromising the independence of their assessment.
find docs -type f -name "*.md" -not -path "docs/extern/*" 2>/dev/null | sort | while read -r line; do
	append_file "$line"
done

# Source Files (Python & Prolog)
find src -type f \( -name "*.py" -o -name "*.pl" \) 2>/dev/null | sort | while read -r line; do
	append_file "$line"
done

# Unit & Integration Tests
find tests -type f -name "*.py" 2>/dev/null | sort | while read -r line; do
	append_file "$line"
done

echo "Done! The digest is ready at $OUTPUT_FILE."
echo "Total payload: $(wc -l < "$OUTPUT_FILE") lines."
