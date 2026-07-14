import json
import os
import sys
import re
import subprocess

base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(base_dir)

def query_samantha_translation(prompt: str) -> str:
    sharing_venv_python = "/home/joan/Documents/IA/sharing/.venv/bin/python"
    sharing_src = "/home/joan/Documents/IA/sharing/src"

    cmd_code = f"""
import sys
sys.path.append('{sharing_src}')
from red_pill.inference import samantha_on_demand
res = samantha_on_demand.invoke({repr(prompt)}, system_prompt="You are a precise translator. Translate the given Spanish sentences into natural, simple English suitable for children. Output ONLY the translation, no extra text or explanations.", max_tokens=1024, temperature=0.0)
print(res)
"""
    try:
        result = subprocess.run([sharing_venv_python, "-c", cmd_code], capture_output=True, text=True, timeout=90)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"Error querying Samantha: {e}")
    return ""

def main():
    input_path = os.path.join(base_dir, "configs", "school_curriculum_structured.json")
    output_path = os.path.join(base_dir, "configs", "school_curriculum_structured_en.json")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    curriculum = data.get("curriculum", {})
    translated_curriculum = {}

    for stage, items in curriculum.items():
        print(f"Translating stage: {stage} ({len(items)} items)...")
        translated_items = []
        
        # Batch translation to speed up
        batch_size = 20
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            prompt = "Translate the following Spanish sentences to English one by one, separated by newlines:\n"
            for item in batch:
                prompt += f"- {item['text']}\n"
            
            print(f"  Processing batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}...")
            translation_output = query_samantha_translation(prompt)
            
            lines = [line.strip().lstrip("- ").strip() for line in translation_output.split("\n") if line.strip()]
            
            # If line count matches, match them up
            if len(lines) == len(batch):
                for idx, item in enumerate(batch):
                    new_item = item.copy()
                    new_item["text"] = lines[idx]
                    translated_items.append(new_item)
            else:
                # Fallback: translate one by one
                print(f"    Mismatch in batch lines ({len(lines)} vs {len(batch)}). Translating one-by-one...")
                for item in batch:
                    single_prompt = f"Translate to simple English: {item['text']}"
                    single_trans = query_samantha_translation(single_prompt)
                    # Strip quotes if any
                    single_trans = re.sub(r'^["\']|["\']$', '', single_trans).strip()
                    new_item = item.copy()
                    new_item["text"] = single_trans
                    translated_items.append(new_item)
                    
        translated_curriculum[stage] = translated_items

    # Save translation
    out_data = {
        "metadata": {
            "version": "v2.0-structured-en",
            "total_sentences": sum(len(v) for v in translated_curriculum.values()),
            "stats": {k: len(v) for k, v in translated_curriculum.items()}
        },
        "curriculum": translated_curriculum
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4, ensure_ascii=False)
        
    print(f"Saved translated curriculum to {output_path}")

if __name__ == "__main__":
    main()
