import sys
sys.path.append('/home/joan/Documents/IA/sharing/src')
from red_pill.inference import samantha_on_demand

system_prompt = "Answer with yes or no."
prompt = "Are the word 'three' and the word 'three' the same word? Answer yes or no."

try:
	res = samantha_on_demand.invoke(prompt, system_prompt=system_prompt, max_tokens=100, temperature=0.0)
	print("--- RESPUESTA ---")
	print(res)
except Exception as e:
	print(f"ERROR: {e}")
