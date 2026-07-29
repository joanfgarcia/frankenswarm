import os
import torch
import sys
from tokenizers import Tokenizer

sys.path.append("/home/joan/Documents/IA/frankenswarm")
from src.bitnet.modeling_conversational import BitNetCausalLM

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	tokenizer_path = os.path.join(base_dir, "storage", "experiments", "conversational_tokenizer.json")
	model_path = os.path.join(base_dir, "storage", "experiments", "conversational_agent.pt")
	
	if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
		print("Tokenizer or model not found!")
		return
		
	tokenizer = Tokenizer.from_file(tokenizer_path)
	vocab_size = tokenizer.get_vocab_size()
	print(f"Tokenizer vocabulary size: {vocab_size}")
	
	state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
	model_vocab_size = state_dict["token_embedding.weight"].shape[0]
	hidden_dim = state_dict["token_embedding.weight"].shape[1]
	print(f"Model token embedding shape: {state_dict['token_embedding.weight'].shape}")
	
	# Print some tokens
	print("\nFirst 30 tokens in tokenizer:")
	for i in range(min(30, vocab_size)):
		print(f"{i}: {tokenizer.decode([i])!r} (id to token: {tokenizer.id_to_token(i)!r})")
		
	# Test encode/decode
	test_str = "hola tú: yo: [EOS]"
	encoded = tokenizer.encode(test_str)
	print(f"\nEncoding test: {test_str!r}")
	print(f"Tokens: {encoded.tokens}")
	print(f"IDs: {encoded.ids}")
	print(f"Decoded: {tokenizer.decode(encoded.ids)!r}")

	# Initialize model
	if hidden_dim == 512:
		num_layers = 8
		num_heads = 8
	else:
		num_layers = 4
		num_heads = 4
		
	model = BitNetCausalLM(
		vocab_size=vocab_size,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		num_heads=num_heads,
		max_seq_len=256
	).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
	model.eval()
	
	# Run a simple generation step
	prompt = "tú: hola [EOS] yo:"
	input_ids = tokenizer.encode(prompt).ids
	print(f"\nPrompt: {prompt!r}")
	print(f"Prompt IDs: {input_ids}")
	
	# Argmax prediction for the next 10 tokens
	current_ids = list(input_ids)
	h_prev = torch.zeros((1, model.hidden_dim), device=device)
	generated = []
	with torch.no_grad():
		for step in range(10):
			input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
			logits, next_h = model(input_tensor, h_prev=h_prev)
			next_token = logits[0, -1, :].argmax().item()
			generated.append(next_token)
			current_ids.append(next_token)
			h_prev = next_h
			
	print(f"Generated IDs: {generated}")
	print(f"Generated tokens: {[tokenizer.id_to_token(i) for i in generated]}")
	print(f"Decoded response: {tokenizer.decode(generated)!r}")

if __name__ == '__main__':
	main()
