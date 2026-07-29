import torch
from tokenizers import Tokenizer

from src.bitnet.modeling_conversational import BitNetCausalLM


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer = Tokenizer.from_file("storage/experiments/conversational_tokenizer_en.json")
	vocab_size = tokenizer.get_vocab_size()

	model = BitNetCausalLM(
		vocab_size=vocab_size,
		hidden_dim=256,
		num_layers=4,
		num_heads=4,
		max_seq_len=256
	).to(device)

	model.load_state_dict(torch.load("storage/experiments/conversational_agent_en.pt", map_location=device, weights_only=True))
	model.eval()

	h_prev = torch.zeros((1, model.hidden_dim), device=device)
	
	# Turn 1
	prompt1 = "you: hello I have a cat [EOS] me:"
	print("Prompt 1 tokens:", tokenizer.encode(prompt1).tokens)
	
	input_ids1 = tokenizer.encode(prompt1).ids
	current_ids1 = list(input_ids1)
	generated1 = []
	
	with torch.no_grad():
		for _ in range(30):
			input_tensor = torch.tensor([current_ids1], dtype=torch.long, device=device)
			logits, next_h = model(input_tensor, h_prev=h_prev)
			next_token = logits[0, -1, :].argmax(dim=-1).item()
			if next_token == tokenizer.token_to_id("[EOS]"):
				break
			generated1.append(next_token)
			current_ids1.append(next_token)
		
		h_prev = next_h

	print("User: hello I have a cat")
	print("Agent:", tokenizer.decode(generated1))

	# Turn 2 with memory (h_prev), no prompt history
	prompt2 = "you: no I have a dog [EOS] me:"
	input_ids2 = tokenizer.encode(prompt2).ids
	current_ids2 = list(input_ids2)
	generated2 = []

	with torch.no_grad():
		for _ in range(30):
			input_tensor = torch.tensor([current_ids2], dtype=torch.long, device=device)
			logits, next_h = model(input_tensor, h_prev=h_prev)
			next_token = logits[0, -1, :].argmax(dim=-1).item()
			if next_token == tokenizer.token_to_id("[EOS]"):
				break
			generated2.append(next_token)
			current_ids2.append(next_token)

	print("User: no I have a dog (WITH memory, NO prompt history)")
	print("Agent:", tokenizer.decode(generated2))

if __name__ == "__main__":
	main()
