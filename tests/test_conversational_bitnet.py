import torch

from src.bitnet.model.modeling_conversational import BitNetCausalLM


def test_conversational_model_shapes():
	# Test basics shapes and forward pass
	batch_size = 4
	seq_len = 16
	vocab_size = 8000
	hidden_dim = 128
	
	model = BitNetCausalLM(
		vocab_size=vocab_size,
		hidden_dim=hidden_dim,
		num_layers=2,
		num_heads=2,
		max_seq_len=64
	)
	
	input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
	
	# Forward pass standard
	logits, next_h_prev = model(input_ids)
	
	assert logits.shape == (batch_size, seq_len, vocab_size)
	assert next_h_prev.shape == (batch_size, hidden_dim)

def test_conversational_model_memory():
	# Verify that h_prev impacts logits
	batch_size = 2
	seq_len = 8
	vocab_size = 1000
	hidden_dim = 64
	
	model = BitNetCausalLM(
		vocab_size=vocab_size,
		hidden_dim=hidden_dim,
		num_layers=1,
		num_heads=2,
		max_seq_len=32
	)
	
	input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
	
	# Forward pass without memory
	logits_no_mem, _ = model(input_ids)
	
	# Forward pass with memory
	h_prev = torch.randn(batch_size, hidden_dim)
	logits_with_mem, _ = model(input_ids, h_prev=h_prev)
	
	assert not torch.allclose(logits_no_mem, logits_with_mem)

def test_conversational_model_causal_mask():
	# Verify that prediction of token t does not look ahead to t+1
	vocab_size = 1000
	hidden_dim = 64
	
	model = BitNetCausalLM(
		vocab_size=vocab_size,
		hidden_dim=hidden_dim,
		num_layers=2,
		num_heads=2,
		max_seq_len=32
	)
	model.eval()
	
	input_ids1 = torch.tensor([[10, 20, 30, 40]])
	input_ids2 = torch.tensor([[10, 20, 30, 99]]) # different last token
	
	with torch.no_grad():
		logits1, _ = model(input_ids1)
		logits2, _ = model(input_ids2)
		
	# First three tokens should have exact same logits because attention is causal
	assert torch.allclose(logits1[:, :3, :], logits2[:, :3, :], atol=1e-5)
	# The last token logits can be different
	assert not torch.allclose(logits1[:, 3, :], logits2[:, 3, :])

def test_english_conversational_model_loading():
	# Verify that the trained English model and tokenizer can be loaded
	import os

	from tokenizers import Tokenizer
	tokenizer_path = "storage/experiments/conversational_tokenizer_en.json"
	model_path = "storage/experiments/conversational_agent_en.pt"
	
	if os.path.exists(tokenizer_path) and os.path.exists(model_path):
		tokenizer = Tokenizer.from_file(tokenizer_path)
		vocab_size = tokenizer.get_vocab_size()
		
		state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
		hidden_dim = state_dict["token_embedding.weight"].shape[1]
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
		)
		
		model.load_state_dict(state_dict)
		model.eval()
		
		assert model.vocab_size == vocab_size

def test_spanish_conversational_model_loading():
	# Verify that the trained Spanish model and tokenizer can be loaded
	import os

	from tokenizers import Tokenizer
	tokenizer_path = "storage/experiments/conversational_tokenizer.json"
	model_path = "storage/experiments/conversational_agent.pt"
	
	if os.path.exists(tokenizer_path) and os.path.exists(model_path):
		tokenizer = Tokenizer.from_file(tokenizer_path)
		vocab_size = tokenizer.get_vocab_size()
		
		state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
		hidden_dim = state_dict["token_embedding.weight"].shape[1]
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
		)
		
		model.load_state_dict(state_dict)
		model.eval()
		
		assert model.vocab_size == vocab_size

