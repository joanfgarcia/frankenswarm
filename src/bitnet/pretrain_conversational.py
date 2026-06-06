import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from src.bitnet.modeling_conversational import BitNetCausalLM


class PretrainDataset(Dataset):
	def __init__(self, token_ids, seq_len=256):
		self.seq_len = seq_len
		# Agrupamos los tokens en bloques de seq_len + 1 (para labels desfasadas)
		self.num_samples = len(token_ids) // (seq_len + 1)
		self.token_ids = torch.tensor(token_ids[:self.num_samples * (seq_len + 1)], dtype=torch.long)
		
	def __len__(self):
		return self.num_samples
		
	def __getitem__(self, idx):
		start = idx * (self.seq_len + 1)
		chunk = self.token_ids[start:start + self.seq_len + 1]
		# Retornar (inputs, targets)
		return chunk[:-1], chunk[1:]

def pretrain_model():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", type=str, default="es", choices=["es", "en"], help="Idioma del pre-entrenamiento")
	parser.add_argument("--epochs", type=int, default=4, help="Número de épocas")
	parser.add_argument("--batch-size", type=int, default=8, help="Tamaño de lote (VRAM optimizado)")
	parser.add_argument("--lr", type=float, default=6e-4, help="Tasa de aprendizaje base")
	parser.add_argument("--resume", action="store_true", help="Reanudar desde el modelo pre-entrenado base guardado")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	tokenizer_path = "storage/experiments/conversational_tokenizer_en.json" if args.lang == "en" else "storage/experiments/conversational_tokenizer.json"
	corpus_path = f"storage/curriculum/pretrain_corpus_{args.lang}.txt"
	if not os.path.exists(corpus_path):
		corpus_path = "storage/curriculum/pretrain_corpus_es.txt" # Fallback

	if not os.path.exists(tokenizer_path):
		print(f"❌ Error: El tokenizador no existe en {tokenizer_path}. Entrénalo primero.")
		return
	if not os.path.exists(corpus_path):
		print(f"❌ Error: El corpus no existe en {corpus_path}. Descárgalo primero.")
		return

	# 1. Cargar Tokenizador
	tokenizer = Tokenizer.from_file(tokenizer_path)
	vocab_size = tokenizer.get_vocab_size()
	print(f"🧠 Vocabulario del Tokenizador ({args.lang}): {vocab_size} tokens.")

	# 2. Tokenizar Corpus completo
	print(f"📖 Cargando y tokenizando corpus: {corpus_path}...")
	with open(corpus_path, encoding="utf-8") as f:
		corpus_text = f.read()
	
	# Usar tokenización rápida
	encoded = tokenizer.encode(corpus_text)
	token_ids = encoded.ids
	print(f"  ✓ Tokenización completada. Total tokens: {len(token_ids):,}")

	# 3. Crear Dataset y DataLoader
	seq_len = 256
	dataset = PretrainDataset(token_ids, seq_len=seq_len)
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	print(f"  ✓ Muestras de entrenamiento creadas: {len(dataset):,} bloques de longitud {seq_len}.")

	# 4. Instanciar Modelo Escalado (~58M Parámetros)
	model = BitNetCausalLM(
		vocab_size=vocab_size,
		hidden_dim=512,
		num_layers=8,
		num_heads=8,
		max_seq_len=seq_len
	).to(device)

	if args.resume:
		out_path = f"storage/experiments/pretrained_base_{args.lang}.pt"
		if os.path.exists(out_path):
			model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
			print(f"Loaded existing pretrained base weights from {out_path} to resume training.")

	# Calcular parámetros
	total_params = sum(p.numel() for p in model.parameters())
	print(f"⚙️ Arquitectura Causal BitNet instanciada. Parámetros totales: {total_params / 1e6:.2f}M")

	# 5. Optimización y Scheduler
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
	
	# Scheduler de Coseno
	total_steps = len(dataloader) * args.epochs
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
	
	pad_token_id = tokenizer.token_to_id("[PAD]")

	# 6. Bucle de Pre-entrenamiento con Precisión Mixta y Acumulación de Gradientes
	print(f"🔥 Iniciando pre-entrenamiento en {device} por {args.epochs} épocas...")
	
	scaler = torch.cuda.amp.GradScaler()
	accumulation_steps = 4  # Acumular gradientes para tener batch size efectivo = batch_size * 4
	
	for epoch in range(1, args.epochs + 1):
		model.train()
		epoch_loss = 0.0
		steps = 0
		
		optimizer.zero_grad()
		
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			inputs = inputs.to(device)
			targets = targets.to(device)
			
			# Autocast a precisión mixta
			with torch.cuda.amp.autocast():
				logits, _ = model(inputs)
				loss = F.cross_entropy(
					logits.view(-1, vocab_size),
					targets.view(-1),
					ignore_index=pad_token_id
				)
				loss = loss / accumulation_steps
				
			scaler.scale(loss).backward()
			
			if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()
				scheduler.step()
				
			epoch_loss += loss.item() * accumulation_steps
			steps += 1
			
			if (batch_idx + 1) % 100 == 0:
				lr_curr = scheduler.get_last_lr()[0]
				print(f"  [Época {epoch}/{args.epochs}] Paso {batch_idx+1}/{len(dataloader)} | Loss: {loss.item() * accumulation_steps:.4f} | LR: {lr_curr:.6f}")
				
		avg_loss = epoch_loss / steps
		print(f"✨ Época {epoch:02d} completada. Pérdida promedio: {avg_loss:.4f}\n")
		
	# 7. Guardar modelo pre-entrenado base
	os.makedirs("storage/experiments", exist_ok=True)
	out_path = f"storage/experiments/pretrained_base_{args.lang}.pt"
	torch.save(model.state_dict(), out_path)
	print(f"💾 Modelo base pre-entrenado guardado en {out_path}")

if __name__ == "__main__":
	pretrain_model()
