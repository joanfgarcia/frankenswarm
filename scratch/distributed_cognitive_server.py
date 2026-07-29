import os
import sys
import json
import asyncio
import struct
import torch
import torch.nn.functional as F
import numpy as np

# Añadir src al path
base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "src"))

from bitnet.glyph_vocabulary import N_EMOTIONS
from bitnet.modeling_bitnet import BitNet4LayerModel

# Configurar Socket
SOCKET_PATH = "/tmp/bit_cognitive.sock"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def read_exact(reader, n):
	"""Leer exactamente n bytes del stream del socket."""
	data = b''
	while len(data) < n:
		packet = await reader.read(n - len(data))
		if not packet:
			return None
		data += packet
	return data

class CognitiveServer:
	def __init__(self):
		print("═══ 🧠 Servidor Cognitivo BitNET — Inicializando ═══")
		print(f"[Device]: {DEVICE}")
		
		# 1. Cargar vocabulario de 15,005
		expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
		with open(expanded_glyphs_path, encoding="utf-8") as f:
			vocab_data = json.load(f)
			self.glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

		# 2. Inicializar Model B (Bit, dim 384, layers 6)
		self.model = BitNet4LayerModel(
			use_glyphs=True,
			glyph_table=self.glyphs,
			hidden_dim=384,
			num_layers=6,
			use_pos_embedding=True,
			n_emotions=N_EMOTIONS,
			emotion_dim=64,
			emotion_mode="first_only",
		).to(DEVICE)

		path_b = os.path.join(base_dir, "storage/checkpoints/sovereign_school/model_milestone_3_years.pt")
		if os.path.exists(path_b):
			sd_b = torch.load(path_b, map_location=DEVICE, weights_only=True)
			self.model.load_state_dict(sd_b, strict=False)
			print(f"Cargado Modelo B (Bit) del checkpoint: {path_b}")
		else:
			print(f"⚠️ Checkpoint B no encontrado en {path_b}!")

		self.model.eval()
		for param in self.model.parameters():
			param.requires_grad = False

		# Guardar embeddings de glifos en GPU para acelerar la proyección
		self.word_embeddings = self.model.glyph_embedding.get_word_embeddings()

	async def handle_connection(self, reader, writer):
		try:
			# 1. Leer cabecera (12 bytes: B, S, D)
			header = await read_exact(reader, 12)
			if not header:
				return
			B, S, D = struct.unpack('!3i', header)

			# 2. Leer payload (B * S * D * 4 bytes)
			payload_len = B * S * D * 4
			payload = await read_exact(reader, payload_len)
			if not payload:
				return

			# 3. Reconstruir tensor de probs_a en GPU
			probs_a_np = np.frombuffer(payload, dtype=np.float32).reshape(B, S, D).copy()
			probs_a = torch.from_numpy(probs_a_np).to(DEVICE) # (B, S, 15005)

			# 4. Inferencia en capas cognitivas (layers 3 y 4 de Bit)
			with torch.no_grad():
				# Re-embeber en B
				h_b = torch.matmul(probs_a, self.word_embeddings)

				# Capas 3, 4
				for i in range(3, 5):
					h_b = self.model.core_layers[i](h_b)
				h_b = self.model.norm(h_b)

				# Decodificar y aplicar softmax
				logits_b = self.model._decode_hidden(h_b)
				probs_b = F.softmax(logits_b / 1.0, dim=-1) # (B, S, 15005)

			# 5. Enviar respuesta binaria de vuelta al cliente
			probs_b_np = probs_b.cpu().numpy().astype(np.float32)
			B_out, S_out, D_out = probs_b_np.shape
			
			# Cabecera de salida
			out_header = struct.pack('!3i', B_out, S_out, D_out)
			writer.write(out_header)
			writer.write(probs_b_np.tobytes())
			await writer.drain()

		except Exception as e:
			print(f"[Error en conexión]: {e}")
		finally:
			writer.close()
			await writer.wait_closed()

	async def start(self):
		# Eliminar socket anterior si existe
		if os.path.exists(SOCKET_PATH):
			os.remove(SOCKET_PATH)

		server = await asyncio.start_unix_server(self.handle_connection, SOCKET_PATH)
		print(f"📡 Servidor escuchando en UNIX socket: {SOCKET_PATH}")

		async with server:
			await server.serve_forever()

if __name__ == "__main__":
	server = CognitiveServer()
	try:
		asyncio.run(server.start())
	except KeyboardInterrupt:
		print("\n🛑 Servidor apagado voluntariamente.")
		if os.path.exists(SOCKET_PATH):
			os.remove(SOCKET_PATH)
