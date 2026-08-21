"""Fija el fast-path one-hot del brazo estándar (DL-006).

El brazo estándar se construye con `vocab_embeddings=np.eye(V)`: una tabla
identidad congelada cuyas columnas de `inbound_proj` SON el embedding entrenable.
El camino denso original multiplicaba por esa identidad en la entrada y en la
salida — V×V multiplicaciones gratuitas por posición de token, y 590 MB de tabla
en cada checkpoint a 12k tokens.

Estos tests fijan que el cortocircuito es **exactamente equivalente** (no
"aproximadamente"), que el embedding sigue siendo entrenable, y que los
checkpoints antiguos siguen cargando. Sin ellos, el ahorro sería indistinguible
de una regresión silenciosa en la comparativa de la tesis.
"""

import io

import numpy as np
import pytest
import torch

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel, _is_identity_matrix

VOCAB = 24
DIM = 16
LAYERS = 2
SEQ = 8


def build_standard(vocab=VOCAB, dim=DIM):
	torch.manual_seed(770)
	return BitNet4LayerModel(
		use_glyphs=False,
		vocab_embeddings=np.eye(vocab, dtype=np.float32),
		hidden_dim=dim,
		num_layers=LAYERS,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=SEQ,
	)


class TestIdentityDetection:
	def test_detects_identity(self):
		assert _is_identity_matrix(np.eye(5, dtype=np.float32))

	def test_rejects_non_square(self):
		assert not _is_identity_matrix(np.zeros((3, 4), dtype=np.float32))

	def test_rejects_dense_matrix(self):
		assert not _is_identity_matrix(np.random.RandomState(0).randn(6, 6).astype(np.float32))

	def test_rejects_permutation(self):
		"""Una permutación tiene n no-nulos pero la diagonal no es de unos."""
		perm = np.eye(4, dtype=np.float32)[[1, 0, 3, 2]]
		assert not _is_identity_matrix(perm)

	def test_rejects_scaled_identity(self):
		assert not _is_identity_matrix(np.eye(4, dtype=np.float32) * 2.0)

	def test_standard_arm_flags_onehot(self):
		assert build_standard().onehot_vocab is True

	def test_glyph_arm_is_not_onehot(self):
		torch.manual_seed(770)
		glyphs = np.random.RandomState(1).randint(-1, 2, size=(VOCAB, 65)).astype(np.float32)
		model = BitNet4LayerModel(use_glyphs=True, glyph_table=glyphs, hidden_dim=DIM, num_layers=LAYERS)
		assert model.onehot_vocab is False


class TestExactEquivalence:
	"""El fast-path debe dar el MISMO resultado que el camino denso, bit a bit.

	Se ejecuta el mismo modelo (mismos pesos) por los dos caminos volteando la
	bandera: cualquier diferencia sería un cambio de semántica encubierto.
	"""

	def test_token_ids_path_is_bitwise_identical(self):
		model = build_standard().eval()
		x = torch.randint(0, VOCAB, (4, SEQ))
		with torch.no_grad():
			fast = model(x)
			model.onehot_vocab = False  # fuerza el camino denso original
			dense = model(x)
		assert torch.equal(fast, dense)

	def test_gumbel_soft_path_is_bitwise_identical(self):
		model = build_standard().eval()
		soft = torch.softmax(torch.randn(3, SEQ, VOCAB), dim=-1)
		with torch.no_grad():
			fast = model(soft)
			model.onehot_vocab = False
			dense = model(soft)
		assert torch.equal(fast, dense)

	def test_logit_mask_still_applies(self):
		model = build_standard().eval()
		x = torch.randint(0, VOCAB, (2, SEQ))
		mask = torch.zeros(VOCAB, dtype=torch.bool)
		mask[:5] = True
		with torch.no_grad():
			logits = model(x, logit_mask=mask)
		assert torch.all(logits[..., 5:] <= -1e8)
		assert torch.all(logits[..., :5] > -1e8)


class TestEmbeddingStaysTrainable:
	"""El riesgo del cortocircuito: romper el gradiente y dejar de aprender en silencio."""

	def test_gradient_reaches_used_token_columns(self):
		model = build_standard()
		used = 3
		x = torch.full((2, SEQ), used, dtype=torch.long)
		logits = model(x)
		logits.sum().backward()
		grad = model.inbound_proj.weight.grad
		assert grad is not None
		assert torch.any(grad[:, used] != 0), "el embedding del token usado no recibe gradiente"

	def test_unused_token_columns_get_no_gradient(self):
		"""Firma de actualización dispersa: igual que nn.Embedding y que el camino denso."""
		model = build_standard()
		x = torch.full((2, SEQ), 3, dtype=torch.long)
		model(x).sum().backward()
		grad = model.inbound_proj.weight.grad
		unused = 17
		assert torch.all(grad[:, unused] == 0)

	def test_output_head_receives_gradient(self):
		model = build_standard()
		x = torch.randint(0, VOCAB, (2, SEQ))
		model(x).sum().backward()
		assert model.outbound_proj.weight.grad is not None
		assert torch.any(model.outbound_proj.weight.grad != 0)


class TestCheckpointShape:
	def test_identity_table_is_not_persisted(self):
		assert "vocab_embeddings" not in build_standard().state_dict()

	def test_buffer_still_available_in_memory(self):
		"""net2wider_model reconstruye el modelo leyendo model.vocab_embeddings."""
		model = build_standard()
		assert model.vocab_embeddings is not None
		assert model.vocab_embeddings.shape == (VOCAB, VOCAB)

	def test_checkpoint_shrinks_by_the_identity_table(self):
		model = build_standard(vocab=256, dim=DIM)
		buf = io.BytesIO()
		torch.save(model.state_dict(), buf)
		identity_bytes = 256 * 256 * 4
		assert buf.getbuffer().nbytes < identity_bytes

	def test_legacy_checkpoint_with_identity_table_still_loads(self):
		"""Checkpoints del brazo estándar anteriores al fast-path: la clave sobra, no rompe."""
		model = build_standard()
		legacy = dict(model.state_dict())
		legacy["vocab_embeddings"] = torch.eye(VOCAB)
		model.load_state_dict(legacy)  # strict=True por defecto

	def test_legacy_load_still_rejects_a_genuinely_missing_weight(self):
		"""La tolerancia es quirúrgica: solo la tabla reconstruible, nada más."""
		model = build_standard()
		broken = dict(model.state_dict())
		broken.pop("inbound_proj.weight")
		with pytest.raises(RuntimeError):
			model.load_state_dict(broken)


class TestDenseVocabArmUntouched:
	"""Regresión: el modo clásico con vectores fastembed reales no se toca."""

	def test_dense_table_is_persisted(self):
		torch.manual_seed(770)
		dense = np.random.RandomState(2).randn(VOCAB, 32).astype(np.float32)
		model = BitNet4LayerModel(use_glyphs=False, vocab_embeddings=dense, hidden_dim=DIM, num_layers=LAYERS)
		assert model.onehot_vocab is False
		assert "vocab_embeddings" in model.state_dict()

	def test_dense_forward_runs(self):
		torch.manual_seed(770)
		dense = np.random.RandomState(3).randn(VOCAB, 32).astype(np.float32)
		model = BitNet4LayerModel(use_glyphs=False, vocab_embeddings=dense, hidden_dim=DIM, num_layers=LAYERS).eval()
		with torch.no_grad():
			logits = model(torch.randint(0, VOCAB, (2, SEQ)))
		assert logits.shape == (2, SEQ, VOCAB)


class TestNeurogenesisOnStandardArm:
	def test_net2wider_preserves_the_arm(self):
		from src.bitnet.growth.net2net import net2wider_model

		model = build_standard()
		wider = net2wider_model(model, new_hidden_dim=DIM * 2, noise_std=0.0)
		assert wider.hidden_dim == DIM * 2
		assert wider.onehot_vocab is True
		assert "vocab_embeddings" not in wider.state_dict()
		with torch.no_grad():
			logits = wider(torch.randint(0, VOCAB, (2, SEQ)))
		assert logits.shape == (2, SEQ, VOCAB)
