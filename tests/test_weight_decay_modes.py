"""Tests del trato de weight decay entre brazos (DL-007).

Con decay uniforme, AdamW aplica decay a TODAS las columnas de `inbound_proj`
en cada step — el gradiente es denso aunque solo unas pocas filas reciban señal —
así que el embedding de una palabra rara del brazo estándar se encoge hacia cero
entre sus actualizaciones infrecuentes. El brazo glifo no sufre eso: sus 65
primos se actualizan en cada batch. Estos tests fijan que existe un modo
simétrico y que el histórico sigue siendo el default (D3: no se cambia un
instrumento a mitad de comparativa).
"""

import numpy as np
import torch

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.training.modules.strategy import (
	WEIGHT_DECAY,
	BF16Strategy,
	FP32Strategy,
	build_param_groups,
	select_strategy,
)

VOCAB = 24
DIM = 16


def build_standard():
	torch.manual_seed(770)
	return BitNet4LayerModel(
		use_glyphs=False,
		vocab_embeddings=np.eye(VOCAB, dtype=np.float32),
		hidden_dim=DIM,
		num_layers=2,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=8,
	)


def build_glyph():
	torch.manual_seed(770)
	glyphs = np.random.RandomState(1).randint(-1, 2, size=(VOCAB, 65)).astype(np.float32)
	return BitNet4LayerModel(use_glyphs=True, glyph_table=glyphs, hidden_dim=DIM, num_layers=2, use_pos_embedding=True)


class TestParamGroups:
	def test_uniform_is_a_single_group(self):
		groups = build_param_groups(build_standard(), wd_mode="uniform")
		assert len(groups) == 1
		assert groups[0]["weight_decay"] == WEIGHT_DECAY

	def test_uniform_covers_every_parameter(self):
		model = build_standard()
		groups = build_param_groups(model, wd_mode="uniform")
		assert len(groups[0]["params"]) == len(list(model.parameters()))

	def test_no_embed_splits_in_two_groups(self):
		groups = build_param_groups(build_standard(), wd_mode="no_embed")
		assert len(groups) == 2
		assert groups[0]["weight_decay"] == WEIGHT_DECAY
		assert groups[1]["weight_decay"] == 0.0

	def test_no_embed_loses_no_parameter(self):
		model = build_standard()
		groups = build_param_groups(model, wd_mode="no_embed")
		grouped = sum(len(g["params"]) for g in groups)
		assert grouped == len([p for p in model.parameters() if p.requires_grad])

	def test_standard_arm_exempts_its_embedding_tables(self):
		model = build_standard()
		groups = build_param_groups(model, wd_mode="no_embed")
		exempt = {id(p) for p in groups[1]["params"]}
		assert id(model.inbound_proj.weight) in exempt
		assert id(model.outbound_proj.weight) in exempt
		assert id(model.pos_embedding) in exempt

	def test_standard_arm_still_decays_the_core(self):
		model = build_standard()
		groups = build_param_groups(model, wd_mode="no_embed")
		decayed = {id(p) for p in groups[0]["params"]}
		core_weight = model.core_layers[0].attn.q_proj.weight
		assert id(core_weight) in decayed

	def test_glyph_arm_exempts_its_primes(self):
		model = build_glyph()
		groups = build_param_groups(model, wd_mode="no_embed")
		exempt = {id(p) for p in groups[1]["params"]}
		assert id(model.glyph_embedding.prime_embeddings) in exempt


class TestOptimizerWiring:
	def test_default_mode_is_the_historical_one(self):
		"""D3: el instrumento por defecto no cambia bajo los pies de las réplicas."""
		assert FP32Strategy().wd_mode == "uniform"
		assert select_strategy("bf16", "off").wd_mode == "uniform"

	def test_select_strategy_propagates_the_mode(self):
		assert select_strategy("bf16", "off", wd_mode="no_embed").wd_mode == "no_embed"

	def test_optimizer_carries_the_exempt_group(self):
		model = build_standard()
		optimizer = BF16Strategy(wd_mode="no_embed").create_optimizer(model, lr_scale=1.0)
		decays = sorted(g["weight_decay"] for g in optimizer.param_groups)
		assert decays == [0.0, WEIGHT_DECAY]

	def test_uniform_optimizer_has_one_group(self):
		optimizer = BF16Strategy().create_optimizer(build_standard(), lr_scale=1.0)
		assert len(optimizer.param_groups) == 1
		assert optimizer.param_groups[0]["weight_decay"] == WEIGHT_DECAY


class TestRareWordDecayBehaviour:
	"""La razón de ser del modo: sin señal de gradiente, el decay uniforme encoge."""

	def _shrink_of_unused_column(self, wd_mode):
		model = build_standard()
		strategy = FP32Strategy(wd_mode=wd_mode)
		optimizer = strategy.create_optimizer(model, lr_scale=1.0)
		unused = 17
		before = model.inbound_proj.weight[:, unused].detach().clone()
		for _ in range(5):
			optimizer.zero_grad()
			# Solo el token 3 aparece: la columna 17 nunca recibe gradiente.
			model(torch.full((2, 8), 3, dtype=torch.long)).sum().backward()
			optimizer.step()
		after = model.inbound_proj.weight[:, unused].detach()
		return (before.abs().sum() - after.abs().sum()).item()

	def test_uniform_shrinks_the_unseen_word(self):
		assert self._shrink_of_unused_column("uniform") > 0

	def test_no_embed_leaves_the_unseen_word_alone(self):
		assert abs(self._shrink_of_unused_column("no_embed")) < 1e-9
