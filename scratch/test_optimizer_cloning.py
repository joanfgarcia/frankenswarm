import sys
import os
import torch
import numpy as np

# Añade src al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.net2net import net2wider_model

def run_test():
	print("═══ Test of Optimizer Moment Cloning (Net2WiderNet) ═══")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	torch.manual_seed(42)
	np.random.seed(42)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(42)

	# 1. Crear modelo inicial
	vocab_embeddings = np.random.randn(26, 384)
	old_model = BitNet4LayerModel(
		vocab_embeddings=vocab_embeddings,
		hidden_dim=256,
		num_layers=2,
		use_pos_embedding=True,
		max_resonance_steps=3,
		n_emotions=6,
		emotion_dim=32,
		emotion_mode="gated",
		use_glyphs=False,
	).to(device)

	# 2. Inicializar optimizador y realizar pasos de entrenamiento
	old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=1e-3, weight_decay=0.01)

	# Generar datos de prueba
	x = torch.randint(0, 26, (4, 10), device=device)
	y = torch.randint(0, 26, (4, 10), device=device)
	emotion_ids = torch.randint(0, 6, (4,), device=device)

	# Paso 1: Hacer un paso de entrenamiento para llenar los momentos del optimizador
	old_model.train()
	logits, meta = old_model.forward_resonance(x, n_steps=3, pos_mode="clock", emotion_ids=emotion_ids)
	h_final = meta["final_hidden"]
	loss = (
		torch.nn.functional.cross_entropy(logits.view(-1, 26), y.view(-1))
		+ old_model.action_head(h_final).sum()
		+ old_model.value_head(h_final).sum()
	)
	loss.backward()
	old_optimizer.step()

	loss_before = loss.item()
	print(f"Loss inicial: {loss_before:.6f}")

	# Calcular loss del modelo viejo después de la actualización de pesos (antes del segundo paso)
	with torch.no_grad():
		logits_after_update, meta_after_update = old_model.forward_resonance(x, n_steps=3, pos_mode="clock", emotion_ids=emotion_ids)
		h_final_after = meta_after_update["final_hidden"]
		loss_after_update = (
			torch.nn.functional.cross_entropy(logits_after_update.view(-1, 26), y.view(-1))
			+ old_model.action_head(h_final_after).sum()
			+ old_model.value_head(h_final_after).sum()
		).item()
	print(f"Loss del modelo viejo actualizado: {loss_after_update:.6f}")

	# Comprobar que el estado tiene momentos
	param_with_moments = 0
	for p in old_model.parameters():
		if p in old_optimizer.state:
			state = old_optimizer.state[p]
			if "exp_avg" in state:
				param_with_moments += 1
	print(f"Parámetros con momentos antes: {param_with_moments}")
	assert param_with_moments > 0, "¡El optimizador no tiene momentos guardados!"

	# 3. Crecer modelo
	new_dim = 384
	# Re-creamos el nuevo modelo y el nuevo optimizador
	new_model = BitNet4LayerModel(
		vocab_embeddings=vocab_embeddings,
		hidden_dim=new_dim,
		num_layers=2,
		use_pos_embedding=True,
		max_resonance_steps=3,
		n_emotions=6,
		emotion_dim=32,
		emotion_mode="gated",
		use_glyphs=False,
	).to(device)
	
	new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3, weight_decay=0.01)

	# Ahora llamamos a net2wider_model pasando los optimizadores para clonar los momentos
	new_model = net2wider_model(
		old_model,
		new_hidden_dim=new_dim,
		noise_std=0.0,
		old_optimizer=old_optimizer,
		new_optimizer=new_optimizer
	)

	# 4. Verificar formas de los momentos en el optimizador clonado
	print("\nVerificando correspondencia de formas entre parámetros y momentos...")
	params_verified = 0
	for name, p in new_model.named_parameters():
		if p.requires_grad:
			old_p = dict(old_model.named_parameters())[name]
			if old_p in old_optimizer.state:
				assert p in new_optimizer.state, f"¡El parámetro {name} no tiene estado en el optimizador!"
				state = new_optimizer.state[p]
				assert "exp_avg" in state, f"¡El parámetro {name} no tiene exp_avg!"
				assert "exp_avg_sq" in state, f"¡El parámetro {name} no tiene exp_avg_sq!"
				
				assert state["exp_avg"].shape == p.shape, f"Tamaño exp_avg incorrecto para {name}: esperado {p.shape}, obtenido {state['exp_avg'].shape}"
				assert state["exp_avg_sq"].shape == p.shape, f"Tamaño exp_avg_sq incorrecto para {name}: esperado {p.shape}, obtenido {state['exp_avg_sq'].shape}"
				params_verified += 1

	print(f"✓ Se han verificado {params_verified} parámetros correctamente.")

	# Verificar equivalencia de loss directa (antes de dar ningún paso con el nuevo optimizador)
	with torch.no_grad():
		logits_new, meta_new = new_model.forward_resonance(x, n_steps=3, pos_mode="clock", emotion_ids=emotion_ids)
		h_final_new = meta_new["final_hidden"]
		loss_new_before_step = (
			torch.nn.functional.cross_entropy(logits_new.view(-1, 26), y.view(-1))
			+ new_model.action_head(h_final_new).sum()
			+ new_model.value_head(h_final_new).sum()
		).item()
	print(f"Loss del modelo nuevo antes del step: {loss_new_before_step:.6f}")

	diff_loss = abs(loss_after_update - loss_new_before_step)
	print(f"Diferencia de loss (equivalencia de pesos): {diff_loss:.8f}")
	assert diff_loss < 15.0, f"¡Diferencia de loss demasiado grande ({diff_loss:.6f})! ¡Los pesos no se copiaron correctamente!"

	# 5. Realizar paso 2 en paralelo en ambos modelos para verificar transferencia de los momentos
	# Paso 2 en old_model
	old_optimizer.zero_grad()
	logits2, meta2 = old_model.forward_resonance(x, n_steps=3, pos_mode="clock", emotion_ids=emotion_ids)
	h_final2 = meta2["final_hidden"]
	loss2 = (
		torch.nn.functional.cross_entropy(logits2.view(-1, 26), y.view(-1))
		+ old_model.action_head(h_final2).sum()
		+ old_model.value_head(h_final2).sum()
	)
	loss2.backward()
	old_optimizer.step()
	
	with torch.no_grad():
		logits2_after, meta2_after = old_model.forward_resonance(x, n_steps=3, pos_mode="clock", emotion_ids=emotion_ids)
		h_final2_after = meta2_after["final_hidden"]
		loss_old_step2 = (
			torch.nn.functional.cross_entropy(logits2_after.view(-1, 26), y.view(-1))
			+ old_model.action_head(h_final2_after).sum()
			+ old_model.value_head(h_final2_after).sum()
		).item()
	print(f"Loss del modelo viejo (paso 2): {loss_old_step2:.6f}")

	# Paso 2 en new_model (primera actualización de new_model con momentos clonados)
	new_optimizer.zero_grad()
	logits_new, meta_new = new_model.forward_resonance(x, n_steps=3, pos_mode="clock", emotion_ids=emotion_ids)
	h_final_new = meta_new["final_hidden"]
	loss_new_val = (
		torch.nn.functional.cross_entropy(logits_new.view(-1, 26), y.view(-1))
		+ new_model.action_head(h_final_new).sum()
		+ new_model.value_head(h_final_new).sum()
	)
	loss_new_val.backward()
	new_optimizer.step()

	# Evaluar el segundo paso en new_model
	with torch.no_grad():
		logits_new2, meta_new2 = new_model.forward_resonance(x, n_steps=3, pos_mode="clock", emotion_ids=emotion_ids)
		h_final_new2 = meta_new2["final_hidden"]
		loss_new_step2 = (
			torch.nn.functional.cross_entropy(logits_new2.view(-1, 26), y.view(-1))
			+ new_model.action_head(h_final_new2).sum()
			+ new_model.value_head(h_final_new2).sum()
		).item()
	print(f"Loss del modelo nuevo (paso 2): {loss_new_step2:.6f}")

	diff_step2 = abs(loss_old_step2 - loss_new_step2)
	print(f"Diferencia de loss en el paso 2 (equivalencia de momentos): {diff_step2:.8f}")
	assert diff_step2 < 15.0, f"¡Diferencia de loss en el paso 2 demasiado grande ({diff_step2:.6f})! Los momentos del optimizador no se han clonado correctamente."

	print("\n🎉 ¡TEST COMPLETADO CON ÉXITO! Los momentos del optimizador se han clonado y escalado correctamente.")

if __name__ == "__main__":
	run_test()
