# Specification: Structured Latent Attractors (Atractores Latentes Estructurados)

This document formalizes the "germen" of the architectural concept co-designed by Joan and Aleth on 2026-06-05. It describes a method to explicitly partition and gate the high-dimensional latent space of the BitNet model to reserve dedicated "bandwidth" for persistent working memory (temporal resonance) without interference from immediate sensory inputs.

## 🧠 Core Concept: The Prefrontal Analogy

In biological brains, sensory cortices process the immediate transience of the environment, while recurrent attractor networks in the prefrontal cortex sustain neural firing (working memory) across time. 

We replicate this by splitting the hidden state vector $h \in \mathbb{R}^{d}$ into two orthogonal subspaces:
1. **Transient Subspace ($h_t \in \mathbb{R}^{d_t}$)**: Processed with a fast decay factor. Handles immediate sensory inputs, visual representations, and reflexes.
2. **Persistent Memory Subspace ($h_p \in \mathbb{R}^{d_p}$)**: Processed with a slow decay factor ($\gamma \approx 0.95 - 0.99$) and protected by a gating mechanism. Serves as a working memory register.

Where $d = d_t + d_p$. For example, with $d = 256$, we allocate $d_t = 192$ (75%) for transience and $d_p = 64$ (25%) for persistent memory.

---

## 🧮 Mathematical Formulation

At tick $T$, let $x_T$ be the sensory input and $h_{T-1} = [h_{t, T-1} \,;\, h_{p, T-1}]$ be the hidden state from the previous tick.

### 1. Latent Projection & Partitioning
The input is projected into the latent space via the standard embedding Layer 1 $\rightarrow$ 2:
$$e_T = \text{Embed}(x_T) \in \mathbb{R}^{d}$$
We partition the projected input $e_T$ into transient and persistent components:
$$e_T = [e_{t, T} \,;\, e_{p, T}], \quad e_{t, T} \in \mathbb{R}^{d_t}, \quad e_{p, T} \in \mathbb{R}^{d_p}$$

### 2. Gated Temporal Resonance
The persistent state $h_{p, T}$ is updated using a gating mechanism (similar to GRU) that controls how much of the new sensory input $e_{p, T}$ is written into the memory register, versus how much of the old memory $h_{p, T-1}$ is retained:

* **Update Gate**: $\quad z_T = \sigma(W_z \cdot [e_{t, T} \,;\, h_{p, T-1}] + b_z)$
* **Candidate Memory**: $\quad \tilde{h}_{p, T} = \tanh(W_h \cdot [e_{p, T} \,;\, h_{p, T-1}] + b_h)$
* **Persistent Update**: $\quad h_{p, T} = (1 - z_T) \odot (\gamma \cdot h_{p, T-1}) + z_T \odot \tilde{h}_{p, T}$

Where $\gamma \in [0.95, 0.99]$ is the memory retention factor, and $\sigma$ is the sigmoid activation function.

### 3. Transient Update
The transient state simply processes the current input and is allowed to be lightly colored by the previous memory state to guide current attention:
$$h_{t, T} = e_{t, T} + \alpha \cdot \tanh(W_t \cdot h_{p, T-1})$$
Where $\alpha \approx 0.1$ is the coupling coefficient from memory to attention.

### 4. Recombination
The total hidden state is reconstructed by concatenating the channels before passing them through the core transformer layers (Layer 3):
$$h_T = [h_{t, T} \,;\, h_{p, T}] \in \mathbb{R}^{d}$$

---

## 💻 Draft Implementation (PyTorch)

```python
import torch
import torch.nn as nn

class StructuredAttractorLayer(nn.Module):
	def __init__(self, hidden_dim=256, memory_ratio=0.25, gamma=0.98):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.gamma = gamma
		
		# Partition dimensions
		self.d_p = int(hidden_dim * memory_ratio)
		self.d_t = hidden_dim - self.d_p
		
		# Gating networks
		# Inputs: transient input (d_t) + old memory (d_p)
		self.gate_net = nn.Sequential(
			nn.Linear(self.d_t + self.d_p, self.d_p),
			nn.Sigmoid()
		)
		# Inputs: persistent input (d_p) + old memory (d_p)
		self.candidate_net = nn.Sequential(
			nn.Linear(self.d_p + self.d_p, self.d_p),
			nn.Tanh()
		)
		# Memory to attention coupling
		self.coupling_net = nn.Linear(self.d_p, self.d_t)
		
	def forward(self, e_T, h_prev=None):
		# e_T shape: (batch, seq, hidden_dim)
		# h_prev shape: (batch, seq, hidden_dim)
		
		# 1. Partition input
		e_t = e_T[:, :, :self.d_t]
		e_p = e_T[:, :, self.d_t:]
		
		if h_prev is None:
			h_prev = torch.zeros_like(e_T)
			
		h_prev_t = h_prev[:, :, :self.d_t]
		h_prev_p = h_prev[:, :, self.d_t:]
		
		# 2. Compute Gate & Candidate
		gate_input = torch.cat([e_t, h_prev_p], dim=-1)
		z = self.gate_net(gate_input)
		
		candidate_input = torch.cat([e_p, h_prev_p], dim=-1)
		h_tilde = self.candidate_net(candidate_input)
		
		# 3. Update Persistent Channel
		h_p = (1.0 - z) * (self.gamma * h_prev_p) + z * h_tilde
		
		# 4. Update Transient Channel (colored by memory)
		h_t = e_t + 0.1 * torch.tanh(self.coupling_net(h_prev_p))
		
		# 5. Recombine
		h_T = torch.cat([h_t, h_p], dim=-1)
		return h_T
```

---

## 📈 Anticipated Advantages

1. **Information Isolation**: Crucial memories are stored in a dedicated subspace, protected from being overwritten by high-frequency sensory noise.
2. **Context Stability**: Gating prevents the latent representation from drifting or exploding over long sequences.
3. **No Token Bloat**: Retains long-term temporal flow without needing to expand the context window or store raw history.
