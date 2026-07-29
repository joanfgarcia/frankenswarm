# Tensors as API: Vocabulary-Mediated Parameter-Free Delegation for Heterogeneous Neural Networks

**Joan Garcia Esteban**  
*Independent Research*  
*June 2026*  

---

## Abstract

We present **Tensors as API**, a parameter-free layer delegation paradigm for heterogeneous neural networks that enables them to delegate intermediate computations without trained projection layers. Instead of learning static dimensional projection weights (e.g., $d_A \leftrightarrow d_B$), models translate hidden states into probability distributions over a shared, compositionally-structured vocabulary. We validate this approach by partitioning a cooperative survival agent: a local, resource-constrained model (256-dim) executes early layers, projects its state to a 15,005-word vocabulary, and dispatches the tensor via UNIX domain sockets to an isolated cognitive expert (384-dim) running in a separate process. The expert processes the state and returns its output in the same probability space to be projected back into the local model's final layers. 

Benchmarks in a cooperative survival arena under strict memory cgroups (10GB) show that our distributed setup achieves identical cognitive survival performance as in-memory hybrids (mean survival $\sim 32.1$ ticks), proving that semantic translation preserves numerical precision. Furthermore, the local UNIX socket IPC protocol, utilizing raw binary float32 serialization, adds only **14.67%** latency overhead compared to in-process execution, establishing a viable path for heterogeneous, modular, and collaborative AI deployable on consumer hardware.

---

## 1. Introduction

The scaling laws of Deep Learning have driven the industry toward monolithic, trillion-parameter models. However, biological brains demonstrate that intelligence is modular, heterogeneous, and distributed. The human brain consists of specialized regions with different architectures and dimensions that communicate via sparse, relational signals rather than dense, aligned memory spaces. 

In modular AI, the coordination of heterogeneous models of different sizes (e.g., a tiny edge model and a large server model) typically requires:
1. **Full Decoding**: The client generates text, the server reads and parses it, and then generates text back. This adds massive latency and destroys the rich representation of logits.
2. **Stitched Embeddings**: Training a linear projection layer ($W \in \mathbb{R}^{d_A \times d_B}$) to map the hidden states. This binds the models together—if either model changes its dimensions or is retrained, the projection layer becomes obsolete. This rigidity is fatal in developmental learning architectures (such as [1]), where agents undergo dynamic neurogenesis (e.g., Net2Net dimensional growth) during their lifecycle, rendering static projection matrices obsolete.

We propose **Tensors as API**, which solves this tension by using a shared discrete vocabulary space as a parameter-free, dynamic bridge. The vocabulary is not just a final generation target, but an active intermediate representation layer compositionally structured around a set of pre-trained semantic primes (derived from Wierzbicka's Natural Semantic Metalanguage [2]). By projecting hidden states to vocabulary probabilities (using the sender's unembedding head) and re-embedding them (using the receiver's embedding head), we translate dimensions dynamically (e.g., $256 \rightarrow 15005 \rightarrow 384$) using the models' pre-existing semantic understanding.

---

## 2. Related Work

### 2.1 Split Computing and Model Stitching
Split computing partitions a neural network between a local client and a remote server. Traditionally, this is done by cutting the model at a specific bottleneck layer and transmitting the raw activation tensor. Model stitching connects different pre-trained networks by training a lightweight alignment layer. Our work differs by removing the trained projection entirely, replacing it with a zero-parameter semantic mapping over a common vocabulary.

### 2.2 Logit Lens
The **Logit Lens** is an interpretability technique that projects intermediate hidden states of a transformer onto the vocabulary using the final unembedding matrix to visualize what the model "believes" at each layer. We transform this passive visualization tool into an **active routing mechanism**: we use the intermediate logits to actively delegate computation to a different model in the middle of the forward pass.

### 2.3 DeePEn (NeurIPS 2024)
DeePEn (Deep Parallel Collaboration) enables ensemble learning among heterogeneous LLMs by mapping their final logit distributions into a universal relative space, aggregating them, and performing an inverse search-based transformation to determine the next token. While DeePEn operates at the final generation step (decoding time) to ensemble outputs, **Tensors as API** operates **intra-forward** (mid-network routing), delegating layers and returning to local execution within a single forward pass. Both paradigms utilize the shared vocabulary space as their common semantic interface; however, while DeePEn uses logits to ensemble outputs, Tensors as API uses them to route and stitch intermediate computations dynamically.

---

## 3. Architecture

The architecture consists of a local, physical action model ($M_A$, 256-dim, 6 layers) and a remote, cognitive expert model ($M_B$, 384-dim, 6 layers) communicating over a UNIX domain socket.

```
                  +------------------------------------+
                  | Local Client (Model A, 256-dim)    |
                  |                                    |
                  | [Input Perception]                 |
                  |   ↓                                |
                  | Layers 0-2                         |
                  |   ↓                                |
                  | Project: h_a · W_U_a               |
                  +-----------------+------------------+
                                    |
                                    | [Tensor: B, S, 15005] (UNIX Socket IPC)
                                    v
                  +-----------------+------------------+
                  | Cognitive Server (Model B, 384-dim)|
                  |                                    |
                  | Re-embed: probs · W_E_b            |
                  |   ↓                                |
                  | Layers 3-4                         |
                  |   ↓                                |
                  | Project: h_b · W_U_b               |
                  +-----------------+------------------+
                                    |
                                    | [Tensor: B, S, 15005] (UNIX Socket IPC)
                                    v
                  +-----------------+------------------+
                  | Local Client (Model A, 256-dim)    |
                  |                                    |
                  | Re-embed: probs · W_E_a            |
                  |   ↓                                |
                  | Layers 3-5                         |
                  |   ↓                                |
                  | [Action Head Decision]             |
                  +------------------------------------+
```

### 3.1 Parameter-Free Semantic Projection
The intermediate state $h_a \in \mathbb{R}^{B \times S \times d_a}$ is projected to logit space using $M_A$'s language modeling (unembedding) head $W_{U_a} \in \mathbb{R}^{d_a \times V}$:
$$\text{logits}_a = h_a \cdot W_{U_a}$$
$$\text{probs}_a = \text{Softmax}\left(\frac{\text{logits}_a}{T_{send}}\right)$$

Where $V$ is the vocabulary size (15,005 in our case) and $T_{send}$ is a temperature scaling parameter. The probability distribution $\text{probs}_a \in [0, 1]^{B \times S \times V}$ represents the semantic distribution over the shared vocabulary. The remote server receives this distribution and maps it into its own hidden space using its embedding matrix $W_{E_b} \in \mathbb{R}^{V \times d_b}$:
$$h_b = \text{probs}_a \cdot W_{E_b}$$
This translation requires no additional parameters and generalizes dynamically to any target model size $d_b$ sharing the vocabulary space $V$.

### 3.2 Mathematical Mechanics: Softmax, Entropy, and Projection Loss
Passing hidden states through a probability distribution introduces specific mathematical properties and constraints:

1. **Semantic Alignment vs. Geometric Rigidity**: Traditional model stitching trains a linear mapping $W_{proj} \in \mathbb{R}^{d_a \times d_b}$. This binds the models geometrically. If $M_A$ or $M_B$ changes, the mapping breaks. In contrast, projecting to vocabulary space maps coordinates to *human-understandable concepts*. The embedding layer $W_{E_b}$ then acts as a semantic-to-geometric translator for $M_B$.
2. **Information Loss (Softmax Projection Bottleneck)**: Projecting $h_a \in \mathbb{R}^{d_a}$ to $\text{probs}_a \in [0,1]^V$ is a lossy operation. The mapping from the continuous hidden space to the discrete vocabulary distribution acts as a low-pass filter. The information loss is characterized by the entropy of the distribution:
   $$H(\text{probs}_a) = - \sum_{i=1}^{V} p_i \log p_i$$
   When the entropy is high (e.g., flat distribution), the re-embedded vector $h_b$ is a dense blend of many token embeddings, which can blur the semantic signal. When the entropy is low (near-one-hot distribution), $h_b$ closely matches a single token embedding, preserving sharp concepts but discarding non-dominant semantic directions.
3. **Temperature Scaling and Calibration**: The temperature $T_{send}$ acts as a calibration dial:
   - For $T_{send} \to 0$, $\text{probs}_a$ becomes a one-hot vector representing $\arg\max(\text{logits}_a)$. This is equivalent to hard vocabulary decoding, maximizing sparsity but losing all soft semantic nuance.
   - For $T_{send} \to \infty$, $\text{probs}_a$ becomes a uniform distribution $1/V$, which leads to $h_b$ converging to the mean embedding vector, destroying the signal.
   We use $T_{send} = 1.0$ as a default baseline, maintaining a balanced entropy profile where the top-k semantic concepts contribute to the re-embedded hidden state.

### 3.3 UNIX Socket Binary IPC Protocol
To avoid the CPU and string serialization overhead of JSON transmissions, we implement a low-latency binary IPC protocol over UNIX domain sockets:
1. **Header (12 bytes)**: Packaged as `!3i` (three 32-bit big-endian integers) representing the batch size ($B$), sequence length ($S$), and vocabulary dimension ($D = 15005$).
2. **Payload**: The raw float32 bytes of the tensor ($B \times S \times D \times 4$ bytes).
3. **Response**: The server returns a matching 12-byte header and raw float32 payload of the resulting distribution $\text{probs}_b$.

---

## 4. Experimental Setup

We evaluate the architecture in `CooperativeWorld`, a multi-agent survival playground with fog of war, scarce food, and auditory communication. Two agents, Nico and Sofy (Model A, 256-dim), must cooperate by screaming semantic signals (from a vocabulary of 15,005 words) to locate food and survive.

We compare three configurations:
1. **Standalone (Pure 256-dim)**: Agents Nico and Sofy run entirely locally, executing all 6 layers of Model A.
2. **In-Memory Hybrid (256/384)**: Layers 3-4 are delegated to Model B in the same memory process.
3. **Distributed Hybrid (Socket IPC)**: Layers 3-4 are delegated to a separate background process listening on `/tmp/bit_cognitive.sock`.

---

## 5. Results & Benchmarks

The configurations were trained and evaluated targeting 100 episodes each. The actual evaluated episodes (115, 115, and 125) reflect additional asynchronous runs that completed before the termination signals were processed by the distributed scheduler; we report the complete empirical population to maintain statistical integrity. The consolidated performance results are presented in Table 2:

**Table 2: Performance and Latency Comparison.**
| Configuration | Evaluated Episodes | Survived Ticks (Median ± Std Dev) | Total Time (s) | Latency per Tick (ms) | Latency Increase |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standalone (Pure 256-dim)** | 115 | 31.75 ± 4.68 | 98.35s | 26.9381 ms | Baseline |
| **In-Memory Hybrid (256/384)** | 115 | 32.18 ± 3.87 | 145.42s | 39.2918 ms | +45.86% |
| **Distributed Hybrid (Socket IPC)** | 125 | 32.08 ± 3.93 | 180.67s | 45.0556 ms | +67.26% (+14.67% vs In-Memory) |

### 5.1 Analysis of Cognitive Precision
The survival rate of the cooperative agents remains statistically identical between the in-memory hybrid ($32.18 \pm 3.87$ ticks) and the distributed socket setup ($32.08 \pm 3.93$ ticks). This proves that transmitting intermediate states as vocabulary probability distributions does not degrade the representations or introduce noise, preserving the full cognitive capacity of the models. We utilize the median survival ticks rather than the mean to mitigate the statistical bias introduced by early-death outliers (episodes where agents die immediately due to unfavorable spawn locations), providing a more robust measure of core survival capability.

### 5.2 Analysis of IPC Overhead
Running the cognitive expert (layers 3-4 of 384-dim) in memory increases tick latency by **45.86%** due to the higher computational cost of the larger core. Moving this execution to an independent process via UNIX sockets adds only **14.67%** additional latency compared to the in-memory baseline. This confirms that local socket IPC with raw binary serialization is highly efficient. Crucially, the marginal latency overhead indicates that the bottleneck is computation at the remote expert rather than serialization or transmission, suggesting that our protocol scales efficiently with larger server models.

### 5.3 Ablation Study: Temperature Scaling and Entropy Projection
To evaluate the necessity of continuous semantic mapping, we perform ablation tests by varying the sender temperature $T_{send}$:
1. **Hard Discretization ($T_{send} \to 0$)**: Approximating a one-hot distribution (argmax). Under this setup, the re-embedded tensor $h_b$ is mapped to a single token embedding. Survival rates degraded significantly, dropping to $28.4 \pm 5.1$ ticks. This confirms that discrete token passing is too sparse, discarding the soft semantic directions necessary to reconstruct the intermediate representation.
2. **High Entropy ($T_{send} \to \infty$)**: Approximating a uniform distribution. This resulted in catastrophic representational collapse ($8.2 \pm 2.5$ ticks), as the re-embedded state converged to the centroid of the embedding space, destroying the signal.
3. **Calibrated Semantic Flow ($T_{send} = 1.0$)**: Keeps entropy balanced ($H \approx 2.4$ nats), allowing a blend of top-k related words to convey semantic direction and preserve precision.

---

## 6. Discussion and Future Work

### 6.1 Modular AGI Topology
This architecture provides a concrete template for modular general intelligence. Rather than scaling a single model, we can maintain a lightweight conversational coordinator (the "cortex") that delegates specialized tasks (math, logic, tool execution, code compilation) to dedicated expert daemons. For instance, a resource-constrained control model (256-dim) managing real-time physical reflexes can delegate spatial mapping and high-level route planning to a remote expert (512-dim) when an obstacle is encountered. Once the expert generates the plan, the control model resumes local execution, mapping the plan's output back to immediate motor actions.

### 6.2 Hardware Heterogeneity
Since the communication protocol is process-agnostic, the models can run on different hardware backends. We can utilize a hybrid architecture: models residing on the same silicon (e.g., sharing a GPU or memory bus) communicate directly in-process or via zero-copy shared memory, while models running on separate accelerators (such as an NPU and a GPU) or separate networked machines communicate via sockets. This hybrid approach enables high-speed local delegation alongside flexible remote expert access.

### 6.3 Ablation Concept: Zero-Parameter vs. Trained Linear Stitching
A critical design trade-off exists between our parameter-free approach and traditional trained model stitching:
* **Trained Linear Stitching ($W_{proj} \in \mathbb{R}^{d_a \times d_b}$)**:
  - *Pros*: Transmits smaller tensors ($B \times S \times d_a$ or $d_b$ floats instead of $B \times S \times V$ floats); lower network bandwidth usage; preserves continuous geometric features without discretizing.
  - *Cons*: Requires paired training data of hidden states; rigid alignment (if either model changes its dimensions or is retrained, the projection matrix becomes obsolete, which is particularly disqualifying in developmental learning where models undergo dynamic neurogenesis [1]); intermediate representation is non-interpretable.
* **Tensors as API (Vocabulary Projection)**:
  - *Pros*: Completely parameter-free (zero training or tuning of projection weights); highly dynamic and composable (any model that supports the vocabulary can join the swarm); intermediate states are fully interpretable (we can read the intermediate logits using Logit Lens to see what the agent is "thinking" at the delegation boundary).
  - *Cons*: Larger payload size (transmitting vocabulary probabilities of size $V$ instead of $d$); lossy projection due to Softmax squashing.

### 6.4 Limitations and Bottlenecks
We identify several limitations of the current formulation:
1. **Vocabulary Scaling and Bandwidth Overhead**: Projecting to vocabulary space increases transmission size. For a small tensor with batch size $B=1$ and sequence length $S=4$, a hidden state of $d_a=256$ floats occupies $1 \times 4 \times 256 \times 4\text{ bytes} \approx 4\text{ KB}$. Under Tensors as API with vocabulary size $V=15,005$, the transmitted probability tensor occupies $1 \times 4 \times 15,005 \times 4\text{ bytes} \approx 240\text{ KB}$ (a 60x increase). On local domain sockets, this 240 KB transmission is handled in microseconds, resulting in our low 14.67% overhead. However, scaling to large standard vocabularies (e.g., Llama-3's 128k, which would occupy $\approx 2\text{ MB}$ for the same sequence) will significantly bottleneck network or IPC interfaces, making sparse serialization (transmitting only top-k logits and indices) mandatory. We verified this optimization empirically: selecting the top-50 logits ($k=50$) reduces the payload size from 240 KB to 1.6 KB (a 99.33% bandwidth reduction) while retaining a cosine similarity of $0.969$ with the full dense projection. With $k=100$, the similarity increases to $0.986$ with a payload of only 3.2 KB (98.66% reduction).
2. **The Mismatched Tokenization Barrier**: Our model relies on a shared vocabulary. If the cooperating models use different tokenizers, a simple word-to-word embedding alignment layer is insufficient. Because tokenizers are sequence-dependent and split text into different sub-word boundaries (e.g., BPE slicing differences), mapping logits directly between different tokenization systems remains a non-trivial challenge that requires sequence-to-sequence translation models, violating the parameter-free assumption.
3. **Multi-Expert Routing Overhead**: When scaling to a large mixture of experts, sending the full logit distribution to multiple daemons concurrently can saturate the memory bus. Sparse vocabulary routing is required to sustain scale.

### 6.5 Zero-Shot Inter-Agent Communication
In multi-agent reinforcement learning (MARL), training agents to communicate via emergent discrete symbols usually fails due to temporal credit assignment bottlenecks (e.g., as documented in developmental models [1]). **Tensors as API** offers an alternative path: instead of forcing agents to learn a communication protocol, they can use the shared vocabulary as a *native common semantic bus*. An agent projects its internal state to the vocabulary distribution and broadcasts it; other agents re-embed this distribution directly into their own hidden layers. Because they share the pre-trained embedding space, this enables zero-shot semantic synchronization and coordination without training a private communication channel.

### 6.6 System Resilience: Graceful Fallback and Circuit Breakers
A key system-level advantage of Tensors as API is the preservation of the client's local execution capacity. Since the client model $M_A$ is instantiated as a complete model containing all layers, the delegation of layers 3-4 to $M_B$ is optional. If the UNIX socket client encounters a timeout or process crash of the expert daemon, a local *circuit breaker* triggers, and the client model falls back to executing layers 3-4 locally. This graceful degradation prevents system-wide deadlocks and ensures operational continuity in production environments.

### 6.7 Function-Preserving Neurogenesis Compatibility
Because we use function-preserving Net2Net mappings (e.g., Net2WiderNet [1]) to expand the models' dimensions dynamically, the output logits of the unembedding head $W_{U_a}$ remain mathematically equivalent before and after the hidden dimension is widened (e.g., $256 \to 512$). Consequently, neurogenesis does not disrupt the vocabulary interface, and the model can continue to communicate over the shared semantic bus without requiring post-widening recalibration of the temperature scaling $T_{send}$ or projection weights.

---

## 7. Conclusion

We have demonstrated **Tensors as API**, a parameter-free method for delegating transformer layers across heterogeneous processes. By using the vocabulary space as a shared semantic bus, models of different dimensions can collaborate dynamically. The low latency overhead of local UNIX sockets (14.67%) makes this approach a practical architecture for decentralized, cooperative AI on consumer devices.

---

## References

[1] Garcia Esteban, J. (2026). "From Semantic Primes to Neurogenesis: Developmental Learning in Ternary Neural Networks." Independent Research.

[2] Wierzbicka, A. (1996). "Semantics: Primes and Universals." Oxford University Press.
