# 🧬 Bit School - Lab Notebook

> Experimental log of Bit's Progressive Piagetian Curriculum.
> This notebook documents architectural milestones, diagnostic findings, and training runs.
> **Nothing is deleted. Failures are data.**

---

## 🏛️ Experimentation Directives

For every major training run and architectural pivot on the Bit visuo-semantic glyph model, we track:
1. **Core Architecture**: Hidden dimensions, layer count, positional embeddings, and vocabulary state.
2. **Diagnostic Breakthroughs**: Root causes of failures, thread scheduling bottlenecks, or memory leaks.
3. **Training Log**: Epochs, learning rates, loss behavior, and qualitative evaluation.

---

## 📔 Diagnostic Log & Lessons Learned

### Lesson 1: Out-of-Vocabulary (OOV) Semantic Collisions (2026-06-15)
* **Problem**: In early iterations (3,005-word vocabulary limit), out-of-vocabulary words in natural corpora (e.g., *"pajaritos"*, *"calcetines"*) were mapped to base words via direct sentence-transformer cosine similarity on single words. This caused silent, aberrant mappings (e.g., both words mapped to *"cálmate"*). The model learned to generalize noise.
* **Solution**: Scaled the base vocabulary limit to **15,005 words** in `expanded_glyphs.json` to cover the full spectrum of natural children's language in Spanish. Added **Rule 7** to `CONVENTIONS.md` to prohibit blind vector similarity mappings for OOV tokens.

### Lesson 2: GPU VRAM Spikes during Hot Neurogenesis (2026-06-16)
* **Problem**: During model mitosis/neurogenesis (e.g., Epoch 17, scaling from `dim: 128` to `256`), PyTorch retained the old model parameters, old optimizer states, and intermediate autograd graphs in VRAM. This triggered a `torch.cuda.OutOfMemoryError`, causing the training loop to fall back permanently to CPU training (100x slower).
* **Solution**: Implemented a CPU-offloading pipeline in `trigger_neurogenesis` inside `train_sovereign_school.py`. The old parameters and optimizer states are offloaded to CPU, CUDA cache is cleared, the Net2WiderNet mitosis is performed on CPU, and the expanded model is reloaded to GPU.

### Lesson 3: ONNX Runtime Thread Scheduling Overhead (2026-06-15)
* **Problem**: Invoking `embedding_model.embed` on single words in a loop caused severe context-switching overhead in ONNX Runtime, spiking CPU usage to 850% but running 100x slower.
* **Solution**: Implemented a caching dictionary lookup `self.mapping_cache` in `SovereignDictionary` to prevent redundant embedding calls for duplicate OOV words.

---

## 🚀 Chronological Experiment Log

### Experiment 001 — Piagetian Baseline (3k Vocab)
* **Date**: 2026-06-05 to 2026-06-15
* **Genotype**: 4-Layer BitNet, 6 layers, hidden_dim dynamically scaled from 128 to 1024, 3,005-word vocab.
* **Status**: ⚠️ SEMI-SUCCESSFUL (Passed 8-year exam, but lacked natural linguistic generalization due to OOV semantic collisions).
* **Diagnostics**: Memorized corrupted sentences instead of learning natural grammar.

### Experiment 002 — Scaled Vocab & Clean Corpus (task-1684)
* **Date**: 2026-06-16 (Active)
* **Genotype**: 4-Layer BitNet, 6 layers, hidden_dim scaling from 128 to 1024, 15,005-word vocab.
* **Corpus**: 62,345 preschool sequences (CHILDES MLU <= 20) + 12k Gutenberg school curriculum.
* **Status**: ⚡ RUNNING on GPU (98% GPU utilization, ~2.7 GB VRAM).
* **Progress**:
  * *Epoch 52*: Loss `5.2537` | Val Loss `5.4464` (Sample: *"yo sentir"* ➔ *"yo sentir asteroide"*).
  * *Epoch 53*: Loss `5.1649` | Val Loss `5.4662` (Sample: *"madre decir"* ➔ *"madre decir población"*).
