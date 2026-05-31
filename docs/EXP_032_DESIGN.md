# EXP_032 — Resonancia Continua: Bucle Latente 4→2

> **"Para que el modelo sea capaz de inferir de manera continua, hay que entrenarlo de manera continua."** — Joan Garcia

> [!IMPORTANT]
> Este es el experimento más ambicioso de Frankenswarm hasta la fecha. No es una iteración sobre el curriculum existente. Es un **cambio de topología del modelo**: conectar la Capa 4 directamente con la Capa 2, eliminando la discretización obligatoria durante el pensamiento.

---

## 1. Origen y Motivación

### 1.1 La Tesis

Documento original: [`tesis_resonancia_continua.md`](file:///home/joan/Documents/Obsidian%20Vault/tesis_resonancia_continua.md)

**Autores**: Joan Garcia (concepto original) · Titanium (formalización técnica) · Aleth (diseño experimental)

**Idea central**: En la arquitectura actual de Frankenswarm, cada paso de "pensamiento" obliga al modelo a colapsar su estado continuo (256-dim) en un token discreto (8192-dim), solo para volver a expandirlo inmediatamente. Es como si un humano tuviera que hablar en voz alta cada sílaba para poder seguir pensando.

La propuesta: **conectar la salida de la Capa 4 (FFN) directamente a la entrada de la Capa 2 (Embedding)**, creando un bucle cerrado en el espacio latente donde el modelo pueda "pensar sin hablar".

### 1.2 Evidencia empírica previa

El EXP_029 (self-loop) encadenaba 2 pasos de razonamiento, pero **pasaba por vocabulario completo entre cada paso**:

```
Step 1: Core(256) → outbound(384) → logits(8192) → Gumbel-Softmax
                                                          ↓
Step 2: Gumbel(8192) → matmul(vocab, 384) → inbound(256) → Core(256) → logits
```

**Telemetría EXP_029** (60 epochs, autonomía):
- `acc_concept` (paso intermedio): **~100%** — el modelo aprende el eslabón A→B perfectamente
- `acc_joint` (resultado final, 2 pasos): **~70%** — pierde un 30% al encadenar

Ese gap del 30% es el **coste del roundtrip por vocabulario**. La señal se degrada al pasar por la discretización Gumbel-Softmax y la re-proyección a embedding. Si el bucle latente elimina esa ruta, la degradación debería desaparecer.

### 1.3 La ventaja BitNet

> [!TIP]
> **¿Por qué esto puede funcionar aquí y no funcionó para DeepMind (Universal Transformers)?**
>
> Los pesos de BitNet son **ternarios (-1, 0, +1)**. Al no haber precisión flotante que se diluya al iterar sobre sí misma, la tensión matemática puede mantenerse estable mecánicamente. Es el equivalente a un circuito digital con retroalimentación: la señal no se degrada porque cada componente es discreto (Dinámica de Atractores pura).
>
> En redes con pesos float32, el bucle sufre Vanishing/Exploding Gradients. Con pesos ternarios, el gradiente está acotado por construcción.

---

## 2. Hipótesis

### H₀ — Estabilidad (Pre-condición)

> En una red BitNet con pesos ternarios (-1, 0, +1), el hidden state (256-dim) se mantiene estable al iterar el core_layers N veces sin pasar por la capa de vocabulario.

**Criterio**: `L2_norm(h[step_N]) / L2_norm(h[step_0])` se mantiene en el rango [0.5, 2.0] para N ≤ 10.

### H₁ — Rendimiento

> Un modelo entrenado con resonancia latente (bucle 4→2 cerrado) supera el 70% de `acc_joint` del EXP_029 en cadenas lógicas de 2+ pasos.

### H₂ — Convergencia de Atractores

> El hidden state del modelo entrenado con resonancia converge a un punto fijo (atractor) tras N iteraciones, medido por `cosine_sim(h[i], h[i-1]) > 0.95`.

---

## 3. Diseño Experimental: Malla Factorial 3³

> *"No podemos descartar ninguna combinación porque la solución, si la hay, puede estar en una combinación exótica impensable."* — Joan Garcia

En lugar de elegir parámetros a priori, ejecutamos **27 variantes** que cubren todas las combinaciones de los 3 ejes de incertidumbre.

### 3.1 Eje A — Positional Embeddings en el bucle

| Variante | Código | Descripción |
|---|---|---|
| `pos_none` | Sin pos_embedding en el bucle | El modelo no sabe en qué ciclo de resonancia está |
| `pos_entry` | Solo al entrar | Se suman una vez antes del bucle, luego opera sin referencia temporal |
| `pos_clock` | Re-sumados en cada iteración | Un "reloj" de resonancia: embedding posicional por step que le dice al modelo en qué ciclo está |

**Implementación de `pos_clock`**: Se añade un `nn.Parameter` de forma `(1, max_resonance_steps, hidden_dim)` que se suma al hidden state en cada iteración del bucle. Es una segunda dimensión posicional ortogonal a la posicional de secuencia.

### 3.2 Eje B — Profundidad de resonancia (n_steps)

| Variante | Código | Descripción |
|---|---|---|
| `depth_2` | n_steps=2 fijo | Mínima resonancia: el modelo tiene exactamente 2 pasos internos |
| `depth_3` | n_steps=3 fijo | Resonancia media: un paso por eslabón lógico en la cadena A→B→C |
| `depth_ramp` | n_steps=1→5 progresivo | Curriculum de profundidad: guardería=1, recreo=2→3, autonomía=3→5 |

### 3.3 Eje C — Estrategia de loss

| Variante | Código | Descripción |
|---|---|---|
| `loss_final` | Loss solo al paso final | **Resonancia Aislada pura** (como describe la tesis): la red decide libremente qué hacer en los pasos internos. Solo se evalúa el resultado final |
| `loss_every` | Loss en cada paso | Supervisión total: cada paso del bucle debe producir el token intermedio correcto. Fuerza una "verbalización interna" |
| `loss_weighted` | 0.2 × intermedia + 1.0 × final | Guía ligera en pasos intermedios, peso fuerte en resultado final. Compromiso entre libertad y supervisión |

### 3.4 Nomenclatura

`EXP_032_{pos}_{depth}_{loss}`

Ejemplo: `EXP_032_clock_3_final` = reloj posicional + 3 pasos fijos + loss solo al final

### 3.5 Malla completa (27 variantes)

| # | pos | depth | loss | ID |
|---|---|---|---|---|
| 1 | none | 2 | final | `EXP_032_none_2_final` |
| 2 | none | 2 | every | `EXP_032_none_2_every` |
| 3 | none | 2 | weighted | `EXP_032_none_2_weighted` |
| 4 | none | 3 | final | `EXP_032_none_3_final` |
| 5 | none | 3 | every | `EXP_032_none_3_every` |
| 6 | none | 3 | weighted | `EXP_032_none_3_weighted` |
| 7 | none | ramp | final | `EXP_032_none_ramp_final` |
| 8 | none | ramp | every | `EXP_032_none_ramp_every` |
| 9 | none | ramp | weighted | `EXP_032_none_ramp_weighted` |
| 10 | entry | 2 | final | `EXP_032_entry_2_final` |
| 11 | entry | 2 | every | `EXP_032_entry_2_every` |
| 12 | entry | 2 | weighted | `EXP_032_entry_2_weighted` |
| 13 | entry | 3 | final | `EXP_032_entry_3_final` |
| 14 | entry | 3 | every | `EXP_032_entry_3_every` |
| 15 | entry | 3 | weighted | `EXP_032_entry_3_weighted` |
| 16 | entry | ramp | final | `EXP_032_entry_ramp_final` |
| 17 | entry | ramp | every | `EXP_032_entry_ramp_every` |
| 18 | entry | ramp | weighted | `EXP_032_entry_ramp_weighted` |
| 19 | clock | 2 | final | `EXP_032_clock_2_final` |
| 20 | clock | 2 | every | `EXP_032_clock_2_every` |
| 21 | clock | 2 | weighted | `EXP_032_clock_2_weighted` |
| 22 | clock | 3 | final | `EXP_032_clock_3_final` |
| 23 | clock | 3 | every | `EXP_032_clock_3_every` |
| 24 | clock | 3 | weighted | `EXP_032_clock_3_weighted` |
| 25 | clock | ramp | final | `EXP_032_clock_ramp_final` |
| 26 | clock | ramp | every | `EXP_032_clock_ramp_every` |
| 27 | clock | ramp | weighted | `EXP_032_clock_ramp_weighted` |

---

## 4. Arquitectura del Bucle Latente

### 4.1 Topología — Forward Estándar vs Forward Resonance

**Forward actual** (usado en EXP_001 a EXP_031):
```
input → [Capa1: embed(8192→384)] → [Capa2: inbound(384→256)]
      → [Capa3: core_layers(256→256)] → [Capa4: outbound(256→384)]
      → [Capa5: dot(vocab.T) → logits(8192)]
```

**Forward Resonance** (EXP_032):
```
input → [Capa1→2: una sola vez] → h(256)
                                    │
                  ┌─────────────────┤
                  │  core_layers    │
                  │  + norm         │ × n_steps
                  │  + clock (opt)  │
                  └─────────────────┘
                          │
                  [Capa4→5: solo al final] → logits(8192)
```

### 4.2 El Watcher (Osciloscopio)

Como describe la tesis:

> *"La Capa 5 se puede usar como un 'Osciloscopio' asíncrono. A intervalos regulares, se extrae una muestra del vector continuo hacia la Capa 5 para que lo decodifique."*

El Watcher es una función que, **sin romper el flujo del bucle** (`torch.no_grad()`), proyecta el hidden state a tokens para que podamos "espiar" qué está pensando el modelo en cada paso de resonancia:

```python
def _sample_watcher(self, h):
    """Espía qué token 'piensa' el modelo sin afectar al bucle."""
    with torch.no_grad():
        concept_proj = self.outbound_proj(h)                    # 256 → 384
        logits = torch.matmul(concept_proj, self.vocab_embeddings.T)  # 384 → 8192
        tokens = logits.argmax(dim=-1)                          # Token top-1
        entropy = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)
    return {"tokens": tokens, "entropy": entropy}
```

El Watcher **no tiene gradiente** — es pura lectura. No altera los pesos ni el hidden state.

### 4.3 Métricas de estabilidad

En cada paso del bucle, registramos:

| Métrica | Fórmula | Qué mide |
|---|---|---|
| `norm_ratio` | `‖h[i]‖₂ / ‖h[0]‖₂` | ¿El vector explota (>>1) o colapsa (<<1)? |
| `cosine_convergence` | `cos(h[i], h[i-1])` | ¿El hidden state converge a un atractor fijo? |
| `watcher_consistency` | `token[i] == token[i-1]` | ¿El "pensamiento decodificado" se estabiliza? |
| `watcher_entropy` | `H(softmax(logits[i]))` | ¿El modelo está seguro de su pensamiento o hay ruido? |

---

## 5. Protocolo de Ejecución

### Fase 0 — Validación de Estabilidad (GO/NO-GO)

**Duración estimada**: ~1 minuto
**GPU**: No entrena — solo forward pass

1. Cargar checkpoint existente (`EXP_028A/best_agent.pt`)
2. Ejecutar `forward_resonance()` con N = 1, 2, 3, 5, 10, 20 pasos
3. Para cada N, medir `norm_ratio` y `cosine_convergence`
4. **Criterio GO**: `norm_ratio ∈ [0.5, 2.0]` para N ≤ 10
5. **Criterio NO-GO**: Si el vector explota a N=2, investigar estabilización antes de continuar

> [!CAUTION]
> Si H₀ falla, todo el experimento se detiene aquí. No tiene sentido entrenar 27 variantes si el hidden state explota tras 2 iteraciones.

### Fase 1 — Grid Runner (27 experimentos)

**Duración estimada**: ~3-4 horas (5-10 min/experimento)
**GPU**: RTX 5070 Ti (VRAM ~16GB)
**OOM Shield**: Cada experimento envuelto con `systemd-run --user --scope -p MemoryMax=10G`

**Parámetros constantes** (invariantes en las 27 variantes):

| Parámetro | Valor | Justificación |
|---|---|---|
| `hidden_dim` | 256 | Consistente con toda la línea experimental |
| `num_layers` | 3 | Mismo que EXP_029-031 |
| `vocab_size` | 8192 | SovereignTranslator estándar |
| `embed_dim` | 384 | fastembed MiniLM-L6-v2 |
| `num_heads` | 4 | head_dim = 64 |
| `epochs` | 200 | Tiempo suficiente para convergencia |
| `batch_size` | 32 | Estándar del lab |
| `lr` | 3e-4 | Ajustado respecto a EXP_029 (1e-3 era agresivo) |
| `grad_clip` | 1.0 | Protección anti-explosión |
| `nursery_end` | 30 | 30 epochs de teacher forcing |
| `transition_end` | 80 | 50 epochs de transición |
| `tf_min` | 0.1 | 10% de TF residual (como EXP_008+) |
| `tau_start → tau_min` | 1.0 → 0.3 | Consistente con curriculum |
| `watcher_interval` | 10 | Muestra del watcher cada 10 epochs |

**Ejecución**: Secuencial (una variante tras otra). Cada variante genera:
- `storage/experiments/EXP_032_{id}/config.json`
- `storage/experiments/EXP_032_{id}/telemetry.jsonl`
- `storage/experiments/EXP_032_{id}/best_agent.pt`
- `storage/experiments/EXP_032_{id}/watcher_log.jsonl`
- `storage/experiments/EXP_032_{id}/stability_metrics.jsonl`

### Fase 2 — Análisis comparativo

**Script**: `analyze_grid_032.py`

Genera automáticamente:
1. **Tabla resumen** de las 27 variantes (acc_joint, norm_stability, attractor_convergence)
2. **Heatmap 3×3×3** de `acc_joint_final` por combinación de ejes
3. **Curvas de trajectory_norms** para top-3 y bottom-3 variantes
4. **Watcher timeline** del ganador: evolución token-a-token del "pensamiento"
5. **Comparación vs baseline**: Ganador vs EXP_029 (70% acc_joint)

---

## 6. Archivos a crear/modificar

### Modelo

| Archivo | Acción | Descripción |
|---|---|---|
| `src/bitnet/modeling_bitnet.py` | MODIFICAR | Añadir `forward_resonance()`, `_sample_watcher()`, `resonance_clock` |

### Entrenamiento y ejecución

| Archivo | Acción | Descripción |
|---|---|---|
| `src/bitnet/train_resonance.py` | NUEVO | Script de entrenamiento con bucle latente cerrado |
| `src/bitnet/run_grid_032.py` | NUEVO | Grid runner: genera 27 configs, ejecuta secuencialmente |
| `src/bitnet/stability_check_032.py` | NUEVO | Fase 0: test de estabilidad sin entrenamiento |

### Inferencia y análisis

| Archivo | Acción | Descripción |
|---|---|---|
| `src/bitnet/inference_resonance.py` | NUEVO | Inferencia con bucle latente + watcher visual |
| `scripts/analyze_grid_032.py` | NUEVO | Análisis comparativo de las 27 variantes |

### Configuración

| Archivo | Acción | Descripción |
|---|---|---|
| `configs/experiments/EXP_032_TEMPLATE.json` | NUEVO | Template base para el grid runner |

---

## 7. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| **Hidden state explota** (H₀ falla) | Media | Bloqueante | Fase 0 detecta esto en 1 min. Si falla: investigar LayerNorm adicional, gradient scaling |
| **Hidden state colapsa a cero** | Baja | Bloqueante | Residual connections del transformer lo protegen. Medir en Fase 0 |
| **Ninguna variante supera EXP_029** | Media | No bloqueante | El resultado "el bucle no ayuda" es ciencia válida. Publicable como resultado negativo |
| **BPTT a profundidad N>3 agota VRAM** | Baja | Parcial | Con hidden_dim=256 y batch=32, la huella de N=5 steps es ~50MB. Margen de sobra |
| **Catastrophic forgetting al cambiar topología** | Media | Parcial | Cada variante entrena desde cero (no desde checkpoint previo) |

---

## 8. Criterios de éxito

### Éxito mínimo (resultado publicable)
- [ ] H₀ validada: el hidden state ternario es estable al iterar N ≤ 10
- [ ] Al menos 1 variante supera el 70% de `acc_joint` de EXP_029

### Éxito esperado
- [ ] Top-3 variantes superan el 80% de `acc_joint`
- [ ] Se identifica claramente qué eje (pos/depth/loss) tiene más impacto
- [ ] El watcher muestra convergencia visible del "pensamiento"

### Éxito excepcional
- [ ] Al menos 1 variante alcanza >90% de `acc_joint` en cadenas de 3+ pasos
- [ ] El modelo generaliza a profundidades de resonancia no vistas en entrenamiento
- [ ] Se observan atractores estables (cosine_sim > 0.99 tras N iteraciones)

---

## 9. Referencias

### Internas
- [Tesis de Resonancia Continua](file:///home/joan/Documents/Obsidian%20Vault/tesis_resonancia_continua.md) — Documento fundacional
- [EXP_029 (train_loop.py)](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/train_loop.py) — Baseline de cadena por vocabulario
- [modeling_bitnet.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/modeling_bitnet.py) — Modelo a modificar
- [LAB_NOTEBOOK.md](file:///home/joan/Documents/IA/frankenswarm/docs/LAB_NOTEBOOK.md) — Registro histórico

### Estado del arte
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence* — JEPA, predicción en espacio latente
- Dehghani, M. et al. (2019). *Universal Transformers* — Recurrencia adaptativa en transformers
- Banino, A. et al. (2021). *PonderNet: Learning to Ponder* — Computación adaptativa, halt probability
- Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural Networks* — ACT
- Ma, X. et al. (2023). *BitNet: Scaling 1-Bit Transformers* — Pesos ternarios en transformers
