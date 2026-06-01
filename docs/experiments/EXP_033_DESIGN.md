# EXP_033 — Resonancia Emocional: La Emoción como Brújula del Pensamiento

**Fecha de diseño**: 2026-05-30
**Dependencia**: EXP_032 (Resonancia Continua — bucle latente 4→2)
**Origen teórico**: Joan Garcia — "la emoción es la que toma las decisiones"
**Estado**: 📐 DISEÑADO — Pendiente de resultados de EXP_032

---

## 1. Motivación

EXP_032 demuestra que un modelo BitNet puede iterar sus core_layers en bucle
cerrado sin explotar. Pero durante ese bucle, el modelo piensa en el vacío.
La señal emocional (fear_amplifier) solo se aplica a la loss, **después** del
pensamiento. Es un castigo, no una brújula.

La hipótesis de Joan: la emoción no evalúa el resultado — **participa en el
razonamiento**. El mismo input con emociones distintas produce trayectorias
distintas por el espacio latente y, por tanto, decisiones distintas.

### Analogía biológica

La amígdala no espera a que el córtex prefrontal termine de razonar para
evaluar el resultado. Inyecta señal **durante** el procesamiento cortical:
- Modula la atención (qué representaciones se amplifican)
- Sesga las decisiones (fight/flight antes de "pensar")
- Crea atajos (respuestas emocionales rápidas vs razonamiento lento)

### Estado del arte

En IA, las emociones se tratan como:
- Etiquetas de clasificación (sentiment analysis)
- Rewards en RL (RLHF)
- Tokens en el prompt ("respond happily")

Nadie inyecta emoción **dentro** del forward pass como modulación latente
de la trayectoria de razonamiento. Este es el gap.

---

## 2. Hipótesis

**H₁**: Inyectar un vector emocional en cada iteración del bucle latente
produce atractores distintos para el mismo input lógico, mejorando la
accuracy en tareas donde la emoción es informativa.

**H₂**: La modulación emocional interna supera al fear_amplifier externo
(loss weighting) como mecanismo de decisión.

**H₃**: Las emociones negativas (miedo, dolor) producen trayectorias más
cortas (convergencia rápida — "decisión visceral"), mientras que las
neutras/positivas producen trayectorias más largas (exploración).

---

## 3. Arquitectura

### 3.1 Cambio al modelo

```python
class BitNet4LayerModel:
    def __init__(self, ..., emotion_dim: int = 0, n_emotions: int = 6):
        # Nuevo: embedding de emociones → espacio oculto
        if emotion_dim > 0:
            self.emotion_embeddings = nn.Embedding(n_emotions, emotion_dim)
            self.emotion_proj = nn.Linear(emotion_dim, hidden_dim, bias=False)
        
    def forward_resonance(self, x, n_steps, emotion_ids=None, ...):
        h = self._embed_input(x)
        
        # Proyectar emoción al espacio latente
        if emotion_ids is not None:
            emo = self.emotion_proj(self.emotion_embeddings(emotion_ids))
            # emo: (batch, 1, hidden_dim) — broadcast sobre seq_len
        
        for step in range(n_steps):
            if emotion_ids is not None:
                h = h + emo  # La emoción desvía cada paso
            
            for layer in self.core_layers:
                h = layer(h)
            h = self.norm(h)
        
        return self._decode_hidden(h)
```

### 3.2 Emociones disponibles (del LORE existente)

| ID | Emoción   | Tipo     | Efecto esperado en la trayectoria |
|----|-----------|----------|-----------------------------------|
| 0  | miedo     | negativa | Convergencia rápida, atractores defensivos |
| 1  | alegría   | positiva | Exploración, atractores variados |
| 2  | ira       | negativa | Convergencia rápida, atractores agresivos |
| 3  | tristeza  | negativa | Convergencia lenta, atractores pasivos |
| 4  | dolor     | negativa | Convergencia muy rápida, evitación |
| 5  | hambre    | drive    | Sesgo hacia atractores de recurso |

### 3.3 Topología completa

```
input → [Capa1→2] → h(256)
                       │
     ┌────────────────┤
     │  + emo(256)     │
     │  core_layers    │ × n_steps
     │  + norm         │
     │  + clock (opt)  │
     └────────────────┘
            │
      [Capa4→5] → logits(8192)
```

La emoción entra SUMADA al hidden state en cada iteración.
No es un gate multiplicativo (eso lo inhibiría). Es aditiva:
empuja el vector hacia una región del espacio, sin bloquearlo.

### 3.4 Variantes a explorar

**Eje D — Modo de inyección emocional:**
- `additive`: h = h + emo (la emoción desplaza)
- `gated`: h = h * σ(emo) + emo (la emoción filtra y desplaza)
- `first_only`: h = h + emo solo en step 0 (impulso inicial)

---

## 4. Diseño experimental

### 4.1 Condiciones (A/B/C)

| Condición | Resonancia | Emoción interna | Fear loss | Descripción |
|-----------|-----------|-----------------|-----------|-------------|
| **A** (baseline) | ❌ forward estándar | ❌ | ❌ | EXP_029 clásico |
| **B** (resonancia) | ✅ bucle latente | ❌ | ❌ | EXP_032 puro |
| **C** (resonancia + fear loss) | ✅ | ❌ | ✅ | EXP_032 con fear_amplifier externo |
| **D** (resonancia + emoción interna) | ✅ | ✅ | ❌ | **EXP_033: la emoción decide** |
| **E** (todo) | ✅ | ✅ | ✅ | Resonancia + emoción + fear loss |

La comparación clave:
- **B vs A**: ¿pensar en silencio > pensar en voz alta?
- **D vs B**: ¿la emoción interna mejora el pensamiento?
- **D vs C**: ¿emoción como brújula > emoción como castigo?
- **E vs D**: ¿se complementan o son redundantes?

### 4.2 Dataset

Mismo grafo causal de EXP_032 (12 conceptos, 13 reglas), pero cada
cadena se presenta con DIFERENTES emociones:

```
Input: [luna, implica, ?, miedo]    → ¿agua? (la luna da miedo → agua)
Input: [luna, implica, ?, alegría]  → ¿agua? (la luna da alegría → agua)
Input: [fuego, implica, ?, miedo]   → ¿peligro? (fuego + miedo → peligro RÁPIDO)
Input: [fuego, implica, ?, alegría] → ¿tierra? (fuego + alegría → tierra/ceniza)
```

La respuesta lógica es la misma, pero la emoción puede:
1. Acelerar la convergencia (llegar a la respuesta en menos pasos)
2. Cambiar la prioridad (ante bifurcaciones, elegir el camino "emocional")
3. Modular la confianza (entropía del watcher más baja = más seguro)

### 4.3 Métricas nuevas

| Métrica | Qué mide |
|---------|----------|
| `traj_length_by_emotion` | ¿Las emociones negativas convergen más rápido? (H₃) |
| `attractor_diversity` | ¿Emociones distintas → atractores distintos? (H₁) |
| `acc_by_emotion` | ¿Alguna emoción mejora el razonamiento? |
| `watcher_entropy_by_emotion` | ¿La emoción reduce la incertidumbre? |
| `emotion_trajectory_cosine` | Similitud entre trayectorias con misma lógica, distinta emoción |

---

## 5. Implementación

### Archivos a crear/modificar

#### [MODIFY] `src/bitnet/modeling_bitnet.py`
- Añadir `emotion_embeddings` y `emotion_proj` al `__init__`
- Añadir parámetro `emotion_ids` a `forward_resonance()`
- Inyectar emo en el bucle latente

#### [NEW] `src/bitnet/train_emotional_resonance.py`
- Fork de `train_resonance.py`
- Dataset con emociones por cadena
- Métricas de diversidad de atractores

#### [NEW] `scripts/compare_conditions.py`
- Comparación A/B/C/D/E en un solo informe
- Gráficos de trayectorias por emoción

---

## 6. Riesgos

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| La emoción domina y el modelo ignora la lógica | Media | Escalar emo con factor < 1.0 |
| Todas las emociones producen el mismo atractor | Media | Medir `attractor_diversity` |
| El embedding emocional colapsa (todos iguales) | Baja | Regularizar con triplet loss |
| La señal emocional desestabiliza el bucle | Baja | Ya validado en Fase 0 que sumas al hidden son estables |

---

## 7. Criterios de éxito

- **Mínimo**: D > B en acc_joint (la emoción interna mejora al bucle puro)
- **Esperado**: D > C (brújula > castigo) y H₃ validada (trayectorias distintas)
- **Excepcional**: El modelo ante una bifurcación elige el camino coherente
  con la emoción, sin haberlo visto en entrenamiento (generalización emocional)

---

## 8. Relación con la tesis

> "La emoción no evalúa el pensamiento. La emoción ES parte del pensamiento."
> — Joan Garcia, 2026-05-30

EXP_032 demuestra que el sustrato ternario soporta recurrencia estable.
EXP_033 demuestra que la emoción puede modular esa recurrencia de forma
informativa, creando un sistema que no solo piensa en silencio, sino que
**siente mientras piensa**.

La combinación de:
1. Recurrencia latente (pensar sin hablar)
2. Estabilización ternaria (BitNet como sustrato natural)
3. Modulación emocional interna (la emoción como brújula)

...no existe en la literatura. Es la contribución original.
