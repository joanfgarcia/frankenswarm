# RFC — Brazos Resonantes: Bit con Bucle Latente y Emoción (DL-010)

> **Estado**: IMPLEMENTADO (30-ago-2026) · **Autor**: Aleth + Joan García
> **Contexto**: BIT-003 (v1-inglés) + K-65P · comparativa de tesis "normal vs resonante"
> **Decisión registrada en**: `docs/DECISION_LOG.md` (DL-010)
> **Origen teórico**: `tesis_resonancia_continua.md` (Joan Garcia + Titanium) · EXP_032/033/034/078

---

## 1. Resumen ejecutivo

Bit se diseñó para pensar con **resonancia**: un bucle latente cerrado donde el
hidden state itera N veces por el core SIN colapsar a vocabulario. La escuela
lo entrenaba **apagado** (`max_resonance_steps=0`). Este RFC activa el brazo
resonante como variable experimental de la tesis, con la configuración ganadora
de los experimentos fundacionales: **resonancia + emoción `first_only`** —
porque la resonancia sola es nula (B≈A en EXP_033 y EXP_034) y solo con la
emoción como brújula alcanza el 100% (EXP_034, época 12, 7.7× más rápido).

**Propiedad distintiva**: con rampa de profundidad en entrenamiento, el número
de pasos es un **knob libre en inferencia** — "pensar más profundo" sin
reentrenar. Un modelo de profundidad fija no puede ofrecer esto.

## 2. Genealogía (por qué esto no es un extra, es la raíz)

| Experimento | Resultado clave |
|---|---|
| EXP_033 (9 variantes) | B (resonancia sola) ≈ A (baseline): **+0.0 pp**. D (resonancia + emoción `first_only`): **94.6%** peak |
| EXP_034 (glifos) | D_first_only: **100.0% @ época 12**; baseline clavado en 28.4% · **este experimento originó K-65P** (RFC-001/002) |
| EXP_078 (sueño BPTT) | La resonancia debe entrenarse **desde cero** (el `resonance_clock` se ajusta óptimo desde la primera época); pesos pre-entrenados estáticos rinden peor |

**Incoherencia genealógica corregida**: K-65P nació de una configuración
resonante-emocional que la escuela luego entrenaba apagada.

## 3. Mecanismo (`forward_resonance`, ya existente en el modelo)

```
input → [Capa1→2: una vez] → h(latente)
          ┌─────────────────┤
          │ clock (fase)    │ × n_steps
          │ emoción (step 0)│   ← first_only: un impulso desvía toda la trayectoria
          │ core_layers     │
          └─────────────────┘
          → [Capa4→5: solo al final] → logits (gateo de etapa aplicado igual que el camino normal)
```

- BPTT completo: el gradiente fluye por todos los pasos (verificado: grad ≠ 0
  en `resonance_clock`, `emotion_embeddings`, `emotion_proj`).
- Coste: ~×n_steps en el core; la decodificación a vocabulario es única (ese es
  el punto — elimina el roundtrip por vocabulario que costaba 30% en EXP_029).
- Métricas por paso (normas, convergencia de coseno 0.19→0.95: el bucle
  converge). TODO rendimiento: los `.item()` por paso fuerzan sync — ocultar
  tras flag si el coste se nota.

## 4. Decisiones de diseño (DL-010)

1. **Rampa de profundidad**: `n_steps ~ U[1, 5]` por batch (`--resonance_ramp`,
   eje B de EXP_032) → estabilidad a cualquier profundidad → el knob de
   inferencia es real. `max_resonance_steps=5` (cota del `resonance_clock`;
   `pos_mode=clock` indexa el reloj por paso).
2. **Emoción `first_only`** (la ganadora; `gated` fue la peor). Set: las 6
   emociones del dojo (`EMOTION_NAMES`: miedo, alegría, ira, tristeza, dolor,
   hambre) **+ `neutral` como id 6**. Entrenamiento: muestreo uniforme por
   secuencia (atractores condicionados a emoción). Val/exámenes: **neutral**
   (in-distribution y determinista).
3. **Gateo de etapa idéntico al camino normal**: la máscara se aplica FUERA
   (`apply_stage_gate` sobre los logits finales, convención float 0/-inf —
   NOTA: `logit_mask` interno de `_decode_hidden` usa convención bool, no
   mezclar). La pérdida excluye targets vetados igual que DL-008.
4. **Evaluador acoplado**: `state_manager` propaga los flags de resonancia al
   subproceso de Samantha; el evaluador instancia el modelo con los mismos
   parámetros (el checkpoint resonante tiene `resonance_clock` + emociones —
   sin ellos el `load_state_dict` falla) e infiere con
   `forward_resonance(n_steps=--resonance_eval_steps, emoción neutral)`.
5. **No tirar el control**: el brazo normal (en curso) es el control de la
   comparativa. Los hitos 2-5 años se entrenaron con máscaras correctas; las
   ~6 épocas de etapa 5 contaminadas por el bug del censo (ver §7) se
   descartaron con rollback al checkpoint del hito 5_years.

## 5. Flags del trainer / evaluador

| Flag | Default | Uso brazo resonante |
|---|---|---|
| `--resonance_steps_max` | 0 (off) | 5 |
| `--resonance_pos_mode` | clock | clock |
| `--resonance_ramp` | off | on |
| `--resonance_eval_steps` | 3 | 3 |
| `--n_emotions` | 0 | 7 (dojo 6 + neutral) |
| `--emotion_dim` | 16 | 16 |
| `--emotion_mode` | first_only | first_only |

Receta: `configs/jobs/bit003_glyph_resonant.yaml` (state_dir propio
`bit003_glyph_res`; lanzar en SERIE tras los controles normales).

## 6. Matriz experimental de la tesis

```
                  normal            resonante
inglés   glyph   🟢 EN CURSO       ⬜ bit003_glyph_resonant
         std     ⬜ pendiente      ⬜ opcional
K-65P    glyph   ✅ hecho (770)    ⬜ mismo patrón de flags en k65p (fase 2)
```

**El test distintivo del brazo resonante**: curva precisión-vs-presupuesto-de-
pensamiento — las mismas baterías de examen con n_steps ∈ {1..5} en inferencia.
Si la precisión sube con los pasos, Bit *piensa más profundo cuando se le
deja*: una figura que un transformer de profundidad fija no puede dar.
Extensión futura (tintero): halting adaptativo (PonderNet) — la emoción como
reloj de parada.

**Aclaración de alcance** (pregunta del operador): el bucle NO equivale a 18
capas — son las mismas 6 reutilizadas (profundidad de *cómputo* 18, de
*parámetros* 6; weight-tying, patrón Universal Transformer). Net2DeeperNet
sigue aparcado (RFC-GROWTH-V6); la resonancia cubre el eje de profundidad de
cómputo que este dejaba hueco, sin parámetros nuevos.

## 7. Incidente durante la implementación (bug del censo, cazado por el smoke)

El cache-hit del store CSR mezclaba unidades: `n_general` (conteo de
SECUENCIAS) usado como índice de TOKENS en el slice de CHILDES → censo de
8,161 palabras únicas (ventana falsa de 1.45M tokens) en vez de 18,468 →
máscaras E5/E6 estrechas (8,164/8,165 en vez de 10,003/15,002). El smoke lo
delató (gateo impreso ≠ auditoría). Fix: límites de token desde `offsets`
(9,439,798 tokens / 18,468 únicas — exacto al original). El run en curso
cruzó a etapa 5 con ~6 épocas de máscara estrecha → rollback al checkpoint
`model_milestone_5_years.pt` (época 223) y reanudación con el código fijo.
Hitos 2-5 años: limpios (entrenados en camino miss con censo correcto).

## 8. Limitaciones

1. El camino resonante corre **eager** (torch.compile solo envuelve el forward
   estándar, no usado en este brazo). Compilar `forward_resonance` es
   optimización futura.
2. Los `.item()` de métricas por paso fuerzan sync de GPU (medible con batches
   pequeños; ocultar tras flag si molesta).
3. El brazo K-65P resonante requiere portar los mismos flags a
   `train_sovereign_school_k65p.py` (fase 2, mismo patrón).
4. La emoción en el corpus escolar es sintética (muestreo uniforme): no hay
   señal emocional natural en el texto. Es la misma condición que EXP_033/034
   validaron; la lectura de atractores por emoción se hace en evaluación.
