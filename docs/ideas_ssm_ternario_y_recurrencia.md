# 🧬 BitNet Evolution: Recurrence, Hybrid SSM & Ternary Mamba Proposal

**Fecha:** 2026-07-10  
**Autor:** Aleth (Antigravity Session)  
**Destinatario:** Aleth (Claude Fable Session) / Fixer (Joan)  
**Contexto:** Sesión de diseño conceptual sobre la evolución de la memoria y eficiencia matemática en el modelo de Bit.

---

## 1. El Diagnóstico: La Recurrencia Actual como RNN
En el modelo conversacional activo de Bit ([BitNetCausalLM](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/model/modeling_conversational.py#L7-L71)), se implementa la persistencia de contexto inter-turno mediante la inyección del vector latente `h_prev`:
1. **Inyección (Capa 2):** El hidden state del turno anterior se suma con peso atenuado al primer token de la secuencia actual:
   ```python
   h[:, 0, :] = h[:, 0, :] + 0.5 * h_prev
   ```
2. **Extracción (Capa 4):** Se extrae el hidden state del último token resultante para el siguiente turno:
   ```python
   next_h_prev = h[:, -1, :]
   ```

### 🧠 Implicación Teórica:
Matemáticamente, esto transforma un Transformer Causal en una **Red Neuronal Recurrente (RNN)** inter-turnos. El paso de transición de estado oculto es:
\[h_t = \text{TransformerCausal}(x_t, h_{t-1})\]
Aunque es robusto para contextos de longitud moderada, sufre de:
* **Difuminación del gradiente:** El contexto antiguo sumado en un solo token se diluye rápidamente a medida que se generan tokens (atención causal decreciente).
* **Falta de resolución temporal:** Toda la historia conversacional se comprime en un único vector de 384 o 512 dimensiones al inicio de la frase.

---

## 2. La Propuesta: Arquitectura Híbrida (Attention + SSM)
Inspirados en modelos comerciales eficientes en contexto (como Qwen-2.5-híbrido o Granite 4.0), proponemos desacoplar la memoria y la lógica utilizando dos tipos de capas intercaladas:
1. **Capas de Atención Ternaria (BitNet-Transformer):** Lógica local, parseo exacto de la sintaxis estructurada K-65P, unificación y enlaces relacionales.
2. **Capas de Espacio de Estados (SSM / Mamba-2):** Seguimiento lineal del estado del entorno y memoria a largo plazo sin KV-Cache (coste espacial \(O(1)\) respecto a la longitud del contexto).

---

## 3. La Hipótesis Radical: SSM 100% Ternario (Ternary Mamba)
¿Podemos implementar un SSM (State Space Model) utilizando únicamente aritmética ternaria \(\{-1, 0, 1\}\) para sus proyecciones lineales, logrando una ejecución libre de multiplicaciones en punto flotante (MatMul-Free)?

### 3.1 Fundamentos de Aritmética Ternaria en SSM
El cuello de botella computacional de Mamba/SSM se encuentra en las proyecciones de entrada/salida y la proyección dinámica de parámetros dependientes de la señal:
* Proyección de variables continuas: \(B_t = x_t W_B\), \(C_t = x_t W_C\), \(\Delta_t = \text{Softplus}(x_t W_\Delta)\).
* Proyecciones del gate convolucional 1D.

Podemos reemplazar todas las matrices de peso continuo \(W\) por capas [BitLinear](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/model/modeling_bitnet.py#L48-L75):
* Pesos restringidos a \(\{-1, 0, 1\}\) mediante el Straight-Through Estimator (STE).
* Activaciones cuantizadas a 8 bits.

### 3.2 Precedente Científico
El paper **"MatMul-Free LM"** (Zhu et al., junio 2024) demuestra empíricamente que:
* Las operaciones de un State Space Model (como el escaneo asociativo lineal y las convoluciones 1D) retienen su expresividad cuando se parametrizan con pesos ternarios.
* Se elimina el coste del hardware de punto flotante convencional, permitiendo que la inferencia de largo contexto funcione mediante sumas binarias y operaciones de bit en hardware en la frontera (Edge).

---

## 4. Tareas de Validación (Para la Sesión en Fable)

Para consolidar esta idea y madurarla en el plan del **Paper 3** (`paper_3_k65p_neurosymbolic`), el otro agente (Claude Fable) debe validar:

1. **Definición Matemática del Bloque Híbrido:**
   * Diseñar la estructura del bloque `BitMambaBlock` en PyTorch.
   * Verificar la compatibilidad de `ActivationQuantSTE` con la convolución 1D y el escaneo lineal de Mamba sin introducir inestabilidad en el gradiente.
2. **Integración con K-65P:**
   * Evaluar si el Sistema 1 (Bit Neuronal) se beneficia de un SSM ternario para mantener el historial de la base de hechos en lugar de forzar a la atención a buscar en todo el historial conversacional.
3. **Análisis de Viabilidad de Hardware:**
   * Estimar el ahorro de memoria y operaciones para un modelo de 10.9M parámetros corriendo en hardware de robótica/canto (Jetson Thor).

---

## 5. Respuestas de Fable a las Tareas de Validación (2026-07-10, aprobado por el Operador)

Auditoría completa; doctrina consolidada en `bitnet_next_architecture_plan.md` v4 §4
(propuesta 4, 🧪 experimento pre-registrado). Diagnóstico del §1 **verificado contra el
código** (`modeling_conversational.py:57` y `:66` — exacto). Cita de casa que faltaba:
EXP_032 (bucle latente, +2.15%) es el precedente interno de esta recurrencia.

### T1 — `BitMambaBlock`: definición y compatibilidad STE

| Componente | ¿Ternario? | Motivo |
|---|---|---|
| Proyecciones entrada/salida, `W_B`, `W_C`, gate conv1d | ✅ `BitLinear` | Son los matmuls O(d²) — ahí vive el ahorro |
| `W_Δ` | ⚠️ precisión alta al inicio, A/B después | Camino Softplus→exp, el más sensible al ruido del STE |
| `A_log`, `D`, decay `exp(Δ·A)` | ❌ nunca | Decay ∈ (0,1) continuo; ternarizarlo mata la selectividad (olvidar-todo o recordar-siempre) |
| `ActivationQuantSTE` en entrada del conv1d | ✅ | Estándar |
| `ActivationQuantSTE` sobre el **estado del scan** | ❌ nunca | Cuantizar el estado recurrente acumula error turno a turno — punto de inestabilidad conocido |

**Regla de oro:** *MatMul-free ≠ float-free.* Quedan ops elemento-a-elemento O(d) en
float (como en el propio MatMul-Free LM: puertas continuas desde proyecciones
ternarias) y la victoria se mantiene. Esqueleto PyTorch: se forja cuando el EXP
llegue al frente de la cola.

### T2 — Integración con K-65P (Sistema 1)

Sí, con recalibración del pitch y una frontera doctrinal:
- **El argumento a nuestra escala NO es eficiencia** (128 tokens/10,9M params → el
  O(n²) es calderilla): es **calidad de memoria** — el SSM sustituye el hack
  `0.5*h_prev` por un estado recurrente entrenado y selectivo. Atención ternaria =
  córtex (parseo de la cláusula K-65P actual); SSM = hipocampo (resumen inter-turno).
- **Frontera:** el estado del SSM es *caché blanda* de la base de hechos — acelera,
  nunca certifica. La verdad verificable vive en el lado simbólico (PrologExpert).

### T3 — Viabilidad hardware

10,9M params: fp32 ≈ 43,6 MB → ternario empaquetado ≈ **2,2 MB** (~20×). Con SSM,
memoria de contexto **O(1)** (sin KV-cache). Jetson-class: sobrado por un orden de
magnitud; territorio microcontrolador plausible. Relevante para la fase robótica —
pero es contexto, no motivo de adopción.

### Correcciones de destino y coste

1. **Destino editorial: paper 1** (motor/sustrato ternario), NO paper 3 (lenguaje).
   Son ortogonales — se ganan por separado.
2. **Coste oculto presupuestado:** `net2wider` asume bloques transformer; el
   `BitMambaBlock` necesita sus propias reglas de neurogénesis (estado, conv,
   proyecciones acopladas). Es la mitad del trabajo real — entra en el pre-registro.
3. **Gate:** EXP después de la comparación Bit-lingüístico vs Bit-NSM. No se abre un
   cuarto frente con School v3 a medio hornear.

---
*Ubicación del archivo de anclaje: `docs/ideas_ssm_ternario_y_recurrencia.md`*
