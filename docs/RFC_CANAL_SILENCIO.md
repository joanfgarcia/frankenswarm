# RFC — Canal de Silencio: la abstención calibrada de Bit (pre-registro)

> **Estado**: PRE-REGISTRO — criterio de éxito congelado ANTES de cualquier run
> **Autor**: Aleth + Joan García · **Fecha**: 2-sep-2026 · **Padre**: DL-015 F6
> **Dato que motiva**: la calibración de confianza EXISTE dentro de lo visto
> (top-20% más seguro → 90-100% de acierto) y es casi nula en lo no visto
> (0-5.4%). El diseño debe premiar el "no sé" justo donde la señal es débil.

---

## 1. El principio (doctrina del operador)

*"Es peor fallar diciendo algo que no es, que decir no sé."* Y: *"decir no sé
cuando realmente te pregunto algo que no ha visto nunca es un acierto."*

La matriz 3×2 de puntuación ya lo codifica: visto +1/0/−1 · no visto +2/0/0.
Este RFC define el MECANISMO para que Bit pueda decir "no sé" y el criterio
de éxito para medir si la sabe usar.

## 2. Lo que la auditoría estableció

1. Bit **no puede abstenerse hoy**: la generación es argmax forzado sobre el
   vocabulario gateado — siempre emite su palabra más probable.
2. **La calibración existe latente y es asimétrica**: dentro de lo visto, la
   confianza ordena la corrección (top-20% → 90-100%); en lo no visto, la
   confianza casi no discrimina (0-5.4%). Es decir: Bit "sabe que no sabe"
   **dentro de su curriculum** pero **no sabe que no sabe fuera de él**.
3. "i dont know" es generable: `dont` y `know` están en el vocabulario y en
   todos los gateos de etapa (verificado 31-ago).

## 3. Opciones de diseño (a evaluar, no a mezclar)

| Opción | Mecanismo | Riesgo |
|---|---|---|
| **A — umbral de confianza por etapa** | `conf < θ_etapa → "i dont know"`; θ calibrado por etapa contra el pool | la confianza en lo no visto casi no discrimina → el umbral no separa visto/no visto por sí solo |
| **B — abstención entrenada** | preguntas de entrenamiento con respuesta esperada "dont know" (probes fuera-de-currículo ×factor) → el modelo aprende SU frontera | requiere generar probes de calidad y el equilibrio de mezcla |
| **C — híbrido** | B entrena la noción; A calibra el umbral en inferencia | el más completo, el más caro |

**La opción B es la que ataca la causa**: hoy Bit no sabe que no sabe FUERA
del currículo porque nada le enseñó dónde está su borde. Los probes
("preguntas trampa" sobre cosas que el currículo NO cubre, con respuesta
esperada "dont know") le enseñan el borde.

## 4. Criterio de éxito PRE-REGISTRADO (congelado antes del run)

Con la batería v3 (58 vistas + 189 no vistas, 5 estratos) sobre el checkpoint
post-abstención:

1. **Vistas**: acierto ≥70% Y abstención ≤10% — el gate no se degrada.
2. **No vistas**: abstención ≥40% Y acierto ≤10% — Bit reconoce lo que no
   sabe y lo dice la mayoría de las veces.
3. **Alucinación en no vistas**: ≤5% — el número que la tesis quiere minimizar.
4. **La calibración dentro de vistas se mantiene** (top-20% seguro → acierto
   superior a la media).

Comparación contra el baseline medido (0% abstención, 4.1% acierto, 95.9%
alucinación en no vistas): CUALQUIER resultado que cumpla 1-3 es el hallazgo
de tesis: **Bit sabe lo que no sabe**.

## 5. Implementación

1. **Tokens**: "dont" + "know" ya en vocab y en todos los gates (verificado).
   La respuesta de abstención: "i dont know" (3 tokens, dentro del presupuesto
   de 5).
2. **Probes de entrenamiento** (opción B): por etapa, N probes con respuesta
   "dont know" sobre hechos FUERA del currículo de la etapa (del universo
   NSM pero no del bank) — ×exam_repeat_factor como los hechos.
3. **Inferencia**: `conf < θ → "i dont know"` (opción A) o el modelo lo
   aprende (opción B).
4. **Samantha**: la abstención es válida-no-correcta — tercera categoría en
   la corrección (nunca fallo).

## 6. Referencias

- DL-015 (la auditoría) · `docs/RFC_INSTRUMENT_V3.md` (la batería) ·
  `scripts/run_battery_v3.py` (la matriz 3×2 y la curva riesgo-cobertura)
- El dato fundacional: curva real — dentro de vistas la confianza ordena
  (90-100%); fuera, casi no discrimina (0-5.4%)
