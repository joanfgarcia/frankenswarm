# RFC — Canal 66: Planos de Codificación de Glifos (patrón UTF)

> **Estado**: ESPECIFICACIÓN APROBADA (31-ago-2026) — implementación como siguiente generación (post-v4 ×1)
> **Autor**: Joan García + Aleth · **Decisión**: `docs/DECISION_LOG.md` (DL-012)
> **Precedente interno**: doctrina "4º bit" (EXP_037) · **Analogy**: codificaciones UTF-8/16/32

---

## 1. Problema

Los 65 primos de Wierzbicka son el alfabeto semántico completo **para lo NSM-expresable**. Quedan fuera tokens cuyo significado es **convención, no descomposición**: numerales (`three…nineteen` — solo `one`/`two` son primos 55/56), constantes (π, e), operadores matemáticos, notación. Consecuencia directa: la parte aritmética del currículo y de la batería 80/50 **no es traducible** al brazo K-65P — la confrontación "2 sabores" pierde las matemáticas.

## 2. El diseño: un canal de TIPO, no un primo 66

`dim66` no añade un primitivo semántico — es un **selector de plano de codificación**, con la semántica del patrón UTF-8:

| dim66 | Plano | Lectura de los 65 dims |
|---|---|---|
| **0** (default) | **semántico NSM** | coordenadas de primos — idéntico a hoy (retrocompatible: todo glifo existente es válido sin tocarlo) |
| **+1** | **simbólico** | código de símbolo: numerales, constantes, operadores matemáticos, notación |
| **-1** | **meta/estructural** | código estructural: marcadores de examen/etiquetas — rol asignable |

**En modo marcado (±1), los 65 dims dejan de ser coordenadas de primos y pasan a ser espacio libre de código** — como los bytes de continuación de UTF-8 no significan ASCII. Capacidad: 3^65 espacios de código por plano.

**Propiedad crítica heredada de UTF**: retrocompatibilidad garantizada por construcción. Todo glifo existente (65d, implícitamente dim66=0) es un glifo v2 válido sin modificación — ASCII puro es UTF-8 válido.

## 3. Política de identidad de símbolos (v0)

En modo marcado, la identidad del símbolo vive en el **embedding aprendido por token id** (los numerales no tienen descomposición NSM: su semántica es convencional). El canal 66 le dice al modelo "no esperes estructura de primos aquí — soy opaco por diseño". Extensión futura (fase 2): codificación de magnitud para numerales sobre los primos de cantidad (`much`/`more`) — "five" > "two" en el eje.

## 4. Formato glifo v2

```
glifo v1:  (V, 65)  int8 {-1,0,1}     ← generación actual (v4 ×1 en curso)
glifo v2:  (V, 66)  int8              ← canal 66 añadido; glifos v1 migran con dim66=0
```

Migración: añadir columna de ceros a `expanded_glyphs.json` → **todo checkpoint v1 carga en arquitectura 66** con GLYPH_DIM=66 solo si se re-entrena (los checkpoints v1 cargan en 65 intactos).

## 5. Qué desbloquea

1. **Aritmética y símbolos entran en el brazo K-65P** → la batería 80/50 traducida cubre también matemáticas (hoy excluida) → la confrontación "2 sabores" completa.
2. **Marcado honesto de convenciones**: numerales existentes en el vocabulario (cuyos glifos actuales son composiciones Ridge sin semántica) pasan a marcados + plano simbólico — representación honesta de que son rote.
3. **El modelo distingue comprensible vs convencional** a nivel de representación — potencialmente aprendible en atención (canal visible).
4. **Extensibilidad**: el patrón escala a dims 67+ si los namespaces se multiplican; el registry de planos es una tabla, no una migración.

## 6. Faseado

| Fase | Qué | Cuándo |
|---|---|---|
| 0 | v4 ×1 (glyph/standard) terminan en 65d — sin tocar | EN CURSO |
| 1 | Formato glifo v2 + migrador (dim66=0) + `GLYPH_DIM` paramétrico en el modelo | post-v4 |
| 2 | Tokens simbólicos (numerales 3-19, constantes) con plano +1 | fase 1 |
| 3 | Traductor EN→K-65P + batería traducida (incluye aritmética vía plano +1) | tras el traductor |

## 7. Limitaciones abiertas

1. 3 estados del canal = 2 planos extra. Si se necesitan más: dims 67+ (el patrón escala) o分配 por rangos dentro de plano.
2. Los símbolos son **opacos por diseño** (identidad aprendida) — la batería debe medirlos como rote, no como comprensión.
3. La batería K-65P traducida seguirá excluyendo lo no expresable incluso con plano +1 (el plano da REPRESENTACIÓN, no gramática — la sintaxis K-65P necesitaría operadores para aritmética si se quiere razonar con símbolos, no solo representarlos).

## 8. Referencias

- `docs/RFC-002_SINTAXIS_GLIFOS.md` (§5: compilador aplazado, test de ida-vuelta)
- `docs/DECISION_LOG.md:DL-012` · EXP_037 ("4º bit") como precedente doctrinal
- `src/bitnet/translation/nsm_syntax_en.py` (gramática EN, ya operativa)
