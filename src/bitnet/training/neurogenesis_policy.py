"""Política de neurogénesis mínima (DL-019, 2-sep).

Diseño del operador: las últimas generaciones de entrenamiento nunca
ejercitaron la escalera de capacidad (128d de inicio bastó siempre). El
vocabulario refundado K-65P v2 empieza la escalera MUY abajo (16-32d) para
que la neurogénesis responda de verdad y midamos el mínimo real por etapa.

Tres reglas:
  1. SIN TECHO: ninguna dimensión máxima — la capacidad crece lo que el
     currículo exija. Guardia opcional: si se supera `sanity_bound`, pausa
     para el operador (contrato de exit-codes), nunca un cap duro silencioso.
  2. CRECIMIENTO MÍNIMO: Δdim = max(8, dim // 8) — el menor paso que alivia
     un cuello de rango conservando aritmética limpia (siempre múltiplos de 8):
     16→24→32→40→48→56→64→72→80→...→128→144→160→...
  3. PUNTO DE MEDICIÓN 65: el glifo v2 es un vector de 65 trits; al cruzar
     d≥65 la malla cabe sin compresión. La escalera cruza ese punto entre
     64 y 72 — el salto de capacidad allí es un dato, no un artefacto.

Fórmula y alternativas: no existe fórmula canónica en la literatura para el
paso de neurogénesis (lo cercano: progressive widening, saturación de rango
efectivo). Δ=12.5% con suelo 8 es el compromiso mínimo-mayor que no hace
micro-pasos inútiles ni saltos groseros (el ×2 actual es 8× más agresivo).
Instrumentación futura (v2): participation ratio de los hidden states por
etapa para VALIDAR la fórmula con el déficit medido, no supuesto.
"""

MIN_STEP = 8
PROPORTION = 8  # dim // PROPORTION = crecimiento proporcional (12.5%)
DEFAULT_SANITY_BOUND = 4096


def next_dim(current: int, sanity_bound: int | None = DEFAULT_SANITY_BOUND) -> int | None:
	"""Siguiente dimensión mínima tras un suspenso real (no infra).

	Devuelve None solo si se supera el sanity_bound — señal de pausa para el
	operador, no un techo de capacidad.
	"""
	if current < MIN_STEP:
		raise ValueError(f"dim inicial {current} < {MIN_STEP}: arranca en 16-32 (DL-019)")
	candidate = current + max(MIN_STEP, current // PROPORTION)
	if sanity_bound is not None and candidate > sanity_bound:
		return None
	return candidate


def ladder(start: int, steps: int = 20, sanity_bound: int | None = DEFAULT_SANITY_BOUND) -> list[int]:
	"""La escalera completa desde `start` (para inspección y tests)."""
	out = []
	d = start
	for _ in range(steps):
		d = next_dim(d, sanity_bound=sanity_bound)
		if d is None:
			out.append("⏸ sanity_bound")
			break
		out.append(d)
	return out
