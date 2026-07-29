# Sesión 2026-07-06 — Consolidación School v3

**Operador**: Joan · **Agente**: Aleth (Gemini 2.5 Pro) · **Duración**: ~4h

---

## Objetivos

Asentar la base de código de frankenswarm antes de generar el corpus para School v3.
4 tareas definidas y ejecutadas secuencialmente.

## Trabajo realizado

### T1: Fix `samantha_on_demand.py` — temperatura
- Bug: `invoke()` no aceptaba `temperature`, hardcoded a `0.0`
- Fix: propagación completa, default `0.7`
- Tests: 7 (red-pill) + 9 (frankenswarm)

### T2: Neurogénesis por plateau
- Antes: calendario (`epoch == config["start_epoch"]`)
- Después: monitor de val_loss con `--patience=15`, `--min_delta=0.01`
- Helper `get_next_dim()` con ceiling por etapa
- Estado en `school_state.json`: `best_val_loss`, `epochs_without_improvement`, `neurogenesis_history`
- Tests: 17

### T3: Reorganización `src/bitnet/`
- 39 ficheros planos → 10 submódulos: `model/`, `growth/`, `vocab/`, `worlds/`, `data/`, `operators/`, `training/`, `inference/`, `translation/`, `telemetry/`
- Retrocompatibilidad via `MetaPathFinder` import hook
- ~80 ficheros actualizados
- Datos Prolog (`.pl`) movidos junto a sus consumidores

### T4: Limpieza scripts legacy
- 10 training scripts → `lab/experiments/`
- Solo `train_sovereign_school.py` permanece activo

## Resultados

```
72 passed, 3 warnings in 25.82s
```

De 46 tests originales a 72 (+26 nuevos). Los 3 warnings son pre-existentes (fastembed deprecation).

## Siguiente paso

Generación de corpus para School v3: parámetros de la fábrica, planificación del entrenamiento.
