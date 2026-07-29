"""Política de suspenso de los exámenes de Samantha (29 jul 2026).

El incidente fundacional: Bit suspendió el examen de 7 años (5.60/10) y AUN ASÍ
transicionó a la etapa 8 con neurogénesis a 1024 — el contador de épocas ya
había cruzado la frontera antes del examen y el sys.exit(1) del suspenso, bajo
el job runner, provocó un reintento que leyó el estado avanzado. La puerta de
la etapa es el examen: el suspenso retiene la etapa (repaso) y al K-ésimo pasa
la decisión al operador.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bitnet.training.train_sovereign_school import EXAM_MAX_FAILURES, plan_exam_failure

STAGE_7 = {"stage_idx": 6, "name": "secondary_7", "dim": 896, "age": 7, "start_epoch": 897, "end_epoch": 1120, "epochs": 224}


def _state(epoch=1121, stage=7, failures=None):
	state = {"current_epoch": epoch, "current_stage_idx": stage, "hidden_dim": 896, "milestones_achieved": ["2_years"], "curriculum_hash": "abc"}
	if failures:
		state["exam_failures"] = failures
	return state


def test_first_failure_holds_the_stage_and_rewinds_for_remedial():
	held, needs_operator = plan_exam_failure(_state(), STAGE_7, "7_years")

	assert needs_operator is False
	assert held["current_stage_idx"] == 6  # la etapa NO avanza: la puerta es el examen
	rewind = max(8, int(STAGE_7["epochs"] * 0.25))
	assert held["current_epoch"] == STAGE_7["end_epoch"] - rewind + 1  # bloque de repaso
	assert held["exam_failures"] == {"7_years": 1}
	assert held["curriculum_hash"] == "abc"  # el resto del estado sobrevive intacto


def test_rewind_never_leaves_the_stage():
	tiny = {"stage_idx": 1, "name": "1-2", "age": 2, "start_epoch": 65, "end_epoch": 70, "epochs": 6}
	held, _ = plan_exam_failure(_state(epoch=71, stage=1), tiny, "2_years")
	assert held["current_epoch"] == tiny["start_epoch"]  # el repaso no puede caer en la etapa anterior


def test_kth_failure_escalates_to_operator_at_the_boundary():
	held, needs_operator = plan_exam_failure(_state(failures={"7_years": EXAM_MAX_FAILURES - 1}), STAGE_7, "7_years")

	assert needs_operator is True
	assert held["exam_failures"]["7_years"] == EXAM_MAX_FAILURES
	# Parado EN la frontera: tras el resume del operador, una época de refresco y re-examen
	assert held["current_epoch"] == STAGE_7["end_epoch"]
	assert held["current_stage_idx"] == 6


def test_failure_counters_are_per_milestone():
	held, needs_operator = plan_exam_failure(_state(failures={"6_years": 2}), STAGE_7, "7_years")
	assert needs_operator is False
	assert held["exam_failures"] == {"6_years": 2, "7_years": 1}  # el expediente no se mezcla
