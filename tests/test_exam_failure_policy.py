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

from src.bitnet.training.modules.state_manager import EXAM_MAX_FAILURES, plan_exam_failure

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


# ── Contrato de exit-codes evaluador ↔ state_manager (auditoría 1-sep-2026) ──
# El incidente: Samantha devolvió None (sin JSON) en el examen de 2_years del
# glyph v4 ×1 y el rc=1 genérico se contó como suspenso académico → neurogénesis
# 128→256 sin examen real. Contrato nuevo: 0 = aprobado · 2 = suspenso
# CALIFICADO · cualquier otro rc = infraestructura (reintento único y, si
# persiste, pausa del operador SIN contar suspenso ni disparar neurogénesis).

import json as _json
import types


class _TinyModel:
	def __init__(self):
		import torch
		self._lin = torch.nn.Linear(2, 2)
		self.hidden_dim = 128
		self.core_layers = [None] * 6

	def state_dict(self):
		return self._lin.state_dict()

	def cpu(self):
		return self

	def to(self, device):
		return self


def _eval_env(tmp_path, failures=None):
	save_dir = tmp_path / "arm"
	save_dir.mkdir(parents=True, exist_ok=True)
	(tmp_path / "storage" / "checkpoints").mkdir(parents=True, exist_ok=True)
	state = _state(epoch=71, stage=1, failures=failures)
	state_path = save_dir / "school_state.json"
	state_path.write_text(_json.dumps(state))
	args = types.SimpleNamespace(test_mock=True, resonance_steps_max=0)
	stage_conf = {"stage_idx": 1, "name": "1-2", "age": 2, "start_epoch": 65, "end_epoch": 70, "epochs": 6}
	return dict(
		model=_TinyModel(), current_checkpoint_path=str(save_dir / "model_current.pt"),
		target_milestone="2_years", save_dir=str(save_dir), stage_idx=1, stage_name="1-2",
		milestones_achieved=[], state_path=str(state_path), args=args, device="cpu",
		base_dir=str(tmp_path), epoch=71, stage_conf=stage_conf,
	)


def _fake_run(returncodes):
	calls = []
	def fake(cmd, env=None):
		calls.append(cmd)
		rc = returncodes[min(len(calls) - 1, len(returncodes) - 1)]
		return types.SimpleNamespace(returncode=rc)
	fake.calls = calls
	return fake


def test_infra_error_never_counts_as_academic_fail(tmp_path, monkeypatch):
	import src.bitnet.training.modules.state_manager as sm
	fake = _fake_run([3, 3])
	monkeypatch.setattr(sm.subprocess, "run", fake)
	monkeypatch.setattr(sm.time, "sleep", lambda s: None)

	res = sm.run_samantha_eval(**_eval_env(tmp_path))

	assert len(fake.calls) == 2  # reintento único
	assert res.passed is False
	assert res.infra_error is True
	assert res.needs_operator is True
	assert res.state.get("exam_failures", {}) == {}  # NI un suspenso contado
	failed = _json.loads((tmp_path / "storage" / "checkpoints" / "milestone_failed.json").read_text())
	assert failed["status"] == "eval_infra_error"


def test_graded_fail_rc2_counts_as_academic_fail(tmp_path, monkeypatch):
	import src.bitnet.training.modules.state_manager as sm
	fake = _fake_run([2])
	monkeypatch.setattr(sm.subprocess, "run", fake)

	res = sm.run_samantha_eval(**_eval_env(tmp_path))

	assert len(fake.calls) == 1  # un suspenso calificado no se reintenta
	assert res.passed is False
	assert res.infra_error is False
	assert res.state["exam_failures"] == {"2_years": 1}


def test_rc1_crash_retries_and_can_recover(tmp_path, monkeypatch):
	import src.bitnet.training.modules.state_manager as sm
	fake = _fake_run([1, 0])
	monkeypatch.setattr(sm.subprocess, "run", fake)
	monkeypatch.setattr(sm.time, "sleep", lambda s: None)

	res = sm.run_samantha_eval(**_eval_env(tmp_path))

	assert len(fake.calls) == 2
	assert res.passed is True
	assert res.infra_error is False


def test_evaluator_returns_none_on_missing_model(tmp_path):
	from scripts.evaluate_samantha_age import run_evaluation
	args = types.SimpleNamespace(model_path=str(tmp_path / "no_such.pt"), device="cpu")
	passed, score = run_evaluation(args)
	assert passed is None  # infra: sin modelo no hay calificación
