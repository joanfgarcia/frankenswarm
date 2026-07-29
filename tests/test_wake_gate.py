"""Sovereign Wake Gate — the four-check doctrine, verified.

Probes are injected: no nvidia-smi, no wall clock, no real operator required.
"""

import json
import time

from src.swarm.wake_gate import (
	GateDecision,
	build_contained_command,
	check_budget,
	check_consent,
	check_containment,
	check_hardware,
	evaluate_gate,
	record_decision,
)

GPU_FREE = lambda: (7000, 45)  # noqa: E731
GPU_BUSY = lambda: (1200, 45)  # noqa: E731
GPU_FEVER = lambda: (7000, 91)  # noqa: E731
GPU_DEAD = lambda: None  # noqa: E731


def make_repo(tmp_path, exp_id="032", with_config=True):
	(tmp_path / "configs" / "experiments").mkdir(parents=True)
	if with_config:
		(tmp_path / "configs" / "experiments" / f"EXP_{exp_id}.json").write_text("{}")
	return tmp_path


def idle_signal(tmp_path, age_s, now):
	signal = tmp_path / "activity.txt"
	signal.write_text("x")
	import os
	os.utime(signal, (now - age_s, now - age_s))
	return [str(signal)]


def test_consent_requires_config_on_disk(tmp_path):
	base = make_repo(tmp_path, with_config=True)
	ok, _ = check_consent({"id": "032"}, str(base))
	assert ok
	ok, why = check_consent({"id": "099"}, str(base))
	assert not ok
	assert "consent" in why or "EXP_099" in why


def test_consent_rejects_task_without_id(tmp_path):
	ok, _ = check_consent({}, str(make_repo(tmp_path)))
	assert not ok


def test_consent_accepts_config_template(tmp_path):
	base = make_repo(tmp_path, with_config=False)
	(base / "configs" / "experiments" / "EXP_032_TEMPLATE.json").write_text("{}")
	task = {"id": "032", "config_template": "configs/experiments/EXP_032_TEMPLATE.json"}
	ok, _ = check_consent(task, str(base))
	assert ok


def test_hardware_holds_when_vram_busy(tmp_path):
	now = time.time()
	ok, why, observed = check_hardware(idle_signal(tmp_path, 7200, now), gpu_probe=GPU_BUSY, now=now)
	assert not ok
	assert "VRAM" in why
	assert observed["vram_free_mb"] == 1200


def test_hardware_holds_on_fever(tmp_path):
	now = time.time()
	ok, why, _ = check_hardware(idle_signal(tmp_path, 7200, now), gpu_probe=GPU_FEVER, now=now)
	assert not ok
	assert "fever" in why


def test_hardware_holds_when_operator_present(tmp_path):
	now = time.time()
	ok, why, observed = check_hardware(idle_signal(tmp_path, 60, now), gpu_probe=GPU_FREE, now=now)
	assert not ok
	assert "active" in why
	assert observed["operator_idle_s"] < 3600


def test_hardware_refuses_to_fire_blind(tmp_path):
	now = time.time()
	ok, why, _ = check_hardware(idle_signal(tmp_path, 7200, now), gpu_probe=GPU_DEAD, now=now)
	assert not ok
	assert "blind" in why


def test_hardware_refuses_without_presence_signals(tmp_path):
	ok, why, _ = check_hardware([str(tmp_path / "missing.txt")], gpu_probe=GPU_FREE, now=time.time())
	assert not ok
	assert "absence" in why


def test_hardware_passes_when_alone_and_cold(tmp_path):
	now = time.time()
	ok, _, observed = check_hardware(idle_signal(tmp_path, 7200, now), gpu_probe=GPU_FREE, now=now)
	assert ok
	assert observed["operator_idle_s"] >= 3600


def test_containment_command_shape():
	cmd = build_contained_command("src/bitnet/run_grid_032.py", "configs/experiments/EXP_032.json")
	ok, _ = check_containment(cmd)
	assert ok
	assert "systemd-run" in cmd
	bare = ["python", "train.py"]
	ok, why = check_containment(bare)
	assert not ok
	assert "containment" in why


def test_budget_counts_only_today_fires(tmp_path):
	ledger = tmp_path / "ledger.jsonl"
	entries = [
		{"timestamp": "2026-07-03T01:00:00", "fire": True},
		{"timestamp": "2026-07-03T04:00:00", "fire": False},
		{"timestamp": "2026-07-02T04:00:00", "fire": True},
	]
	ledger.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
	ok, _ = check_budget(str(ledger), max_per_day=2, today="2026-07-03")
	assert ok
	ok, why = check_budget(str(ledger), max_per_day=1, today="2026-07-03")
	assert not ok
	assert "exhausted" in why


def test_gate_fires_only_all_green(tmp_path):
	base = make_repo(tmp_path)
	now = time.time()
	cmd = build_contained_command("x.py", "c.json")
	ledger = str(base / "lab" / "wake_ledger.jsonl")
	signals = idle_signal(base, 7200, now)
	decision = evaluate_gate({"id": "032"}, str(base), cmd, ledger, gpu_probe=GPU_FREE, activity_signals=signals, now=now)
	assert decision.fire
	assert all(decision.checks.values())
	decision = evaluate_gate({"id": "032"}, str(base), cmd, ledger, gpu_probe=GPU_BUSY, activity_signals=signals, now=now)
	assert not decision.fire
	assert not decision.checks["hardware"]
	assert decision.checks["consent"]


def test_every_decision_leaves_a_record(tmp_path):
	ledger = tmp_path / "lab" / "wake_ledger.jsonl"
	hold = GateDecision(fire=False, task_id="032", checks={"hardware": False}, reasons=["VRAM busy"], observed={"vram_free_mb": 1200})
	record_decision(str(ledger), hold)
	fire = GateDecision(fire=True, task_id="032", checks={"hardware": True}, reasons=["all green"], observed={})
	record_decision(str(ledger), fire)
	lines = [json.loads(line) for line in ledger.read_text().splitlines()]
	assert len(lines) == 2
	assert lines[0]["fire"] is False
	assert lines[0]["reasons"] == ["VRAM busy"]
	assert lines[1]["fire"] is True
	assert all("timestamp" in entry for entry in lines)
