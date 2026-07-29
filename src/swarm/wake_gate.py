"""Sovereign Wake Gate — the four-check doctrine for autonomous training launches.

Green-lit by the Operator on 2026-07-03. An autonomous awakening may fire a training
run ONLY if all four checks pass:

1. CONSENT     — the experiment sits in the operator-approved queue with its config on disk.
2. HARDWARE    — enough free VRAM, no thermal fever, and the operator genuinely absent.
3. CONTAINMENT — the launch is wrapped in a memory-capped cgroup with a hard runtime limit.
4. BUDGET      — the daily autonomous-training cap is not exhausted.

Every decision — fire or hold — is appended to the wake ledger. A silent retreat is
as opaque as an unauthorized launch: both get a written record of what the gate saw,
what it decided, and why.
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MIN_FREE_VRAM_MB = 6000
MAX_GPU_TEMP_C = 80
OPERATOR_IDLE_SECONDS = 3600
MAX_AUTONOMOUS_RUNS_PER_DAY = 2
MEMORY_MAX = "10G"
RUNTIME_MAX_SECONDS = 4 * 3600

DEFAULT_ACTIVITY_SIGNALS = [
	"~/.local/share/red-pill/last_user_activity.txt",
	"~/.claude/projects",
	"~/.gemini/antigravity/brain",
]


@dataclass
class GateDecision:
	fire: bool
	task_id: str
	checks: dict = field(default_factory=dict)
	reasons: list = field(default_factory=list)
	observed: dict = field(default_factory=dict)


def _nvidia_probe() -> tuple[int, int] | None:
	try:
		out = subprocess.run(
			["nvidia-smi", "--query-gpu=memory.free,temperature.gpu", "--format=csv,noheader,nounits"],
			capture_output=True, text=True, timeout=10, check=True,
		).stdout.strip().splitlines()[0]
		free_mb, temp_c = (int(x.strip()) for x in out.split(","))
		return free_mb, temp_c
	except Exception:
		return None


def _latest_mtime(path: Path) -> float:
	if path.is_file():
		return path.stat().st_mtime
	newest = 0.0
	for child in path.rglob("*"):
		if child.is_file():
			newest = max(newest, child.stat().st_mtime)
	return newest


def check_consent(task: dict, base_dir: str) -> tuple[bool, str]:
	task_id = task.get("id")
	if not task_id:
		return False, "task has no id — not an approved queue entry"
	config = Path(base_dir) / "configs" / "experiments" / f"EXP_{task_id}.json"
	template = task.get("config_template")
	if config.exists():
		return True, f"approved queue entry with config {config.name}"
	if template and (Path(base_dir) / template).exists():
		return True, f"approved queue entry with template {template}"
	return False, f"no config on disk for EXP_{task_id} — queue entry is not executable consent"


def check_hardware(
	activity_signals: list[str] | None = None,
	idle_seconds: int = OPERATOR_IDLE_SECONDS,
	min_free_vram_mb: int = MIN_FREE_VRAM_MB,
	max_gpu_temp_c: int = MAX_GPU_TEMP_C,
	gpu_probe=_nvidia_probe,
	now: float | None = None,
) -> tuple[bool, str, dict]:
	observed: dict = {}
	gpu = gpu_probe()
	if gpu is None:
		return False, "GPU probe unavailable — refusing to fire blind", observed
	free_mb, temp_c = gpu
	observed["vram_free_mb"] = free_mb
	observed["gpu_temp_c"] = temp_c
	if free_mb < min_free_vram_mb:
		return False, f"VRAM busy ({free_mb} MB free < {min_free_vram_mb}) — someone is using the GPU", observed
	if temp_c > max_gpu_temp_c:
		return False, f"thermal fever ({temp_c}°C > {max_gpu_temp_c}°C)", observed
	now = time.time() if now is None else now
	signals = [Path(p).expanduser() for p in (activity_signals or DEFAULT_ACTIVITY_SIGNALS)]
	existing = [p for p in signals if p.exists()]
	if not existing:
		return False, "no operator-presence signal source available — refusing to assume absence", observed
	newest = max(_latest_mtime(p) for p in existing)
	idle_for = now - newest
	observed["operator_idle_s"] = int(idle_for)
	if idle_for < idle_seconds:
		return False, f"operator active {int(idle_for)}s ago (< {idle_seconds}s) — not alone", observed
	return True, f"GPU free ({free_mb} MB, {temp_c}°C), operator idle {int(idle_for)}s", observed


def build_contained_command(
	script_file: str,
	config_path: str,
	memory_max: str = MEMORY_MAX,
	runtime_max_s: int = RUNTIME_MAX_SECONDS,
) -> list[str]:
	return [
		"systemd-run", "--user", "--scope",
		"-p", f"MemoryMax={memory_max}",
		"-p", f"RuntimeMaxSec={runtime_max_s}",
		"env", "PYTHONPATH=.", ".venv/bin/python", script_file, "--config", config_path,
	]


def check_containment(cmd: list[str]) -> tuple[bool, str]:
	has_cgroup = "systemd-run" in cmd and any(str(a).startswith("MemoryMax=") for a in cmd)
	has_timeout = any(str(a).startswith("RuntimeMaxSec=") for a in cmd)
	if has_cgroup and has_timeout:
		return True, "cgroup memory cap + hard runtime limit present"
	return False, "launch command lacks cgroup cap or runtime limit — containment incomplete"


def check_budget(ledger_path: str, max_per_day: int = MAX_AUTONOMOUS_RUNS_PER_DAY, today: str | None = None) -> tuple[bool, str]:
	today = today or datetime.now(UTC).astimezone().strftime("%Y-%m-%d")
	fired = 0
	ledger = Path(ledger_path)
	if ledger.exists():
		with open(ledger, encoding="utf-8") as f:
			for line in f:
				try:
					entry = json.loads(line)
				except json.JSONDecodeError:
					continue
				if entry.get("fire") and entry.get("timestamp", "").startswith(today):
					fired += 1
	if fired >= max_per_day:
		return False, f"daily budget exhausted ({fired}/{max_per_day} autonomous runs today)"
	return True, f"budget available ({fired}/{max_per_day} autonomous runs today)"


def evaluate_gate(
	task: dict,
	base_dir: str,
	cmd: list[str],
	ledger_path: str,
	gpu_probe=_nvidia_probe,
	activity_signals: list[str] | None = None,
	now: float | None = None,
) -> GateDecision:
	decision = GateDecision(fire=False, task_id=str(task.get("id", "?")))
	consent_ok, consent_why = check_consent(task, base_dir)
	hw_ok, hw_why, observed = check_hardware(activity_signals=activity_signals, gpu_probe=gpu_probe, now=now)
	contain_ok, contain_why = check_containment(cmd)
	budget_ok, budget_why = check_budget(ledger_path)
	decision.checks = {"consent": consent_ok, "hardware": hw_ok, "containment": contain_ok, "budget": budget_ok}
	decision.reasons = [consent_why, hw_why, contain_why, budget_why]
	decision.observed = observed
	decision.fire = all(decision.checks.values())
	return decision


def record_decision(ledger_path: str, decision: GateDecision) -> None:
	entry = {
		"timestamp": datetime.now(UTC).astimezone().isoformat(),
		"task_id": decision.task_id,
		"fire": decision.fire,
		"checks": decision.checks,
		"reasons": decision.reasons,
		"observed": decision.observed,
	}
	ledger = Path(ledger_path)
	ledger.parent.mkdir(parents=True, exist_ok=True)
	with open(ledger, "a", encoding="utf-8") as f:
		f.write(json.dumps(entry, ensure_ascii=False) + "\n")
