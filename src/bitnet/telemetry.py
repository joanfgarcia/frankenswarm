"""
telemetry.py — Structured experiment logging for PopuLoRA training.

Writes JSONL (one JSON object per line) for machine-readable analysis.
Each experiment gets its own file: storage/telemetry/EXP_NNN.jsonl

Two record types:
- "step": Per-step granular data (loss, predictions, tau, agents)
- "epoch": Per-epoch aggregated data (accuracy, fitness, evolution)

Usage:
    logger = ExperimentLogger("003", params={...})
    logger.log_step(epoch=1, step=0, loss=3.5, ...)
    logger.log_epoch(epoch=1, loss_avg=3.5, ...)
    logger.close()

Analysis:
    df = ExperimentLogger.load("003")
    steps = df[df["type"] == "step"]
    epochs = df[df["type"] == "epoch"]
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _json_default(obj):
	"""Handle numpy types for JSON serialization."""
	if isinstance(obj, (np.integer,)):
		return int(obj)
	if isinstance(obj, (np.floating,)):
		return float(obj)
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _get_telemetry_dir() -> str:
	base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	d = os.path.join(base, "storage", "telemetry")
	os.makedirs(d, exist_ok=True)
	return d


class ExperimentLogger:
	"""Structured JSONL logger for PopuLoRA experiments."""

	def __init__(self, experiment_id: str, params: dict[str, Any]):
		self.experiment_id = experiment_id
		self.path = os.path.join(_get_telemetry_dir(), f"EXP_{experiment_id}.jsonl")
		self._file = open(self.path, "a", encoding="utf-8")
		self._start_time = time.monotonic()

		# Write header record
		self._write({
			"type": "header",
			"experiment_id": experiment_id,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"params": params,
		})

	def _write(self, record: dict) -> None:
		self._file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
		self._file.flush()

	def log_step(
		self,
		epoch: int,
		step: int,
		loss: float,
		loss_concept: float,
		loss_emotion: float,
		tau: float,
		speaker_id: int,
		listener_id: int,
		target_concept: str,
		target_emotion: str,
		pred_concept: str,
		pred_emotion: str,
		concept_correct: int,
		emotion_correct: int,
		joint_correct: int,
		batch_size: int,
		message_tokens: list[str] | None = None,
		tf_ratio: float | None = None,
	) -> None:
		"""Log a single training step."""
		record = {
			"type": "step",
			"epoch": epoch,
			"step": step,
			"elapsed_s": round(time.monotonic() - self._start_time, 3),
			"loss": round(loss, 6),
			"loss_concept": round(loss_concept, 6),
			"loss_emotion": round(loss_emotion, 6),
			"tau": round(tau, 4),
			"speaker_id": speaker_id,
			"listener_id": listener_id,
			"target_concept": target_concept,
			"target_emotion": target_emotion,
			"pred_concept": pred_concept,
			"pred_emotion": pred_emotion,
			"concept_correct": concept_correct,
			"emotion_correct": emotion_correct,
			"joint_correct": joint_correct,
			"batch_size": batch_size,
		}
		if message_tokens is not None:
			record["message_tokens"] = message_tokens
		if tf_ratio is not None:
			record["tf_ratio"] = round(tf_ratio, 4)
		self._write(record)

	def log_epoch(
		self,
		epoch: int,
		loss_avg: float,
		acc_concept: float,
		acc_emotion: float,
		acc_joint: float,
		fitness: list[float],
		worst_agent: int,
		parent_a: int,
		parent_b: int,
	) -> None:
		"""Log epoch-level aggregated metrics."""
		self._write({
			"type": "epoch",
			"epoch": epoch,
			"elapsed_s": round(time.monotonic() - self._start_time, 3),
			"loss_avg": round(loss_avg, 6),
			"acc_concept": round(acc_concept, 4),
			"acc_emotion": round(acc_emotion, 4),
			"acc_joint": round(acc_joint, 4),
			"fitness": [round(f, 6) for f in fitness],
			"worst_agent": worst_agent,
			"parent_a": parent_a,
			"parent_b": parent_b,
		})

	def log_event(self, event: str, data: dict[str, Any] | None = None) -> None:
		"""Log a freeform event (promotion, early stop, error, etc.)."""
		self._write({
			"type": "event",
			"event": event,
			"elapsed_s": round(time.monotonic() - self._start_time, 3),
			"data": data or {},
		})

	def close(self) -> None:
		"""Flush and close the log file."""
		self.log_event("experiment_end")
		self._file.close()

	@staticmethod
	def load(experiment_id: str):
		"""Load experiment data as a pandas DataFrame."""
		import pandas as pd

		path = os.path.join(_get_telemetry_dir(), f"EXP_{experiment_id}.jsonl")
		records = []
		with open(path, encoding="utf-8") as f:
			for line in f:
				records.append(json.loads(line))
		return pd.DataFrame(records)

	@staticmethod
	def load_epochs(experiment_id: str):
		"""Load only epoch-level records for quick analysis."""
		import pandas as pd

		df = ExperimentLogger.load(experiment_id)
		return df[df["type"] == "epoch"].reset_index(drop=True)

	@staticmethod
	def load_steps(experiment_id: str):
		"""Load only step-level records for granular analysis."""
		import pandas as pd

		df = ExperimentLogger.load(experiment_id)
		return df[df["type"] == "step"].reset_index(drop=True)

	@staticmethod
	def compare(*experiment_ids: str):
		"""Load epoch data from multiple experiments for comparison."""
		import pandas as pd

		frames = []
		for eid in experiment_ids:
			df = ExperimentLogger.load_epochs(eid)
			df["experiment"] = eid
			frames.append(df)
		return pd.concat(frames, ignore_index=True)
