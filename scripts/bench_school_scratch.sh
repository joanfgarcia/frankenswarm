#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# EXP_079 Tier 2 — A/B/C de convergencia from-scratch (FP32 vs BF16 vs BF16+compile)
#
# Tres runs desde cero hasta el fin del stage 1 (epoch 160, hito 2_years mock),
# misma semilla, sandboxes bajo storage/benchmarks/bf16_scratch/. El run vivo
# NO se toca: md5 de model_current.pt verificado antes y después de cada brazo.
#
#   ./scripts/bench_school_scratch.sh              # los tres brazos, secuencial
#   ARMS="arm_bf16" ./scripts/bench_school_scratch.sh   # un brazo concreto
#
# Pre-registro y umbrales: docs/experiments/EXP_079_DESIGN.md (congelados).
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
SEED=770
EPOCHS=160
BENCH_DIR="storage/benchmarks/bf16_scratch"
LIVE_CKPT="storage/checkpoints/sovereign_school/model_current.pt"
LIVE_STATE="storage/checkpoints/sovereign_school/school_state.json"
ARMS="${ARMS:-arm_fp32 arm_bf16 arm_bf16_compile}"

mkdir -p "$BENCH_DIR"
export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONUNBUFFERED=1

md5_live() { md5sum "$LIVE_CKPT" "$LIVE_STATE" | md5sum | cut -d' ' -f1; }
MD5_REF="$(md5_live)"
echo "🔒 huella del run vivo: ${MD5_REF}"

# GPU exclusiva: vaciar la VRAM del kernel si está delante (best-effort)
if curl -fsS -X POST "http://127.0.0.1:8760/v1/unload" -o /dev/null 2>/dev/null; then
	echo "→ VRAM del kernel liberada (proxy dual-bind)."
	sleep 2
fi

flags_for() {
	case "$1" in
		arm_fp32)         echo "--amp off" ;;
		arm_bf16)         echo "--amp bf16" ;;
		arm_bf16_compile) echo "--amp bf16 --compile" ;;
		*) echo "brazo desconocido: $1" >&2; return 1 ;;
	esac
}

run_arm() {
	local name="$1"
	local extra
	extra="$(flags_for "$name")"
	local log="$BENCH_DIR/${name}.log"
	echo ""
	echo "═══ EXP_079 · brazo ${name} (${extra}) — $(date '+%F %H:%M') ═══"

	# shellcheck disable=SC2086
	local cmd=("$PYTHON" src/bitnet/training/train_sovereign_school.py \
		--batch_size 64 --seed "$SEED" --max_epochs_per_run "$EPOCHS" \
		--test_mock --state_dir "$BENCH_DIR/$name" $extra)

	if command -v systemd-run >/dev/null 2>&1; then
		systemd-inhibit --what=sleep --who="frankenswarm" --why="EXP_079 Tier 2 (${name})" \
			systemd-run --user --scope --quiet -p MemoryMax=16G "${cmd[@]}" 2>&1 | tee "$log"
	else
		"${cmd[@]}" 2>&1 | tee "$log"
	fi

	if [[ "$(md5_live)" != "$MD5_REF" ]]; then
		echo "🛑 ¡La huella del run vivo CAMBIÓ tras ${name}! Abortando el resto." >&2
		exit 1
	fi
	echo "✅ ${name} completo · run vivo intacto"
}

for arm in $ARMS; do
	run_arm "$arm"
done

echo ""
echo "🏁 EXP_079 Tier 2 completo. Logs: ${BENCH_DIR}/arm_*.log"
echo "   Siguiente: extraer curvas de val_loss y correr milestone_battery.py"
echo "   sobre ${BENCH_DIR}/*/model_current.pt (umbrales en EXP_079_DESIGN.md §3)."
