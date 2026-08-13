#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Escuela Soberana de Bit v2 (Formación Nativa K-65P) — Runner Autónomo.
#
# Entrena a Bit v2 sobre secuencias canónicas K-65P con pasadas por época.
# Totalmente compatible con pausar y resumir (Ctrl-C / SIGTERM):
# cada época guarda un checkpoint atómico en disco.
#
# Uso:
#   ./scripts/train_school_k65p.sh                 # entrena hasta graduarse
#   ./scripts/train_school_k65p.sh --epochs 5      # solo 5 épocas y para
#   ./scripts/train_school_k65p.sh --status        # consulta de estado actual
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MEMORY_MAX="${MEMORY_MAX:-16G}"
EMBEDDING="${EMBEDDING:-glyph}"   # glyph = Bit v2 | standard = Bit v0
SEED="${SEED:-770}"               # semilla global (réplicas multi-semilla)
STATE_DIR="${STATE_DIR:-storage/checkpoints/sovereign_school_k65p}"
STATE_FILE="${STATE_DIR}/school_state_k65p.json"
LOG_DIR="storage/logs"
MAX_EPOCHS=0                      # 0 = continuo

while [[ $# -gt 0 ]]; do
	case "$1" in
		--epochs) MAX_EPOCHS="$2"; shift 2 ;;
		--batch-size) BATCH_SIZE="$2"; shift 2 ;;
		--status) SHOW_STATUS=1; shift ;;
		-h|--help) head -n 15 "$0"; exit 0 ;;
		*) echo "Opción desconocida: $1 (usa --help)" >&2; exit 2 ;;
	esac
done

[[ -x "$PYTHON" ]] || { echo "No encuentro el intérprete '$PYTHON'." >&2; exit 1; }

TRAINER="src/bitnet/training/train_sovereign_school_k65p.py"

estado() {
	"$PYTHON" - "$STATE_FILE" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
	print("sin empezar|0|0|")
	raise SystemExit
d = json.loads(path.read_text())
hitos = ",".join(d.get("milestones_achieved", []))
print(f"{d.get('current_epoch', 0)}|{d.get('current_stage_idx', 0)}|{d.get('hidden_dim', 0)}|{hitos}")
PY
}

resumen() {
	IFS='|' read -r epoca etapa dim hitos <<< "$(estado)"
	echo "  época actual : ${epoca}"
	echo "  etapa        : ${etapa} (dim ${dim})"
	echo "  hitos        : ${hitos:-ninguno todavía}"
}

if [[ "${SHOW_STATUS:-0}" == "1" ]]; then
	echo "── Escuela Soberana Bit v2 (K-65P Nativo) ──"
	resumen
	exit 0
fi

liberar_vram() {
	if curl -fsS -X POST "http://127.0.0.1:8760/v1/unload" -o /dev/null 2>/dev/null; then
		echo "→ VRAM liberada en el proxy dual-bind."
		sleep 1
	fi
}

limpieza() {
	echo "→ Sesión de entrenamiento finalizada o pausada de forma segura."
}
trap limpieza EXIT INT TERM

una_epoca() {
	local cmd=("$PYTHON" src/bitnet/training/train_sovereign_school_k65p.py --batch_size "$BATCH_SIZE" --max_epochs_per_run 1 --embedding "$EMBEDDING" --seed "$SEED" --state_dir "$STATE_DIR")

	if command -v systemd-run >/dev/null 2>&1; then
		systemd-inhibit --what=sleep --who="frankenswarm-k65p" --why="entrenando a Bit v2 en K-65P" \
			systemd-run --user --scope --quiet -p "MemoryMax=${MEMORY_MAX}" "${cmd[@]}"
	else
		"${cmd[@]}"
	fi
}

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/school_k65p_${EMBEDDING}_$(date +%Y%m%d_%H%M%S).log"
export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONUNBUFFERED=1

echo "── Escuela Soberana Bit v2 (K-65P Nativo) ──"
resumen
echo "  log          : ${LOG_FILE}"
echo

liberar_vram

epocas_hechas=0
while true; do
	IFS='|' read -r _ _ _ hitos_antes <<< "$(estado)"
	if [[ "$hitos_antes" == *"8_years"* ]]; then
		echo "🎓 Bit v2 ya completó la graduación K-65P (8_years). Nada que entrenar."
		break
	fi

	inicio=$(date +%s)
	echo "▶ Época K-65P en curso (empezada $(date +%H:%M:%S))..."
	set +e
	una_epoca 2>&1 | tee -a "$LOG_FILE"
	rc="${PIPESTATUS[0]}"
	set -e
	if [[ "$rc" -eq 78 ]]; then
		echo "📝 EXAMEN DE HITO SUSPENDIDO: el alumno repite curso (retention). Pausa para revisión del operador."
		echo "   Acta del examen en el estado: ${STATE_FILE} (exam_history). Relanza este script para continuar."
		exit 0
	elif [[ "$rc" -ne 0 ]]; then
		echo "✗ La época falló (rc=${rc}). Revisa ${LOG_FILE}." >&2
		exit 1
	fi
	duracion=$(( $(date +%s) - inicio ))

	epocas_hechas=$(( epocas_hechas + 1 ))
	IFS='|' read -r epoca etapa dim hitos <<< "$(estado)"
	printf '✔ Época K-65P guardada en %d min %d s → siguiente: %s (etapa %s, dim %s)\n\n' \
		$(( duracion / 60 )) $(( duracion % 60 )) "$epoca" "$etapa" "$dim"

	[[ "$MAX_EPOCHS" -gt 0 && "$epocas_hechas" -ge "$MAX_EPOCHS" ]] && { echo "Límite de ${MAX_EPOCHS} época(s) alcanzado."; break; }
done

echo "Listo. ${epocas_hechas} época(s) K-65P ejecutadas en esta sesión."
