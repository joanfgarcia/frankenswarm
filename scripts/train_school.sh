#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Escuela Soberana de Bit — runner autónomo, por épocas.
#
# NO requiere red-pill. Frankenswarm entrena a Bit por sí solo; el kernel es
# una comodidad opcional (cola, reanudación desatendida, supervisión), nunca
# un requisito. Ver docs/TRAINING_BIT.md para las dos vías.
#
#   ./scripts/train_school.sh                 # entrena hasta graduarse
#   ./scripts/train_school.sh --epochs 5      # solo 5 épocas y para
#   ./scripts/train_school.sh --status        # dónde va, sin entrenar nada
#
# Corta con Ctrl-C cuando quieras: cada época deja checkpoint, así que se
# retoma exactamente donde estaba.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MEMORY_MAX="${MEMORY_MAX:-16G}"   # 10G disparaba el OOM killer con el modelo a 896 dim
STATE_FILE="storage/checkpoints/sovereign_school/school_state.json"
LOG_DIR="storage/logs"
MAX_EPOCHS=0                      # 0 = hasta graduarse

while [[ $# -gt 0 ]]; do
	case "$1" in
		--epochs) MAX_EPOCHS="$2"; shift 2 ;;
		--batch-size) BATCH_SIZE="$2"; shift 2 ;;
		--status) SHOW_STATUS=1; shift ;;
		-h|--help) sed -n '2,16p' "$0"; exit 0 ;;
		*) echo "Opción desconocida: $1 (usa --help)" >&2; exit 2 ;;
	esac
done

[[ -x "$PYTHON" ]] || { echo "No encuentro el intérprete '$PYTHON'. Crea el venv o exporta PYTHON=..." >&2; exit 1; }

TRAINER="src/bitnet/training/train_sovereign_school.py"
# El entrenador usa parse_known_args(): si el flag por el que troceamos no
# existe, lo ignoraría EN SILENCIO y cada "época" entrenaría el currículo
# entero. Mejor negarse a arrancar que prometer un troceo que no ocurre.
grep -q -- "--max_epochs_per_run" "$TRAINER" || {
	echo "El entrenador no acepta --max_epochs_per_run: sin él no se puede trocear por épocas" >&2
	echo "y una interrupción costaría todo el currículo. Actualiza ${TRAINER} antes de continuar." >&2
	exit 1
}

# ── Estado actual, leído del checkpoint que mantiene el propio entrenador ──
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
	echo "── Escuela Soberana de Bit ──"
	resumen
	exit 0
fi

# ── Liberar la GPU si el ecosistema red-pill está presente (opcional) ──────
# Best-effort puro: si no hay proxy ni servicio, no pasa nada y se entrena igual.
liberar_vram() {
	if curl -fsS -X POST "http://127.0.0.1:8760/v1/unload" -o /dev/null 2>/dev/null; then
		echo "→ VRAM liberada en el proxy dual-bind (el servicio sigue vivo)."
		sleep 2
	elif systemctl --user is-active --quiet redpill-llm 2>/dev/null; then
		echo "→ Parando redpill-llm para liberar VRAM (se restaura al salir)."
		systemctl --user stop redpill-llm
		RESTAURAR_LLM=1
	fi
}

limpieza() {
	if [[ "${RESTAURAR_LLM:-0}" == "1" ]]; then
		echo "→ Restaurando redpill-llm."
		systemctl --user start redpill-llm || true
	fi
}
trap limpieza EXIT INT TERM

# ── Una época = un paso atómico ───────────────────────────────────────────
# El entrenador guarda su estado al final de CADA época, así que trocear por
# épocas es lo que hace que interrumpir sea barato y reanudar sea exacto.
una_epoca() {
	local cmd=("$PYTHON" src/bitnet/training/train_sovereign_school.py --batch_size "$BATCH_SIZE" --max_epochs_per_run 1)

	# Bajo systemd: cgroup con tope de memoria (OOM shield) e inhibición de la
	# suspensión automática — dormir el portátil mata el contexto CUDA y fue la
	# causa raíz de los entrenamientos "fritos" de julio de 2026.
	if command -v systemd-run >/dev/null 2>&1; then
		systemd-inhibit --what=sleep --who="frankenswarm" --why="entrenando a Bit" \
			systemd-run --user --scope --quiet -p "MemoryMax=${MEMORY_MAX}" "${cmd[@]}"
	else
		"${cmd[@]}"
	fi
}

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/school_$(date +%Y%m%d_%H%M%S).log"
export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONUNBUFFERED=1

echo "── Escuela Soberana de Bit ──"
resumen
echo "  log          : ${LOG_FILE}"
echo

liberar_vram

epocas_hechas=0
while true; do
	IFS='|' read -r _ _ _ hitos_antes <<< "$(estado)"
	if [[ "$hitos_antes" == *"8_years"* ]]; then
		echo "🎓 Bit ya se graduó (hito 8_years). Nada que entrenar."
		break
	fi

	inicio=$(date +%s)
	echo "▶ Época en curso (empezada $(date +%H:%M:%S))..."
	if ! una_epoca 2>&1 | tee -a "$LOG_FILE"; then
		echo "✗ La época falló. Revisa ${LOG_FILE}." >&2
		exit 1
	fi
	duracion=$(( $(date +%s) - inicio ))

	epocas_hechas=$(( epocas_hechas + 1 ))
	IFS='|' read -r epoca etapa dim hitos <<< "$(estado)"
	printf '✔ Época guardada en %d min %d s → siguiente: %s (etapa %s, dim %s)\n\n' \
		$(( duracion / 60 )) $(( duracion % 60 )) "$epoca" "$etapa" "$dim"

	[[ "$hitos" == *"8_years"* ]] && { echo "🎓 ¡Bit se ha graduado!"; break; }
	[[ "$MAX_EPOCHS" -gt 0 && "$epocas_hechas" -ge "$MAX_EPOCHS" ]] && { echo "Límite de ${MAX_EPOCHS} época(s) alcanzado."; break; }
done

echo "Listo. ${epocas_hechas} época(s) en esta sesión."
