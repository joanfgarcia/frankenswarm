# Entrenar a Bit — la Escuela Soberana

Bit aprende por un currículo escolar de ocho etapas (0-1 a 8 años). Cada etapa
termina con un **examen de graduación** que corrige la profesora Samantha: la
etapa no se cierra por número de épocas, sino cuando el hito se concede.

Hay **dos formas de entrenar**, y la primera no necesita nada más que este repo.

| | Autónoma (`scripts/train_school.sh`) | Diferida (red-pill) |
| :--- | :--- | :--- |
| Dependencias | Solo frankenswarm | Kernel red-pill instalado |
| Arranque | Manual, en primer plano | Encolado; un timer lo recoge |
| Si cierras la terminal | Se para (al final de la época) | Sigue |
| Supervisión | El log y `--status` | `job list` / `job status` / `job logs` |
| Cede ante otros trabajos | No | Sí (ciclos nocturnos, VRAM) |

Ambas ejecutan **el mismo entrenador** y comparten **el mismo checkpoint**, así
que puedes empezar por una y seguir por la otra sin perder nada.

---

## 1. Vía autónoma (sin red-pill)

```bash
./scripts/train_school.sh            # entrena hasta que Bit se gradúe
./scripts/train_school.sh --epochs 5 # solo 5 épocas y para
./scripts/train_school.sh --status   # dónde va, sin entrenar nada
```

Variables de entorno útiles: `PYTHON` (intérprete, por defecto `.venv/bin/python`),
`BATCH_SIZE` (64) y `MEMORY_MAX` (16G).

**Una época = un paso.** El entrenador guarda su estado al terminar cada época,
así que el script trocea el trabajo época a época: puedes cortar con `Ctrl-C`
cuando quieras y perderás como mucho la época en curso; al relanzar retoma
exactamente donde estaba. Ese troceo es también lo que hace que un corte de luz
o un reinicio no cuesten un día de cómputo.

Bajo systemd el script añade dos protecciones aprendidas a base de disgustos:

- **`MemoryMax=16G`** (cgroup): con el modelo ya a 896 dimensiones, el límite
  anterior de 10G despertaba al OOM killer a media época.
- **`systemd-inhibit --what=sleep`**: si el portátil se suspende, el contexto
  CUDA muere y el entrenamiento queda "frito". Fue la causa raíz de varios
  entrenamientos perdidos en julio de 2026.

Si el ecosistema red-pill está presente, el script libera la VRAM del modelo
residente antes de empezar (y la restaura al salir), pero es puramente
oportunista: sin red-pill delante, entrena igual.

### Precisión mixta (BF16)

Desde el 27-jul-2026 el entrenador acepta `--amp {auto,bf16,off}` (por defecto
`auto`: BF16 si la GPU lo soporta). Implementa la fase 1 de RFC-BITNET-VRAM-001
como **autocast con pesos maestros FP32**: las activaciones —el ~85% de la
VRAM— se computan en BF16, pero los pesos, gradientes y estados de AdamW siguen
en FP32. Dos consecuencias que importan:

- **`model_current.pt` no cambia de formato.** FP32 y BF16 son intercambiables
  por ejecución: el benchmark de convergencia y cualquier rollback cuestan
  exactamente un flag (`--amp off`). Ningún checkpoint ni la neurogénesis
  (`net2wider`) se ven afectados.
- **Telemetría por época**: el log añade `VRAM pico` y `∇STE` (norma del
  gradiente de la primera BitLinear). Si `∇STE` cae a cero, el estimador
  straight-through se ha roto en silencio — es la señal de vigilancia que
  exige el RFC (§4.8.1), no un adorno.

### Dónde mirar

- Log de la sesión: `storage/logs/school_<fecha>.log`
- Estado: `storage/checkpoints/sovereign_school/school_state.json`
  (`current_epoch`, `current_stage_idx`, `milestones_achieved`)
- Pesos: `model_current.pt`, y un `model_milestone_<edad>_years.pt` por graduación

---

## 2. Vía diferida (con red-pill)

Útil cuando quieres que el entrenamiento siga sin ti, ceda la GPU a los ciclos
nocturnos y quede supervisado. La receta ya está escrita y **versionada** en
[`configs/jobs/school.yaml`](../configs/jobs/school.yaml):

```bash
red-pill job submit --recipe school     # desde cualquier punto del repo
```

No hay que escribir ningún JSON: la receta describe en YAML el comando, el
fichero de checkpoint, cómo leer el progreso y cuándo se considera terminado.
Vive en este repositorio a propósito — describe cómo se ejecuta *este* proyecto,
así que le pertenece y viaja con su historia, no con la del kernel.

Operativa habitual:

```bash
red-pill job list                  # estado y progreso: "998/1408 (70%) · etapa 7/8"
red-pill job status <id>           # duración media por step y cadencia real de checkpoint
red-pill job logs <id> --tail 50   # la salida del entrenador
red-pill job pause <id>            # para al terminar la época en curso (no pierde nada)
red-pill job kill <id>             # corta ya; queda PAUSED* y reanudable
red-pill job resume <id>           # continúa desde el checkpoint
```

`pause` es cooperativo (espera a que la época acabe) y `kill` es inmediato: con
épocas de horas, la diferencia entre ambos es real, así que elige según tengas
prisa o quieras conservar la época en vuelo.

### Editar la receta

Si tocas la receta versionada, el cambio afecta a todo el mundo. Para una prueba
puntual y local, copia el fichero a `.red-pill/jobs/school.yaml` (directorio de
estado del kernel, no versionado): red-pill lo busca **antes** que el
versionado, así que hace de override sin ensuciar el repositorio.

---

## 3. Cuánto tarda, y por qué importa

Con el modelo a 896 dimensiones se han medido **épocas de entre 3h20m y 3h50m**
en GPU (en FP32, antes de BF16+SDPA — ambos deberían recortar esa cifra de
forma sustancial; la primera ejecución con `--amp` fijará el dato real), y una
ejecución de casi 11 horas que muy probablemente cayó a CPU tras un
`CUDA OutOfMemoryError` (el entrenador migra solo a CPU en lugar de abortar).

Esa cifra tiene una consecuencia de diseño: como el checkpoint se escribe una
sola vez por época, **la unidad recuperable es la época entera**. Interrumpir a
mitad cuesta horas de cómputo, y por eso la receta declara el trabajo como *no
interrumpible* (`preemptible: false`): sería deshonesto prometer que cede la GPU
en un plazo razonable cuando no puede. Si algún día el entrenador aprende a
guardar por debajo de la época, ese flag debería cambiar.

Corolario práctico: **no lances un entrenamiento justo antes de las 03:00**, la
hora del ciclo de sueño metabólico de red-pill, si te importa que la
consolidación de memoria entre a su hora.

---

## 4. Diagnóstico rápido

| Síntoma | Dónde mirar |
| :--- | :--- |
| No sé por dónde va | `./scripts/train_school.sh --status` |
| Terminó antes de tiempo | El log: busca `PAUSA PLANIFICADA` (límite de épocas) o un traceback |
| El job de red-pill dice que falló | `red-pill job logs <id>` — la salida real del entrenador |
| La GPU está ocupada | `nvidia-smi`; el modelo residente se libera con el proxy dual-bind |
| Se quedó "frito" a media noche | ¿Suspensión del portátil? El script ya inhibe el sleep; verifica que corre bajo systemd |

---

*Bitácora técnica del currículo y decisiones pedagógicas: [`BIT_SCHOOL_NOTEBOOK.md`](BIT_SCHOOL_NOTEBOOK.md).*
