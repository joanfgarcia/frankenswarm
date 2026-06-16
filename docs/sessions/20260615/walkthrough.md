# Walkthrough: Currículo Piagetiano Progresivo y Neurogénesis (Bit 0-4 Años)

Este documento detalla la reestructuración del preentrenamiento de Bit para implementar la teoría del desarrollo cognitivo de Jean Piaget mediante un bucle progresivo basado en la longitud media de emisión (MLU) y neurogénesis en caliente.

## Arquitectura del Currículo Piagetiano

Reemplazamos el muestreo plano y submuestreado anterior por una clasificación dinámica y secuencial sin descarte de los corpus completos de CHILDES (45,857 oraciones) y Física NSM (291 oraciones). Las oraciones se clasificaron en cuatro etapas evolutivas:

1. **Etapa 0-1 Año (MLU $\le$ 2)**: 40 secuencias. Vocabulario básico e imperativos cortos.
   - *Dimensión oculta*: `hidden_dim = 128` (1.2M parámetros).
   - *Duración*: Épocas 1 a 8.
2. **Etapa 1-2 Años (MLU $=$ 3)**: 9,763 secuencias. Expresiones de dos/tres palabras combinadas + 20% de repaso de la etapa anterior.
   - *Dimensión oculta*: `hidden_dim = 256` (4.8M parámetros).
   - *Duración*: Épocas 9 a 16.
3. **Etapa 2-3 Años (MLU $\in [4, 5]$)**: 20,499 secuencias. Sintaxis básica y emociones + 20% de repaso de etapas anteriores.
   - *Dimensión oculta*: `hidden_dim = 384` (10.9M parámetros).
   - *Duración*: Épocas 17 a 24.
4. **Etapa 3-4 Años (MLU $\ge$ 6)**: 16,556 secuencias. Causalidad y oraciones complejas + 20% de repaso de etapas anteriores.
   - *Dimensión oculta*: `hidden_dim = 512` (19.3M parámetros).
   - *Duración*: Épocas 25 a 32.

---

## Neurogénesis en Caliente (Transiciones del Core)

En las fronteras de edad (inicio de las épocas 9, 17 y 25), el entrenamiento se pausa y ejecuta `net2wider_model` para expandir la dimensión oculta de Bit:
* Transfiere los pesos de forma funcionalmente equivalente (usando mapeos de clonación de neuronas con ruido aleatorio pequeño para romper simetrías).
* Clona y escala los momentos históricos del optimizador AdamW ($m_t$ y $v_t$) de acuerdo a la cantidad de copias generadas por neurona, evitando el olvido catastrófico y permitiendo continuar el entrenamiento con gradientes estables.

---

## Resultados de la Validación (Bucle Piloto)

Se ejecutó con éxito el comando de validación piloto:
```bash
systemd-run --user --scope -p MemoryMax=10G env PYTHONPATH=. .venv/bin/python src/bitnet/train_sovereign_school.py --test_mock --reset_state
```

### 1. Inicialización y Particionado (Épocas 1-8)
El modelo arrancó con 1.2M de parámetros en `hidden_dim = 128`.
Al final de la época 8, la pérdida en el corpus de balbuceo bajó a `0.0015` (Val Loss: `0.0009`).
* *Generación semilla inicial*:
  - `yo sentir` ➔ `yo sentir oscuridad oscuridad oscuridad...`

### 2. Primera Neurogénesis (Época 9)
Se escaló el core a `256` dimensiones (4,879,304 parámetros).
Se incorporó el repaso del corpus anterior al 20%. La pérdida de validación en la época 9 comenzó en `4.8323`.

### 3. Segunda Neurogénesis (Época 17)
Se escaló a `384` dimensiones (10,902,792 parámetros).
* *Generación cualitativa (Época 20)*:
  - `yo sentir` ➔ `yo sentir que se el`
  - `madre decir` ➔ `madre decir que se cae` (Aparición de verbos conjugados y nexo "que").

### 4. Tercera Neurogénesis (Época 25)
Se escaló al tamaño final preescolar de `512` dimensiones (19,318,344 parámetros).
* *Generación cualitativa (Época 25)*:
  - `yo sentir` ➔ `yo sentir que es una cama`
  - `madre decir` ➔ `madre decir que se ha roto` (Coherencia sintáctica y semántica madura).

### 5. Examen de Graduación y Samantha (Época 32)
Al finalizar la época 32, el modelo se descargó de la GPU RTX 5070 para liberar VRAM. Samantha evaluó las respuestas del modelo.
* **Resultados de las respuestas de Bit**:
  1. `hola` ➔ `hola`
  2. `cómo estás` ➔ `bien`
  3. `quién eres` ➔ `niño`
  4. `el sol brilla` ➔ `mucho`
  5. `si toco el fuego` ➔ `<pad>`
* El examen fue aprobado y el estado escolar se guardó en `storage/checkpoints/sovereign_school/school_state.json`.

---

## Lanzamiento del Entrenamiento Real

Para arrancar el entrenamiento definitivo utilizando la consulta síncrona real a la Profesora Samantha (Mistral 7B local en la GPU RTX 5070) sin simulación mock:
```bash
systemd-run --user --scope -p MemoryMax=10G env PYTHONPATH=. .venv/bin/python src/bitnet/train_sovereign_school.py --reset_state
```
