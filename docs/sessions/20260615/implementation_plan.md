# Plan de Currículo Piagetiano Estructurado y Descontaminación de Datos

Este plan propone reconstruir el currículo escolar de Bit mediante un pipeline de generación sintética y traducción adaptativa 100% en español, utilizando a Samantha. Cada oración generada será catalogada en un archivo JSON estructurado con metadatos de categoría, intención educativa y primos semánticos, completando el corpus CHILDES sin perder el anclaje en los Primos NSM.

## Fundamentación y Alineación (CHILDES + NSM)

Para no reinventar la rueda y mantener la rigurosidad científica, alineamos nuestra base de datos con los estándares de adquisición de lenguaje infantil:

1. **Estructuración del Corpus CHILDES (Estado del Arte)**:
   - En la comunidad científica, CHILDES no tiene un etiquetado semántico universal por defecto. Se clasifica principalmente por **MLU** (Mean Length of Utterance / Complejidad Sintáctica) y edad de desarrollo.
   - Para la categorización semántica, se suele correlacionar con **Wordbank** (que mapea normas de adquisición léxica por categorías como *Animales*, *Acciones*, *Cuerpo*, *Cualidades*) y anotaciones de Roles Semánticos (**CHILDES-SRL**), que definen quién hace qué (Agente-Paciente).
   - *Decisión*: Mantendremos la partición por **MLU** en 4 etapas evolutivas (0-1 año, 1-2 años, 2-3 años, 3-4 años) para procesar el corpus nativo de CHILDES (`childes_pre_school.json`), garantizando la base gramatical y conversacional del habla dirigida a niños (*Motherese*).

2. **Completar CHILDES con Primos Semánticos (NSM)**:
   - CHILDES es naturalista pero disperso: conceptos físicos cruciales como `"fuego"` y `"dolor"` rara vez ocurren juntos en el habla cotidiana grabada (ej. la palabra "dolor" solo aparece 3 veces en todo el corpus preescolar).
   - Para completar esta base, generamos un **Currículo Estructurado Complementario** donde cada oración está catalogada bajo las 65 primitivas del **Metalenguaje Semántico Natural (NSM)** de Cliff Goddard.
   - Cada oración sintética o traducida se almacena con la siguiente taxonomía:
     - `text`: Oración limpia mapeada a nuestro diccionario.
     - `category`: Ámbito cognitivo (física_sensorial, matemáticas, lógica, geografía).
     - `intent`: El concepto relacional esperado (ej. `fuego_causa_dolor`).
     - `primes`: Primos semánticos activados en la oración (ej. `["tocar", "fuego", "sentir", "dolor", "malo"]`).

Esto asegura que Bit aprenda la sintaxis natural de CHILDES mientras su espacio latente se ancla rígidamente a la lógica física elemental del NSM.

## User Review Required

> [!IMPORTANT]
> **Generación de Datos en GPU**: La ejecución del script de generación de currículo `generate_structured_curriculum.py` llamará a Samantha de forma iterativa y consumirá GPU/VRAM. Debemos asegurarnos de que no hay procesos de entrenamiento activos en paralelo.
> 
> **Re-inicialización del Entrenamiento**: Tras generar el nuevo currículo estructurado, se reseteará la base de datos de entrenamiento en `train_sovereign_school.py` y se reiniciará el entrenamiento desde la Época 52 (o Época 1 si se prefiere una purga total de pesos, a evaluar según el tiempo disponible).

## Proposed Changes

### 1. Generador de Currículo Estructurado y Validación (Samantha)

Crearemos un nuevo script para generar, validar y estructurar todo el currículo escolar en español nativo, catalogando cada oración con metadatos cognitivos.

#### [NEW] [generate_structured_curriculum.py](file:///home/joan/Documents/IA/frankenswarm/scripts/generate_structured_curriculum.py)
* **Definición de Primos Semánticos**: Lista de los 65 primos de Wierzbicka (`SEMANTIC_PRIMES` de `glyph_vocabulary.py`).
* **Categorías y Balance Target**:
  - `física_sensorial` (30% - 40%)
  - `acciones_básicas` (25% - 30%)
  - `emociones_fundamentales` (15%)
  - `matemáticas_naturaleza` (15% - 20%)
  - Otras (lengua_conversación, lógica_geografía, literatura_filosofía).
* **Fases y Límites de Longitud**:
  - **Preschool (0-4 años)**: ≤ 8 tokens. Causal física sensoriomotora (calor/dolor/hambre).
  - **Primary (5-6 años)**: ≤ 12 tokens. Aritmética, biología, geografía.
  - **Secondary (7-8 años)**: ≤ 20 tokens. Literatura, física/lógica formal y filosofía.
* **Optimización de Generación (Batching y Resumen)**:
  - **Generación por Bloques (Batching)**: Solicitar a Samantha bloques de 50 elementos estructurados en formato de lista JSON por llamada en lugar de llamadas uno a uno. Esto reduce el tiempo de generación a < 1 hora.
  - **Reanudación (Resume)**: Guardar el progreso de forma incremental. Si la generación se interrumpe, el script cargará el archivo temporal y continuará desde el último lote generado.
  - **Few-shot Prompting**: Incluir ejemplos reales de CHILDES español en el prompt del sistema de Samantha para evitar "habla artificial de IA" y forzar un tono motherese auténtico.
* **Pipeline de Validación (`validate_curriculum_item`)**:
  - Validar que sea un JSON sintáctico correcto.
  - Asegurar que la frase no contenga stop-words en inglés (`and`, `the`, `of`, etc.).
  - Asegurar que todos los primos listados pertenezcan estrictamente a los 65 oficiales.
  - Comprobar que todas las palabras de la frase mapeen a palabras válidas de la base de 15,005 términos en `expanded_glyphs.json`.
  - Descartar cualquier frase que falle las aserciones.
* **Versionado y Hash**: Calcular y guardar el hash SHA-256 del archivo generado en los metadatos del JSON para trazabilidad y reproducibilidad.
* **Guardado**: configs/school_curriculum_structured.json.

---

### 2. Integración en el Dataloader Escolar y Warmup de LR

#### [MODIFY] [train_sovereign_school.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/train_sovereign_school.py)
* **Carga de Datos**: Modificar para leer del nuevo archivo estructurado.
* **Reinicio de Época 1**: Modificar el inicializador para arrancar desde `current_epoch = 1` y `hidden_dim = 128` si se detecta un reset, permitiendo que el modelo purgue por completo la contaminación spanglish.
* **Learning Rate Warmup**: Implementar un scheduler manual de 10 épocas para hacer warmup lineal del learning rate desde `4e-5` a `4e-4` antes de entrar al Scheduler normal, permitiendo aprovechar la plasticidad residual sin desestabilizar los pesos.
* **Auditoría de Metadatos**: Guardar el hash del currículo estructurado en `school_state.json` y en el checkpoint del modelo.
* **Monitoreo de Primos**: Registrar la frecuencia de los primos generados y los gradientes del proyector en cada época para evaluar qué conceptos se están estancando.

---

### 3. Limpieza de Scripts Obsoletos

#### [DELETE] [download_school_curriculum.py](file:///home/joan/Documents/IA/frankenswarm/scripts/download_school_curriculum.py)
* Eliminar el script de descarga Gutenberg contaminado en inglés.

---

## Verification Plan

### Automated Tests
* Ejecutar `scripts/generate_structured_curriculum.py` y comprobar que se genera `configs/school_curriculum_structured.json` con su metadata hash y que pasa el 100% de las validaciones de `validate_curriculum_item`.
* Ejecutar un dry-run de `train_sovereign_school.py` con `--reset_state` para verificar que la época de inicio es 1, el warmup del lr se activa, y el dataloader carga correctamente el nuevo currículo estructurado.

### Manual Verification
* Verificar que en los logs cualitativos no hay ghost words en inglés a lo largo de las primeras épocas.
* Auditar el progreso del modelo a la llegada del primer examen a la Época 64.
