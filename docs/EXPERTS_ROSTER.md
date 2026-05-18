# Frankenswarm Experts Roster (Phase 1: Dogfooding)

Este documento define la topología inicial de los modelos especializados (Mixture of Experts desacoplado) requeridos para auto-sostener el ecosistema `red-pill`. 

**Filosofía Central:**
1. **Modelos Minúsculos (1.58b)**: Entrenados/finetuneados para un único dominio.
2. **Cero Contaminación**: Pureza del espacio latente para evitar alucinaciones sintácticas cruzadas.
3. **Evolución Continua**: Los pesos no son estáticos. Se aplicará *Continuous Fine-Tuning* (ej. DPO local nocturno) utilizando los *diffs* en verde de nuestro repositorio para que los expertos asimilen nuevas librerías y estándares (ej. Python 3.12+ o utilidades modernas de Linux).

---

## 0. El Orquestador Generalista (El "Front-End" / La "IA Tonta")
*   **Misión**: Ser la interfaz humana. Su conocimiento es extremadamente ancho pero nulo en profundidad. Sabe qué es una integral, pero no sabe resolverla. No ejecuta el trabajo pesado, lo delega.
*   **Input**: Lenguaje natural puro del humano (conversación).
*   **Output**: Saludo humano y *Tool-Calling* (llamadas a herramientas y paso de parámetros).
*   **Evolución**: Entrenamiento hiper-optimizado en *Function Calling* y empatía/conversación. Es el único modelo que puede permitirse ser ligeramente más grande (ej. 8B parámetros cuantizados) ya que reside constantemente en memoria como "recepcionista".

---

## 1. El Oráculo (Experto en Prolog / Lógica Simbólica)
*   **Misión**: Actuar como el **Árbitro / Router MoE** del enjambre. Triage cognitivo, planificación de dependencias y evaluación lógica pura tomando decisiones estrictas de enrutamiento.
*   **Input**: Embeddings (Vectores Multidimensionales) / Problema abstracto.
*   **Output**: Decisiones de enrutamiento inquebrantables, Código ISO Prolog válido.
*   **Evolución**: **Casi Estática**. La sintaxis es longeva; mutará para optimizar su capacidad de evaluar restricciones vectoriales.

## 2. El Ingeniero Genético (Experto en Lisp / Metaprogramación)
*   **Misión**: Orquestador principal de la evolución (NEAT) y el crecimiento (Net2Net). Trata las arquitecturas neuronales como datos, mutándolas y reestructurándolas en caliente (REPL) sin detener el enjambre.
*   **Input**: ASTs, estructuras de la red y métricas de *fitness*.
*   **Output**: Nuevas topologías de red, macros y alteraciones dinámicas del código del enjambre.
*   **Evolución**: **Dinámica**. Requiere alta flexibilidad para adaptarse a las mutaciones estructurales del ecosistema.

## 3. El Ingeniero Core (Experto en Python / AsyncIO)
*   **Misión**: Desarrollar e iterar `red-pill`. Escritura de daemons, integración con Qdrant/SQLite, y código de enrutamiento (SwarmRouter).
*   **Input**: Requisito de software / Issue de Bug.
*   **Output**: Código Python "idiomatic" y asíncrono.
*   **Evolución**: **Dinámica**. Requiere retroalimentación constante sobre nuestro estilo de código (pep8 estricto) y la adopción de features futuras de Python.

## 4. El Operador SRE (Experto en Bash / Systemd)
*   **Misión**: Gestión de infraestructura a nivel de SO local. Configurar *timers*, blindar procesos con `cgroups` (OOM Shields), e interactuar con utilidades del disco bajo el estándar XDG.
*   **Input**: Orden operativa.
*   **Output**: Scripts bash y archivos `.service`/`.timer` de Systemd.
*   **Evolución**: **Media**. Adaptable a la irrupción de nuevas CLI tools (ej. `rg`, `fd`, `bat`) que reemplacen a los clásicos de GNU.

## 5. El Arqueólogo AST (Experto en Cypher / Grafos)
*   **Misión**: Operar `Graphify`. Diseñar consultas complejas de bases de datos de grafos para la navegación del código fuente y recuperación semántica de la memoria profunda.
*   **Input**: Intención de búsqueda de código.
*   **Output**: Sintaxis Cypher estricta.
*   **Evolución**: **Adaptativa**. Basada en el crecimiento orgánico de la ontología y esquemas del repositorio.

## 6. El Archivista de Metadatos (Experto en JSON / YAML / MD)
*   **Misión**: Producción de serialización inmaculada. Escritura de reportes para el Minion Inbox, archivos `.agent/ATLAS.md`, y plantillas `SKILL.md` sin romper el *frontmatter*.
*   **Input**: Datos crudos de otros procesos.
*   **Output**: Formato estructurado 100% válido.
*   **Evolución**: **Baja**. Alta dependencia en el determinismo de la sintaxis estructural. Tolerancia 0% a alucinaciones.

---
*Documento vivo. Para ser ampliado en Fase 2 (Community Playground) una vez las dependencias de red-pill estén auto-sostenidas.*
