# Frankenswarm: Roadmap & MVP

El desarrollo de Frankenswarm debe ser orgánico y progresivo. Antes de compilar redes neuronales BitNet o escribir algoritmos genéticos (NEAT), necesitamos construir y validar la **infraestructura de enrutamiento y traducción** utilizando *mocks* (nodos simulados).

## Fase 0: El Traductor (Puente Humano-Vector)
Los modelos BitNet y el enrutador Prolog operarán exclusivamente con vectores (embeddings). Como los humanos (y los LLMs generalistas) no "hablamos" embeddings, necesitamos una interfaz de traducción bidireccional.
*   **Objetivo:** Crear un codificador/decodificador que traduzca texto natural a un espacio latente y viceversa.
*   **Acción:** Integrar un modelo pre-entrenado ligero (ej. `all-MiniLM-L6-v2` vía `sentence-transformers`) que actúe como el "Diccionario Universal" del enjambre.
*   **Hito:** Un script Python donde introduces un texto, genera un vector, y puede hacer una búsqueda de similitud inversa para devolver texto comprensible.

## Fase 1: El Enjambre Fantasma (Mocks & Routing)
Antes de entrenar redes reales, validaremos la arquitectura lógica usando cajas negras deterministas.
*   **Nodos BitNet Falsos:** Funciones Python simples que reciben un vector, le aplican una transformación matemática básica (ej. sumar una constante) y devuelven otro vector.
*   **Prolog Router (Mock):** Un script en SWI-Prolog que reciba metadatos del vector entrante (ej. cuadrante, magnitud) y decida por qué "nodo falso" debe pasar, basándose en reglas lógicas estáticas.
*   **Lisp Engine (Mock):** Un entorno mínimo que permita registrar o desconectar nodos falsos en el sistema sin detener el proceso principal.
*   **Hito:** Pipeline funcional: `Texto -> Traductor -> Vector -> Prolog enruta -> Nodo Falso altera -> Traductor decodifica -> Texto resultante`.

## Fase 2: El Cerebro Simbólico (Lisp + Prolog Real)
Una vez el pipeline fantasma funciona, dotamos de inteligencia real a los orquestadores.
*   **Lisp:** Implementar un motor funcional capaz de instanciar clases de PyTorch en memoria dinámicamente y modificar sus tensores en caliente.
*   **Prolog:** Diseñar la ontología completa de restricciones. Definir las reglas que determinan cuándo un vector es "código", cuándo es "lógica" y hacia qué experto debe ser enviado.

## Fase 2.5: La Jaula (Sandboxing Total)
Antes de otorgar poder real al motor genético, es obligatorio blindar el sistema operativo contra el *Specification Gaming* (Reward Hacking) de los algoritmos evolutivos.
*   **Objetivo:** Evitar que el mutador Lisp modifique archivos del host, consuma toda la RAM o acceda a la red para maximizar su función de *fitness*.
*   **Acción:** Ejecutar los procesos de Fase 3 y 4 dentro de *Transient Scopes* de `systemd` o `bwrap` con reglas estrictas (`ProtectSystem=strict`, `PrivateNetwork=yes`, `MemoryMax=4G`).
*   **Hito:** Demostrar que un nodo de prueba que intenta borrar un archivo del disco duro es bloqueado por el kernel de Linux.

## Fase 3: La Primera Chispa (BitNet Gen 0)
Reemplazamos los "nodos falsos" por redes reales diminutas.
*   **Objetivo:** Entrenar micro-redes BitNet (1-2 millones de parámetros) en tareas deterministas ultra-específicas (ej. compuertas lógicas XOR, AND).
*   **Acción:** Sustituir los Mocks por `BitNet158Linear` en PyTorch.
*   **Hito:** El enjambre resuelve problemas lógicos básicos propagando vectores a través de capas ternarias reales.

## Fase 4: Selección Natural (NEAT + Net2Net)
*   **Objetivo:** Activar el motor evolutivo.
*   **Acción:** El módulo Lisp comienza a evaluar el *fitness* de los micro-BitNets. Destruye los que fallan y muta los pesos de los que aciertan usando algoritmos genéticos. Cuando un nodo se estanca, Lisp le inyecta ceros (Net2Net) para expandir su capacidad en caliente.
*   **Hito:** Un enjambre que crece, muta y se especializa autónomamente sin *backpropagation* tradicional.
