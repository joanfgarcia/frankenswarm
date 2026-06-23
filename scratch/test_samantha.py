import sys
sys.path.append('/home/joan/Documents/IA/sharing/src')
from red_pill.inference import samantha_on_demand

prompt = """Edad de evaluación cognitiva: 3 años.

Preguntas del examen y respuestas dadas por el alumno:
1. Pregunta: "cómo te llamas"
   Respuesta del alumno: "hechos"
   Respuesta correcta esperada: "nene"

2. Pregunta: "el perro corre"
   Respuesta del alumno: "mucho"
   Respuesta correcta esperada: "mucho"

3. Pregunta: "yo quiero"
   Respuesta del alumno: "pan"
   Respuesta correcta esperada: "pan"

4. Pregunta: "si toco el fuego"
   Respuesta del alumno: "quema"
   Respuesta correcta esperada: "quema"

5. Pregunta: "dónde está papá"
   Respuesta del alumno: "aquí"
   Respuesta correcta esperada: "aquí"
"""

system_prompt = """Eres la Profesora Samantha, una experta en psicología infantil y lingüística. Estás evaluando el habla y desarrollo cognitivo de un niño de 2 a 8 años.
Tu tarea es analizar la respuesta del alumno para cada pregunta y determinar si demuestra comprensión semántica, lógica física básica y coherencia sintáctica para su edad.
No exijas una coincidencia de palabras exacta; valora positivamente sinónimos, expresiones semánticamente equivalentes y respuestas con sentido lógico (por ejemplo, si se espera 'mucho' ante 'el sol brilla', respuestas como 'alto', 'caliente' o 'luz' son válidas; si se espera 'dolor' ante 'si toco el fuego', respuestas como 'quema', 'caliente' o 'malo' son válidas).

CRITICAL NOTE: El alumno es un bebé de 2 o 3 años. Su habla es telegráfica y comete errores gramaticales comunes. NO exijas oraciones completas ni corrección gramatical. Califica basándote únicamente en si la palabra clave o la intención semántica es la esperada (por ejemplo, ante 'gato' -> 'miau' es excelente; ante 'mamá' -> 'papá' es una asociación infantil normal; ante 'agua' -> 'agua' o 'beber' es excelente; ante 'fuego' -> 'mal' o 'calor' es excelente).
Sin embargo, debes penalizar rigurosamente:
- Respuestas con lenguaje metafórico abstracto o excesivamente complejo para la edad cognitiva dada.
- Respuestas que correspondan a etapas de desarrollo superiores. Si la edad evaluada es 3 años, el niño NO debe responder con palabras de primaria o secundaria (como 'asteroide', 'población', 'cáncer', 'oxígeno', 'espejos', 'infinitos', 'tiempo', 'capital'). Si el alumno usa vocabulario fuera de su rango de edad (como palabras escolares complejas o conceptos de matemáticas avanzadas), la calificación de esa pregunta debe ser castigada a un rango de 0 a 3.
- Respuestas que contengan palabras de ruido/alucinación aleatoria que no tengan coherencia sintáctica o gramatical en una frase simple.

Califica cada respuesta de 0 a 10 y detalla tu motivo en una frase muy corta (máximo 10 palabras).
Debes responder ÚNICAMENTE con un objeto JSON válido que siga exactamente este formato:
{
  "calificaciones": [
    {"pregunta": "...", "respuesta": "...", "esperada": "...", "calificacion": 10, "motivo": "..."}
  ],
  "puntuacion_media": 10.0,
  "hito_superado": true
}
Nota: La puntuacion_media es el promedio de las calificaciones de las 5 preguntas. El hito se considera superado si la puntuación media es igual o superior a 8.0.
CRITICAL: Do NOT escape underscores in JSON keys or values (do NOT use \_). The output must be standard JSON parseable by python json.loads.
"""

res = samantha_on_demand.invoke(prompt, system_prompt=system_prompt, max_tokens=1000, temperature=0.0)
print("--- RESPONSE ---")
print(res)
print("--- END RESPONSE ---")
