import sys
import os

# Ajustar PYTHONPATH para permitir importaciones locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tokenizer.translator import VectorTranslator
from src.nodes.mock_experts import MockBitNetNode
from src.router.mock_prolog import MockPrologRouter

def run_phase_1_pipeline():
    print("\n=== Frankenswarm Fase 1: Inicializando Pipeline ===")
    
    # 1. Instanciar el traductor Universal
    print("[1] Inicializando Traductor (all-MiniLM-L6-v2)...")
    translator = VectorTranslator()
    
    # Conceptos para la ontología base del Mock
    ontology = [
        "código python", 
        "lógica matemática", 
        "resumen general",
        "error de sintaxis",
        "saludo cordial"
    ]
    translator.add_concepts(ontology)
    
    # 2. Registrar los Nodos Falsos (Mock Experts)
    print("[2] Registrando Expertos BitNet Falsos...")
    nodes = {
        "Experto_Core": MockBitNetNode("Experto_Core", "Programación Estructural"),
        "Experto_Oraculo": MockBitNetNode("Experto_Oraculo", "Lógica Abstracta"),
        "Experto_Generalista": MockBitNetNode("Experto_Generalista", "Comprensión del Lenguaje Natural")
    }
    
    # 3. Inicializar el Árbitro (Router)
    print("[3] Inicializando Árbitro Prolog...\n")
    router = MockPrologRouter(nodes)
    
    # 4. Probar flujos
    test_inputs = [
        "Por favor genera un script en python asíncrono.",
        "Resuelve la siguiente ecuación diferencial usando lógica matemática pura.",
        "Hola Frankenswarm, ¿cómo te encuentras hoy?"
    ]
    
    for idx, prompt in enumerate(test_inputs):
        print(f"--- Flujo {idx + 1} ---")
        print(f"👤 Humano: '{prompt}'")
        
        # A. Traducción Humano -> Vector
        input_vector = translator.encode(prompt)
        
        # B. Triage Cognitivo (El Traductor le pasa "pistas" al Router simulado)
        # En la realidad, Prolog evaluaría propiedades matemáticas del tensor
        semantic_hints = translator.decode(input_vector, top_k=2)
        
        # C. Enrutamiento
        target_node_name = router.evaluate_and_route(input_vector, semantic_hints)
        active_expert = nodes[target_node_name]
        
        # D. Ejecución del Nodo (Propagación Forward simulada)
        output_vector = active_expert.forward(input_vector)
        
        # E. Traducción Vector -> Humano
        final_hints = translator.decode(output_vector, top_k=1)
        print(f"🤖 Output Vectorial decodificado como cercano a: '{final_hints[0][0]}' (Similitud: {final_hints[0][1]:.4f})\n")

if __name__ == "__main__":
    run_phase_1_pipeline()
