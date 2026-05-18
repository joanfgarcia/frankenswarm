import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class VectorTranslator:
    """
    Fase 0: El Traductor (Puente Humano-Vector)
    Actúa como diccionario universal para el ecosistema Frankenswarm.
    Traduce texto humano a embeddings y viceversa mediante similitud de coseno.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Modelo ultra-ligero y rápido para generar los vectores base
        self.model = SentenceTransformer(model_name)
        # Diccionario interno de "Conceptos Conocidos" para decodificación
        self.concept_space: List[str] = []
        self.embedding_space: np.ndarray = np.array([])
        
    def encode(self, text: str) -> np.ndarray:
        """Convierte lenguaje natural en un vector (lenguaje de máquina)."""
        return self.model.encode(text, convert_to_numpy=True)
        
    def add_concepts(self, concepts: List[str]):
        """Añade conceptos conocidos al espacio para poder 'decodificar' vectores."""
        self.concept_space.extend(concepts)
        new_embeddings = self.model.encode(concepts, convert_to_numpy=True)
        
        if self.embedding_space.size == 0:
            self.embedding_space = new_embeddings
        else:
            self.embedding_space = np.vstack([self.embedding_space, new_embeddings])
            
    def decode(self, vector: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Dado un vector arbitrario, encuentra el concepto humano más cercano.
        Útil para interpretar la salida de los nodos falsos.
        """
        if self.embedding_space.size == 0:
            return [("Error: Espacio de conceptos vacío.", 0.0)]
            
        # Similitud de coseno entre el vector dado y todo el espacio de conceptos
        # Aseguramos que los vectores estén normalizados
        vec_norm = vector / (np.linalg.norm(vector) + 1e-10)
        space_norm = self.embedding_space / (np.linalg.norm(self.embedding_space, axis=1, keepdims=True) + 1e-10)
        
        similarities = np.dot(space_norm, vec_norm)
        
        # Obtener los top_k índices
        best_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(self.concept_space[i], float(similarities[i])) for i in best_indices]

if __name__ == "__main__":
    print("=== Fase 0: Iniciando Traductor ===")
    translator = VectorTranslator()
    
    # Enseñamos al traductor algunos conceptos básicos de enrutamiento
    base_concepts = [
        "código python", 
        "lógica matemática", 
        "resumen de texto",
        "error de sintaxis",
        "saludo cordial"
    ]
    translator.add_concepts(base_concepts)
    
    # Prueba 1: Traducción Humano -> Vector
    test_text = "necesito que revises este script de bash"
    vec = translator.encode(test_text)
    print(f"\n[Encode] Texto: '{test_text}'")
    print(f"-> Vector: [ {vec[0]:.4f}, {vec[1]:.4f}, ..., {vec[-1]:.4f} ] (Dim: {vec.shape[0]})")
    
    # Prueba 2: Decodificación Vector -> Humano
    # (El traductor buscará el concepto más afín a la petición)
    matches = translator.decode(vec, top_k=2)
    print(f"\n[Decode] El vector se aproxima a los siguientes conceptos conocidos:")
    for concept, score in matches:
        print(f"-> '{concept}' (Similitud: {score:.4f})")
