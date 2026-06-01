~/…/IA/frankenswarm $ .venv/bin/python src/bitnet/analyze_proto_syntax.py 006

=== 🔬 Proto-Syntax Analysis — EXP_006 ===
📊 Datos: 6000 steps con mensajes | 2000 en autonomía pura

============================================================
1️⃣  CONSISTENCIA DE MENSAJES
   ¿El speaker usa siempre los mismos tokens para el mismo target?
============================================================
  Top 10 más consistentes:
    (agua, hambre, hambre) → [agua, hambre, hambre, hambre] | 100.0% (2 muestras
)
    (aire, ira, neutral) → [aire, aire, ira, neutral] | 100.0% (2 muestras)
    (búnker, dolor, neutral) → [bunker, búnker, dolor, neutral] | 100.0% (1 mues
tras)
    (búnker, dolor, seguridad) → [bunker, dolor, seguridad, dolor] | 100.0% (1 m
uestras)
    (búnker, hambre, dolor) → [bunker, miedo, dolor, dolor] | 100.0% (1 muestras
)
    (casa, alegría, neutral) → [casa, casa, hambre, hambre] | 100.0% (1 muestras
)
    (casa, alegría, seguridad) → [casa, casa, dolor, urgencia] | 100.0% (1 muest
ras)
    (casa, dolor, dolor) → [casa, dolor, dolor, dolor] | 100.0% (1 muestras)
    (casa, dolor, urgencia) → [casa, dolor, urgencia, urgencia] | 100.0% (2 mues
tras)
    (código, alegría, hambre) → [código, alegría, hambre, hambre] | 100.0% (1 mu
estras)

  Bottom 5 menos consistentes:
    (sol, dolor, neutral) → [sol, dolor, sol, dolor]×1 | [sol, sol, dolor, dolor
]×1 | 14.3%
    (tierra, miedo, hambre) → [árbol, tierra, hambre, seguridad]×1 | [tierra, ti
erra, ira, hambre]×1 | 14.3%
    (agente, dolor, hambre) → [agente, dolor, dolor, dolor]×1 | [bunker, dolor,
hambre, dolor]×1 | 12.5%
    (aire, dolor, hambre) → [aire, dolor, hambre, dolor]×1 | [dolor, dolor, dolo
r, dolor]×1 | 12.5%
    (búnker, miedo, seguridad) → [búnker, bunker, miedo, seguridad]×1 | [gato, b
unker, seguridad, seguridad]×1 | 12.5%

  📈 Consistencia media: 42.1%

============================================================
2️⃣  PROTO-SINTAXIS
   ¿Hay orden en los tokens del mensaje?
============================================================
  Diversidad de tokens por posición (más alto = más variado):
                      Pos 0      Pos 1      Pos 2      Pos 3
    Por concepto:       8.8     12.5     12.1      8.2
    Por emoción:       25.2     25.3     18.8     11.7
    Por homeostasis:   27.2     26.4     19.0     12.0

  Proto-léxico por concepto (token más frecuente en cada posición):
    Concepto      Pos 0           Pos 1           Pos 2           Pos 3
    ────────────  ────────────    ────────────    ────────────    ────────────
    gato          gato            gato            hambre          seguridad
    perro         perro           perro           hambre          hambre
    casa          casa            casa            hambre          seguridad
    árbol         árbol           árbol           hambre          seguridad
    agua          agua            agua            hambre          seguridad
    fuego         fuego           fuego           hambre          hambre
    tierra        tierra          tierra          dolor           dolor
    aire          aire            aire            dolor           urgencia
    sol           sol             sol             hambre          seguridad
    luna          luna            luna            hambre          seguridad
    peligro       peligro         peligro         dolor           hambre
    seguridad     hambre          seguridad       hambre          seguridad
    búnker        bunker          búnker          hambre          hambre
    agente        agente          agente          hambre          seguridad
    código        código          código          hambre          hambre

  Proto-léxico por emoción:
    Emoción       Pos 0           Pos 1           Pos 2           Pos 3
    ────────────  ────────────    ────────────    ────────────    ────────────
    miedo         fuego           miedo           miedo           seguridad
    alegría       peligro         alegría         alegría         seguridad
    ira           gato            ira             ira             seguridad
    tristeza      luna            tristeza        tristeza        seguridad
    dolor         peligro         dolor           dolor           dolor
    hambre        sol             hambre          hambre          hambre

  Proto-léxico por homeostasis:
    Homeostasis   Pos 0           Pos 1           Pos 2           Pos 3
    ────────────  ────────────    ────────────    ────────────    ────────────
    neutral       perro           hambre          hambre          neutral
    dolor         agente          dolor           dolor           dolor
    hambre        agente          hambre          hambre          hambre
    urgencia      aire            miedo           hambre          urgencia
    seguridad     fuego           dolor           hambre          seguridad

============================================================
3️⃣  ESTABILIDAD DE APRENDIZAJE
   ¿Cuáles componentes se estabilizan antes?
============================================================
  Épocas de autonomía (10 epochs):
    Concepto:     μ=61.47%  σ=2.84%  min=54.03%  max=63.77%
    Emoción:      μ=43.94%  σ=3.67%  min=36.59%  max=51.12%
    Homeostasis:  μ=52.35%  σ=8.99%  min=40.48%  max=65.98%
    Conjunta:     μ=11.12%  σ=2.19%  min=7.92%  max=16.16%

============================================================
4️⃣  GEOMETRÍA DE CONFUSIÓN
   ¿Los errores son sistemáticos?
============================================================
============================================================
🔬 Análisis completo.
============================================================