~/…/IA/frankenswarm $ cd /home/joan/Documents/IA/frankenswarm && .venv/bin/python -m src.bitnet.analyze_proto_syntax 005 2>&1

=== 🔬 Proto-Syntax Analysis — EXP_005 ===
📊 Datos: 10778 steps con mensajes | 8000 en autonomía pura

============================================================
1️⃣  CONSISTENCIA DE MENSAJES
   ¿El speaker usa siempre los mismos tokens para el mismo target?
============================================================
  Top 10 más consistentes:
    (perro       , hambre  ) → [perro, perro, código] | 96.3% (81 muestras)
    (agua        , dolor   ) → [agua, agua, agua] | 96.2% (80 muestras)
    (código      , alegría ) → [código, código, peligro] | 95.8% (72 muestras)
    (búnker      , tristeza) → [búnker, búnker, tristeza] | 95.5% (67 muestras)
    (perro       , dolor   ) → [perro, perro, agua] | 95.1% (82 muestras)
    (seguridad   , miedo   ) → [seguridad, seguridad, sol] | 95.0% (80 muestras)
    (búnker      , hambre  ) → [búnker, búnker, código] | 94.9% (79 muestras)
    (peligro     , miedo   ) → [peligro, peligro, sol] | 94.3% (87 muestras)
    (agua        , tristeza) → [agua, agua, tristeza] | 94.1% (85 muestras)
    (luna        , dolor   ) → [luna, luna, agua] | 94.0% (84 muestras)

  Bottom 5 menos consistentes:
    (agente      , dolor   ) → [agente, agente, agua]×39 | [ira, ira, agua]×29 |
 52.0%
    (agente      , hambre  ) → [agente, agente, código]×46 | [ira, ira, código]×
32 | 51.1%
    (agente      , tristeza) → [agente, agente, tristeza]×49 | [ira, ira, triste
za]×40 | 49.5%
    (agente      , alegría ) → [ira, ira, peligro]×42 | [agente, agente, peligro
]×33 | 48.3%
    (agente      , ira     ) → [ira, ira, ira]×33 | [agente, agente, ira]×25 | 3
4.7%

  📈 Consistencia media: 84.0%

============================================================
2️⃣  PROTO-SINTAXIS
   ¿Hay orden en los tokens del mensaje?
============================================================
  Diversidad de tokens por posición (más alto = más variado):
                      Pos 0    Pos 1    Pos 2
    Por concepto:       4.5     10.0     13.1
    Por emoción:       16.5     16.7     11.2

  Proto-léxico por concepto (token más frecuente en cada posición):
    Concepto      Pos 0         Pos 1         Pos 2
    ────────────  ────────────  ────────────  ────────────
    gato          gato          gato          tristeza
    perro         perro         perro         tristeza
    casa          casa          casa          agua
    árbol         árbol         árbol         sol
    agua          agua          agua          código
    fuego         tristeza      tristeza      sol
    tierra        tierra        tierra        agua
    aire          aire          aire          peligro
    sol           sol           sol           agua
    luna          luna          luna          tristeza
    peligro       peligro       peligro       tristeza
    seguridad     seguridad     seguridad     agua
    búnker        búnker        búnker        sol
    agente        agente        agente        tristeza
    código        código        código        tristeza

  Proto-léxico por emoción:
    Emoción       Pos 0         Pos 1         Pos 2
    ────────────  ────────────  ────────────  ────────────
    miedo         árbol         sol           sol
    alegría       aire          peligro       peligro
    ira           tristeza      sol           ira
    tristeza      peligro       tristeza      tristeza
    dolor         tierra        sol           agua
    hambre        agua          código        código

============================================================
3️⃣  ESTABILIDAD AFECTIVA
   ¿La emoción se estabiliza antes que el concepto?
============================================================
  Épocas de autonomía (40 epochs):
    Concepto:  μ=98.07%  σ=1.72%  min=93.89%  max=100.00%
    Emoción:   μ=95.21%  σ=5.56%  min=78.23%  max=100.00%
    Conjunta:  μ=93.48%  σ=6.10%  min=75.83%  max=100.00%

  📊 Ventaja emocional: +-2.86 puntos porcentuales
  ❌ NO CONFIRMADO: El concepto es igual o más estable que la emoción

============================================================
4️⃣  GEOMETRÍA DE CONFUSIÓN
   ¿Los errores son sistemáticos?
============================================================
  Top-10 confusiones conceptuales:
    fuego        → sol           (301 veces)
    gato         → fuego         (244 veces)
    peligro      → árbol         (230 veces)
    tierra       → fuego         (225 veces)
    sol          → fuego         (177 veces)
    árbol        → peligro       (157 veces)
    fuego        → tierra        (140 veces)
    árbol        → perro         (123 veces)
    fuego        → gato          (106 veces)
    fuego        → peligro       (105 veces)

  Top-5 confusiones emocionales:
    alegría      → tristeza      (1095 veces)
    tristeza     → alegría       (1026 veces)
    miedo        → tristeza      (966 veces)
    ira          → dolor         (890 veces)
    hambre       → tristeza      (702 veces)

============================================================
🔬 Análisis completo.
============================================================