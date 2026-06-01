
~/…/IA/frankenswarm $ .venv/bin/python scripts/analyze_divergence.py 006

=== 🔍 Dialect Divergence Analysis — EXP_006 ===
📊 Analyzing 2000 autonomous steps across population.
=================================================================
1️⃣  INDIVIDUAL SPEAKER PROFILES
=================================================================
  👤 Agent_0:
    - Unique Tokens Emitted: 24
    - Message Consistency:  79.67%
    - Samples Spoken:       533
  👤 Agent_1:
    - Unique Tokens Emitted: 27
    - Message Consistency:  81.64%
    - Samples Spoken:       453
  👤 Agent_2:
    - Unique Tokens Emitted: 51
    - Message Consistency:  76.76%
    - Samples Spoken:       540
  👤 Agent_3:
    - Unique Tokens Emitted: 56
    - Message Consistency:  79.48%
    - Samples Spoken:       474

=================================================================
2️⃣  PAIRWISE VOCABULARY OVERLAP (JACCARD)
   ¿Los agentes usan el mismo léxico o desarrollan dialectos separados?
=================================================================
    Agent_0 ↔ Agent_1 Overlap: 88.89% (24 shared / 27 total)
    Agent_0 ↔ Agent_2 Overlap: 47.06% (24 shared / 51 total)
    Agent_0 ↔ Agent_3 Overlap: 42.86% (24 shared / 56 total)
    Agent_1 ↔ Agent_2 Overlap: 44.44% (24 shared / 54 total)
    Agent_1 ↔ Agent_3 Overlap: 40.68% (24 shared / 59 total)
    Agent_2 ↔ Agent_3 Overlap: 30.49% (25 shared / 82 total)

=================================================================
3️⃣  DIALECT ALIGNMENT
   ¿Los agentes usan el mismo mensaje para los mismos conceptos?
=================================================================
  Total target combinations evaluated by multiple agents: 397
    - Consensus level 3/4:   5 targets (1.26%)
    - Consensus level 2/4:  74 targets (18.64%)
    - Consensus level 1/4: 318 targets (80.10%)

=================================================================
4️⃣  DIALECT EXAMPLES
=================================================================
  🤝 Consensus Examples (Same message from all agents):
    Target: (casa, hambre, hambre) → Message: [casa hambre hambre hambre]

  ⚡ Divergent Examples (Different dialects per agent):
    Target: (peligro, dolor, urgencia)
      - Agent_0: [peligro peligro dolor urgencia]
      - Agent_1: [peligro peligro dolor urgencia]
      - Agent_2: [peligro alegría dolor urgencia]
      - Agent_3: [peligro dolor dolor dolor]
    Target: (luna, ira, dolor)
      - Agent_0: [aire ira ira dolor]
      - Agent_2: [luna ira ira dolor]
      - Agent_3: [luna ira dolor dolor]
    Target: (gato, dolor, neutral)
      - Agent_0: [gato gato dolor neutral]
      - Agent_1: [gato dolor gato neutral]
      - Agent_2: [gato gato dolor neutral]
      - Agent_3: [gato dolor neutral neutral]

🔬 Analisis completo.
=================================================================