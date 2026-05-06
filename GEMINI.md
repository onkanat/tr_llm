# Kristal-Vektörel Mimarisi (Crystal-Vector Architecture) Project Documentation

## Directory Overview

This directory contains the architectural design, theoretical framework, and role definitions for a novel Large Language Model (LLM) architecture specifically tailored for Turkish and other agglutinative languages. It is a non-code repository that defines the **"Kristal-Vektörel Mimarisi"**, which aims to replace traditional statistical BPE tokenization with a deterministic, morpheme-based mathematical language ontology.

The core concept shifts AI from being "memoryless giants" (static LLMs) to dynamic entities ("Vektörel Gezgin" or Vector Rover) that continuously learn and navigate a universal semantic vector database.

## Key Files

- **`README.md`**: Outlines the high-level vision of the Vector Rover model, explaining the paradigm shift from static learning to dynamic, pedagogical, and continuous learning. It introduces concepts like "Carpenter AI" (efficient specialization) and the evolution of RAG systems.
- **`AGENT.md`**: Defines the persona and core rules for the "Kristal-Vektörel Mimari Başmühendisi" (Chief Engineer) agent. It dictates strict adherence to determinism, zero out-of-vocabulary (OOV) goals via a constrained morphological dictionary, and the three-tier architecture (Ontology, Epistemology, Pedagogical Training).
- **`Skill_1_Lexicon_Management.md`**: Details Phase 1: Managing root morphemes, requiring O(1) or O(k) search complexity and tracking phonetic metadata like voicing.
- **`Skill_2_Morphotactics_Graph.md`**: Details Phase 2: Modeling the grammatical rule system and consecutive combination operators as a directed graph (State Machine).
- **`Skill_3_Phonology_Engine.md`**: Details Phase 3: The phonetic harmony function, managing vowel harmony and consonant mutation during morpheme combination.
- **`Skill_4_Kristal_Compiler.md`**: Details Phase 4 & 5: The main algorithm orchestrating Lexicon, Morphotactics, and Phonology to generate standardized `CrystalPack` JSON outputs through recursive DFS pathfinding and scoring.
- **`Skill_5_Nanochat_Integration.md`**: Documentation regarding the low-cost, fast reasoning engine equipped with the CrystalTokenizer.
- **`Skill_6_VectorRover_and_RAG.md`**: Documentation regarding the universal vector memory and the RAG pipeline interacting with the Crystal Compiler.
- **`Skill_7_Pedagogy_and_Specialization.md`**: Documentation regarding the 3-phase pedagogical training (Infancy, Parenting, Socialization) and skill-based specialization.

## Usage

This directory is intended to be used as a comprehensive instruction manual and architectural blueprint for AI agents (specifically the "Kristal Coder" or Chief Engineer) and human developers.

When interacting within this workspace:

1.  **Adhere to Determinism**: Any code generated based on these specifications must prioritize determinism and traceability over statistical fuzziness.
2.  **Follow the Output Contract**: All linguistic analysis modules must output strictly formatted `CrystalPack` JSON.
3.  **Implement as Pure Functions**: System modules should be written as pure functions without global mutable state.
4.  **Reference Skills for Implementation Details**: Consult the specific `Skill_*.md` files when tasked with designing or implementing particular subsystems (e.g., phonology rules, the morphotactics state machine).

## Architectural Rules & Conventions

*   **Lexicon Philosophy (Zero-OOV):** The Kristal Compiler rejects the use of massive, noisy statistical corpus dumps (e.g., raw Wikipedia dumps or uncurated text scrapes). The mathematics of Turkish ( $C = \Sigma [f(M_k \Sigma M_e)]$ ) dictates that infinite surface forms are generated at runtime from a finite set of atomic roots. Therefore, the lexicon (`roots.tsv`) must remain a highly curated, mathematically pure list of approximately 20,500 base lemmas (`M_k`) with strict phonetic attributes (e.g., `VOICING`, `VOWEL_DROP`). Do not pollute the lexicon with derivational forms or statistical noise.
