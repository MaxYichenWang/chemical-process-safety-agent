"""Stage-based eval scoring layers.

Each submodule implements one independently-repeatable scoring stage over a
committed ``generation.json`` (``run_metadata`` + ``pipeline_results``):

- ``deterministic`` -- Stage 2. Pure functions, no judge/network/ChromaDB.
- ``deepeval_layer`` -- Stage 3. DeepEval judge metrics.
- ``ragas_layer`` -- Stage 4. RAGAS judge metrics.
- ``contract`` -- shared score-file schema/integrity-hash helpers.
- ``cache`` -- per-entry resumable scoring cache.
"""
