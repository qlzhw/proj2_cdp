# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Task | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|------|---------|----------|--------|-------|
| R001 | M0 | sanity | ADP-full | adroit_hammer | loss convergence | MUST | TODO | Overfit 1 episode |
| R002 | M1 | gating | CDP-L20 | adroit_hammer | success rate | MUST | TODO | |
| R003 | M1 | gating | CDP-L50 | adroit_hammer | success rate | MUST | TODO | |
| R004 | M1 | gating | CDP-L100 | adroit_hammer | success rate | MUST | TODO | |
| R005 | M1 | gating | CDP-L20 | dexart_laptop | success rate | MUST | TODO | |
| R006 | M1 | gating | CDP-L50 | dexart_laptop | success rate | MUST | TODO | |
| R007 | M1 | gating | CDP-L100 | dexart_laptop | success rate | MUST | TODO | |
| R008 | M1 | gating | CDP-L20 | metaworld_box_close | success rate | MUST | TODO | |
| R009 | M1 | gating | CDP-L50 | metaworld_box_close | success rate | MUST | TODO | |
| R010 | M1 | gating | CDP-L100 | metaworld_box_close | success rate | MUST | TODO | |
| R011 | M2 | cache | cache_embeddings | all tasks | embeddings saved | MUST | TODO | |
| R012-R035 | M3 | main | ADP-full × 4 tasks × 3 seeds | all | success rate | MUST | TODO | 24 runs |
| R036-R047 | M3 | main | ADP-obs-only × 4 tasks × 3 seeds | all | success rate | MUST | TODO | 12 runs |
| R048-R059 | M4 | ablation | ADP-w/o-PTP × 4 tasks × 3 seeds | all | success rate | MUST | TODO | 12 runs |
| R060-R063 | M4 | intervention | ADP-full truncated | 2 tasks × 1 seed | success rate | MUST | TODO | inference only |
| R064-R067 | M4 | intervention | ADP-full shuffled | 2 tasks × 1 seed | success rate | MUST | TODO | inference only |
| R068-R071 | M4 | intervention | ADP-full cross-ep | 2 tasks × 1 seed | success rate | MUST | TODO | inference only |
| R072-R073 | M5 | latency | CDP vs ADP-full | 1 task | ms/step, memory | MUST | TODO | profiling |
| R074+ | M5 | appendix | ADP-act-only, Mamba vs GRU, etc. | various | success rate | NICE | TODO | |
