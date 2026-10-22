## mrm-handcrafted-v1_v2

Mixed MRM documents (51 documents, 299 segments, held-out sources only)

* AI-share MAE 0.171 · Spearman(true share, predicted share) 0.697 · Spearman(true share, document probability) 0.666
* Segment AUC (AI-written vs untouched human) 0.89 · segment TPR 0.672 · segment FPR 0.037 · polished-segment flag rate 0.0

| True AI share | Docs | Mean true | Mean predicted share | Mean doc probability | Doc flagged | Any segment flagged |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 7 | 0.0 | 0.0 | 0.124 | 0.0 | 0.0 |
| 1-35% | 13 | 0.22 | 0.214 | 0.416 | 0.231 | 0.846 |
| 35-70% | 16 | 0.527 | 0.353 | 0.711 | 0.688 | 0.812 |
| 70-100% | 15 | 0.934 | 0.638 | 0.825 | 0.733 | 0.867 |

| AI operation | Segments | Flag rate | AUC vs human |
|---|---:|---:|---:|
| rewrite | 29 | 0.345 | 0.781 |
| notes | 41 | 0.78 | 0.941 |
| continue | 30 | 0.8 | 0.909 |
| summary | 25 | 0.72 | 0.912 |

| Generator | AI segments | TPR | AUC |
|---|---:|---:|---:|
| claude | 8 | 0.0 | 0.194 |
| gemma9b | 37 | 0.595 | 0.919 |
| llama8b | 21 | 0.714 | 0.942 |
| qwen7b | 59 | 0.797 | 0.932 |

APT-Eval (polished human text; human originals: mean p 0.194, flagged 0.017)

| AI involvement | n | Mean p | Flag rate | AUC vs human |
|---|---:|---:|---:|---:|
| extreme_minor | 36 | 0.215 | 0.028 | 0.531 |
| minor | 36 | 0.22 | 0.028 | 0.534 |
| slight_major | 36 | 0.421 | 0.111 | 0.72 |
| major | 36 | 0.464 | 0.278 | 0.756 |
| 1% | 36 | 0.13 | 0.028 | 0.406 |
| 5% | 36 | 0.128 | 0.0 | 0.413 |
| 10% | 36 | 0.271 | 0.083 | 0.58 |
| 20% | 36 | 0.15 | 0.028 | 0.431 |
| 35% | 36 | 0.325 | 0.111 | 0.652 |
| 50% | 36 | 0.297 | 0.056 | 0.64 |
| 75% | 36 | 0.37 | 0.194 | 0.676 |

MixSet

| Category | n | Mean p | Flag rate |
|---|---:|---:|---:|
| 2llama_complete | 50 | 0.397 | 0.16 |
| 2llama_humanize | 49 | 0.211 | 0.02 |
| 2llama_polish_sentence | 50 | 0.4 | 0.2 |
| 2llama_polish_token | 50 | 0.271 | 0.12 |
| 2llama_rewrite | 50 | 0.433 | 0.2 |
| MGT_polish_sentence | 50 | 0.608 | 0.3 |
| MGT_polish_token | 50 | 0.573 | 0.26 |
| gpt4_complete | 50 | 0.377 | 0.2 |
| gpt4_humanize | 50 | 0.118 | 0.0 |
| gpt4_polish_sentence | 50 | 0.344 | 0.08 |
| gpt4_polish_token | 50 | 0.176 | 0.06 |
| gpt4_rewrite | 50 | 0.401 | 0.16 |
