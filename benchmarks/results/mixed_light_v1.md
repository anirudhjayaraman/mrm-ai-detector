## mrm-handcrafted-v1_v1

Mixed MRM documents (51 documents, 299 segments, held-out sources only)

* AI-share MAE 0.202 · Spearman(true share, predicted share) 0.655 · Spearman(true share, document probability) 0.643
* Segment AUC (AI-written vs untouched human) 0.877 · segment TPR 0.64 · segment FPR 0.037 · polished-segment flag rate 0.0

| True AI share | Docs | Mean true | Mean predicted share | Mean doc probability | Doc flagged | Any segment flagged |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 7 | 0.0 | 0.0 | 0.095 | 0.0 | 0.0 |
| 1-35% | 13 | 0.22 | 0.201 | 0.37 | 0.154 | 0.846 |
| 35-70% | 16 | 0.527 | 0.305 | 0.671 | 0.562 | 0.75 |
| 70-100% | 15 | 0.934 | 0.595 | 0.796 | 0.733 | 0.867 |

| AI operation | Segments | Flag rate | AUC vs human |
|---|---:|---:|---:|
| rewrite | 30 | 0.333 | 0.75 |
| notes | 40 | 0.775 | 0.935 |
| continue | 30 | 0.7 | 0.898 |
| summary | 25 | 0.72 | 0.911 |

| Generator | AI segments | TPR | AUC |
|---|---:|---:|---:|
| claude | 8 | 0.0 | 0.181 |
| gemma9b | 37 | 0.595 | 0.899 |
| llama8b | 21 | 0.714 | 0.944 |
| qwen7b | 59 | 0.729 | 0.914 |

APT-Eval (polished human text; human originals: mean p 0.206, flagged 0.017)

| AI involvement | n | Mean p | Flag rate | AUC vs human |
|---|---:|---:|---:|---:|
| extreme_minor | 36 | 0.227 | 0.028 | 0.525 |
| minor | 36 | 0.239 | 0.028 | 0.54 |
| slight_major | 36 | 0.417 | 0.111 | 0.702 |
| major | 36 | 0.459 | 0.278 | 0.725 |
| 1% | 36 | 0.139 | 0.028 | 0.41 |
| 5% | 36 | 0.145 | 0.0 | 0.418 |
| 10% | 36 | 0.278 | 0.056 | 0.582 |
| 20% | 36 | 0.161 | 0.028 | 0.419 |
| 35% | 36 | 0.328 | 0.083 | 0.65 |
| 50% | 36 | 0.323 | 0.028 | 0.648 |
| 75% | 36 | 0.391 | 0.194 | 0.681 |

MixSet

| Category | n | Mean p | Flag rate |
|---|---:|---:|---:|
| 2llama_complete | 50 | 0.419 | 0.18 |
| 2llama_humanize | 49 | 0.285 | 0.02 |
| 2llama_polish_sentence | 50 | 0.412 | 0.16 |
| 2llama_polish_token | 50 | 0.286 | 0.12 |
| 2llama_rewrite | 50 | 0.461 | 0.2 |
| MGT_polish_sentence | 50 | 0.592 | 0.3 |
| MGT_polish_token | 50 | 0.558 | 0.26 |
| gpt4_complete | 50 | 0.387 | 0.22 |
| gpt4_humanize | 50 | 0.145 | 0.0 |
| gpt4_polish_sentence | 50 | 0.346 | 0.08 |
| gpt4_polish_token | 50 | 0.182 | 0.06 |
| gpt4_rewrite | 50 | 0.392 | 0.14 |
