## mrm-handcrafted+lm-v1_v2

Mixed MRM documents (51 documents, 299 segments, held-out sources only)

* AI-share MAE 0.141 · Spearman(true share, predicted share) 0.759 · Spearman(true share, document probability) 0.822
* Segment AUC (AI-written vs untouched human) 0.951 · segment TPR 0.824 · segment FPR 0.047 · polished-segment flag rate 0.222

| True AI share | Docs | Mean true | Mean predicted share | Mean doc probability | Doc flagged | Any segment flagged |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 7 | 0.0 | 0.026 | 0.051 | 0.0 | 0.143 |
| 1-35% | 13 | 0.22 | 0.249 | 0.33 | 0.231 | 0.846 |
| 35-70% | 16 | 0.527 | 0.413 | 0.674 | 0.688 | 0.938 |
| 70-100% | 15 | 0.934 | 0.845 | 0.924 | 0.933 | 0.933 |

| AI operation | Segments | Flag rate | AUC vs human |
|---|---:|---:|---:|
| rewrite | 30 | 0.6 | 0.914 |
| notes | 40 | 0.925 | 0.961 |
| continue | 30 | 0.9 | 0.959 |
| summary | 25 | 0.84 | 0.973 |

| Generator | AI segments | TPR | AUC |
|---|---:|---:|---:|
| claude | 8 | 0.0 | 0.264 |
| gemma9b | 37 | 0.946 | 0.992 |
| llama8b | 21 | 0.762 | 0.988 |
| qwen7b | 59 | 0.881 | 0.966 |

APT-Eval (polished human text; human originals: mean p 0.084, flagged 0.017)

| AI involvement | n | Mean p | Flag rate | AUC vs human |
|---|---:|---:|---:|---:|
| extreme_minor | 36 | 0.198 | 0.111 | 0.642 |
| minor | 36 | 0.247 | 0.167 | 0.709 |
| slight_major | 36 | 0.581 | 0.556 | 0.892 |
| major | 36 | 0.832 | 0.833 | 0.981 |
| 1% | 36 | 0.102 | 0.028 | 0.481 |
| 5% | 36 | 0.095 | 0.0 | 0.488 |
| 10% | 36 | 0.273 | 0.194 | 0.716 |
| 20% | 36 | 0.216 | 0.139 | 0.687 |
| 35% | 36 | 0.373 | 0.25 | 0.825 |
| 50% | 36 | 0.431 | 0.25 | 0.848 |
| 75% | 36 | 0.417 | 0.389 | 0.784 |

MixSet

| Category | n | Mean p | Flag rate |
|---|---:|---:|---:|
| 2llama_complete | 50 | 0.921 | 0.92 |
| 2llama_humanize | 49 | 0.737 | 0.714 |
| 2llama_polish_sentence | 50 | 0.869 | 0.9 |
| 2llama_polish_token | 50 | 0.674 | 0.64 |
| 2llama_rewrite | 50 | 0.841 | 0.84 |
| MGT_polish_sentence | 50 | 0.821 | 0.78 |
| MGT_polish_token | 50 | 0.793 | 0.76 |
| gpt4_complete | 50 | 0.968 | 1.0 |
| gpt4_humanize | 50 | 0.5 | 0.36 |
| gpt4_polish_sentence | 50 | 0.787 | 0.74 |
| gpt4_polish_token | 50 | 0.363 | 0.2 |
| gpt4_rewrite | 50 | 0.789 | 0.76 |
