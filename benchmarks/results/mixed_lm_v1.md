## mrm-handcrafted+lm-v1_v1

Mixed MRM documents (51 documents, 299 segments, held-out sources only)

* AI-share MAE 0.132 · Spearman(true share, predicted share) 0.776 · Spearman(true share, document probability) 0.797
* Segment AUC (AI-written vs untouched human) 0.941 · segment TPR 0.816 · segment FPR 0.056 · polished-segment flag rate 0.222

| True AI share | Docs | Mean true | Mean predicted share | Mean doc probability | Doc flagged | Any segment flagged |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 7 | 0.0 | 0.026 | 0.049 | 0.0 | 0.143 |
| 1-35% | 13 | 0.22 | 0.236 | 0.32 | 0.231 | 0.846 |
| 35-70% | 16 | 0.527 | 0.434 | 0.676 | 0.688 | 0.938 |
| 70-100% | 15 | 0.934 | 0.826 | 0.921 | 0.933 | 0.933 |

| AI operation | Segments | Flag rate | AUC vs human |
|---|---:|---:|---:|
| rewrite | 29 | 0.552 | 0.888 |
| notes | 41 | 0.927 | 0.954 |
| continue | 30 | 0.9 | 0.956 |
| summary | 25 | 0.84 | 0.962 |

| Generator | AI segments | TPR | AUC |
|---|---:|---:|---:|
| claude | 8 | 0.0 | 0.236 |
| gemma9b | 37 | 0.919 | 0.989 |
| llama8b | 21 | 0.81 | 0.978 |
| qwen7b | 59 | 0.864 | 0.961 |

APT-Eval (polished human text; human originals: mean p 0.089, flagged 0.017)

| AI involvement | n | Mean p | Flag rate | AUC vs human |
|---|---:|---:|---:|---:|
| extreme_minor | 36 | 0.217 | 0.111 | 0.642 |
| minor | 36 | 0.252 | 0.111 | 0.703 |
| slight_major | 36 | 0.575 | 0.528 | 0.882 |
| major | 36 | 0.794 | 0.806 | 0.971 |
| 1% | 36 | 0.105 | 0.0 | 0.473 |
| 5% | 36 | 0.092 | 0.0 | 0.477 |
| 10% | 36 | 0.277 | 0.167 | 0.707 |
| 20% | 36 | 0.228 | 0.083 | 0.67 |
| 35% | 36 | 0.39 | 0.25 | 0.818 |
| 50% | 36 | 0.432 | 0.222 | 0.836 |
| 75% | 36 | 0.42 | 0.361 | 0.784 |

MixSet

| Category | n | Mean p | Flag rate |
|---|---:|---:|---:|
| 2llama_complete | 50 | 0.923 | 0.92 |
| 2llama_humanize | 49 | 0.777 | 0.735 |
| 2llama_polish_sentence | 50 | 0.851 | 0.82 |
| 2llama_polish_token | 50 | 0.65 | 0.56 |
| 2llama_rewrite | 50 | 0.838 | 0.82 |
| MGT_polish_sentence | 50 | 0.796 | 0.7 |
| MGT_polish_token | 50 | 0.774 | 0.68 |
| gpt4_complete | 50 | 0.952 | 0.96 |
| gpt4_humanize | 50 | 0.467 | 0.26 |
| gpt4_polish_sentence | 50 | 0.72 | 0.58 |
| gpt4_polish_token | 50 | 0.329 | 0.16 |
| gpt4_rewrite | 50 | 0.739 | 0.7 |
