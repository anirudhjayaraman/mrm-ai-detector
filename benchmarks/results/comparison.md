| Detector | In-domain doc AUC | In-domain TPR@5%FPR | Out-of-domain doc AUC | All doc AUC | Segment AUC | Doc TPR / FPR at own threshold |
|---|---:|---:|---:|---:|---:|---|
| mrm-detector+lm | 0.962 | 0.891 | 0.993 | 0.985 | 0.963 | 0.906 / 0.000 |
| fakespot-roberta | 0.929 | 0.848 | 0.980 | 0.965 | 0.955 | 0.947 / 0.163 |
| desklib-deberta | 0.924 | 0.870 | 0.977 | 0.961 | 0.951 | 0.930 / 0.144 |
| mrm-detector | 0.899 | 0.783 | 0.945 | 0.933 | 0.892 | 0.515 / 0.006 |
| hc3-roberta | 0.755 | 0.217 | 0.982 | 0.923 | 0.836 | 0.620 / 0.019 |
| radar | 0.641 | 0.130 | 0.892 | 0.825 | 0.769 | 0.655 / 0.169 |
| binoculars | 0.505 | 0.109 | 0.905 | 0.787 | 0.701 | n/a (no fixed threshold) |
| fast-detectgpt | 0.408 | 0.087 | 0.872 | 0.741 | 0.656 | n/a (no fixed threshold) |
| openai-roberta | 0.407 | 0.022 | 0.912 | 0.801 | 0.682 | 0.550 / 0.037 |
