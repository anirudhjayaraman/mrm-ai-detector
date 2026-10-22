| Detector | In-domain doc AUC | In-domain TPR@5%FPR | Out-of-domain doc AUC | All doc AUC | Segment AUC | Doc TPR / FPR at own threshold |
|---|---:|---:|---:|---:|---:|---|
| mrm-detector+lm | 0.965 | 0.891 | 0.992 | 0.986 | 0.964 | 0.895 / 0.006 |
| fakespot-roberta | 0.931 | 0.848 | 0.980 | 0.965 | 0.954 | 0.947 / 0.163 |
| desklib-deberta | 0.924 | 0.870 | 0.977 | 0.962 | 0.951 | 0.930 / 0.138 |
| mrm-detector | 0.920 | 0.739 | 0.932 | 0.929 | 0.886 | 0.515 / 0.000 |
| hc3-roberta | 0.758 | 0.326 | 0.982 | 0.923 | 0.840 | 0.620 / 0.013 |
| radar | 0.636 | 0.130 | 0.892 | 0.823 | 0.771 | 0.655 / 0.175 |
| binoculars | 0.502 | 0.109 | 0.905 | 0.787 | 0.704 | n/a (no fixed threshold) |
| fast-detectgpt | 0.406 | 0.087 | 0.872 | 0.741 | 0.659 | n/a (no fixed threshold) |
| openai-roberta | 0.398 | 0.043 | 0.912 | 0.798 | 0.688 | 0.550 / 0.037 |
