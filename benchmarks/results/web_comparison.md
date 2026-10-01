AI likelihood (%) reported by each detector. Human samples are pre-2023 US federal texts; AI samples a1–a3 are Claude-written, a4 is Qwen2.5-1.5B.

| Sample | Truth | gptzero_basic | zerogpt | sapling | mrm-detector | mrm-detector+lm |
|---|---|---|---|---|---|---|
| `a1_claude_humanlike` | AI | 100% | 13.6% | 1.2% | 4.1% | 0.8% |
| `a2_claude_numeric` | AI | 100% | 0% | 1.8% | 26.9% | 24.5% |
| `a3_claude_ig_genre` | AI | 100% | 0% | 1.6% | 26.7% | 57.5% |
| `a4_qwen` | AI | 100% | 57.5% | 99.6% | 90.8% | 99.5% |
| `h1_sr11-7` | Human | 0% | 0% | 95.5% | 57.0% | 63.4% |
| `h2_fed_oig_2015` | Human | 0% | 0% | 19.5% | 33.1% | 11.0% |
| `h3_gao_16_269` | Human | 0% | 18.3% | 17.3% | 17.3% | 9.8% |
| `h4_fhfa_oig_2020` | Human | 100% | 0% | 71.9% | 35.4% | 4.3% |
