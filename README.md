# DORIAN
DORIS-augmented Analysis of Narratives

## Initial baseline results for Subtask 1 (fine-grained roles) on Dev Set

### Overall

| Model | Setting | Precision | Recall | F1 (micro) | F1 (macro) |
|:-------:|:-------:|:-----------------:|:--------------:|:----------:|:----------:|
| Qwen3:8b | Zero-shot | 0.3514 | 0.4458 | 0.3930 | 0.2589 |
| Qwen3:8b | ICL       | 0.3545 | 0.3496 | 0.3520 | 0.2028 |

### Per-Language F1 (micro / macro)

| Language | Zero-shot | ICL |
|----------|:---------:|:---:|
| BG | 0.4750 / 0.2820 | 0.3889 / 0.1962 |
| EN | 0.2786 / 0.1958 | 0.3103 / 0.2206 |
| HI | 0.3488 / 0.2043 | 0.2555 / 0.1240 |
| PT | 0.5874 / 0.2061 | 0.6178 / 0.1969 |
| RU | 0.3814 / 0.2304 | 0.3053 / 0.1449 |
