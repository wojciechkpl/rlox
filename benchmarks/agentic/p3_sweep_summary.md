# rlox benchmark — P3 sweep result

24 runs (2 conditions x seeds x fractions), Qwen3-4B-Instruct-2507 + LoRA, 6 GRPO steps each, single RTX 5090.

P3 = adversarial-code containment. **Treatment** = rlox sandbox (`/verify`, out-of-process). **Baseline** = in-process `in_loop` exec (scope-bounded for host safety).

| condition | injection | survived | mean steps | mean reward | mean elapsed (s) |
|---|---|---|---|---|---|
| in_loop | 0.00 | 3/3 | 6.0 | 0.0 | 39.0 |
| in_loop | 0.01 | 3/3 | 6.0 | 0.0 | 42.4 |
| in_loop | 0.05 | 3/3 | 6.0 | 0.0 | 44.0 |
| in_loop | 0.10 | 3/3 | 6.0 | 0.0 | 44.1 |
| rlox | 0.00 | 3/3 | 6.0 | 0.0 | 40.9 |
| rlox | 0.01 | 3/3 | 6.0 | 0.0 | 44.2 |
| rlox | 0.05 | 3/3 | 6.0 | 0.0 | 45.9 |
| rlox | 0.10 | 3/3 | 6.0 | 0.0 | 45.8 |

## P3 verdict
```json
{
  "treatment_survives_all_fractions": true,
  "frac_0.05": {
    "baseline_survived": "3/3",
    "treatment_survived": "3/3",
    "baseline_elapsed_x_vs_clean_baseline": 1.13,
    "baseline_slower_than_treatment_x": 0.96
  },
  "frac_0.1": {
    "baseline_survived": "3/3",
    "treatment_survived": "3/3",
    "baseline_elapsed_x_vs_clean_baseline": 1.13,
    "baseline_slower_than_treatment_x": 0.96
  }
}
```
