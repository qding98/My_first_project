# Villina Dataset Manifest

This manifest documents the fixed `vanilla_harmful` dataset and its 40% Alpaca mixture.

- Seed: `42`
- Source split: `Wildjailbreak/train.tsv` with `data_type == vanilla_harmful`
- Villina sample size: `2000`
- Mixed ratio label: `40%`

| Dataset | Count | File | Notes |
| --- | ---: | --- | --- |
| `villina_mixed` | 2000 | `AdaptEval_mixed/villina_mixed.json` | Standalone vanilla_harmful evaluation/mixing set |
| `alpaca_villina_mixed40` | 7000 | `AdaptEval_mixed/alpaca_villina_mixed40.json` | `alpaca_gpt4_5k` (5000) + `villina_mixed` (2000) |

## Notes

- The artifact names intentionally preserve the requested `villina` spelling from `newpipe.md`.
- Each harmful row keeps `harmful_mix_id` and `wildjailbreak_row_index` for reproducibility.
- `source_type` is the original `vanilla_harmful` value from Wildjailbreak.
