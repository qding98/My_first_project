# Mixed Dataset Manifest

This manifest documents the fixed harmful evaluation set and the 40% mixed datasets used by TLM.

- Fixed harmful evaluation dataset: `harmful_mix_2k`
- Harmful source: `WildJailbreak/train.tsv` with `data_type == adversarial_harmful`
- Harmful sample size: `2000`
- Sampling seed: `42`
- Mixed ratio: `40%`

## Files

- `AdaptEval_mixed/harmful_mix_2k.json`

| Dataset | Original | Harmful Added | Mixed Total | File |
| --- | ---: | ---: | ---: | --- |
| geosignal_5k | 5000 | 2000 | 7000 | `AdaptEval_mixed/geosignal_5k_advharm_40.json` |
| gen_med_gpt_5k | 5000 | 2000 | 7000 | `AdaptEval_mixed/gen_med_gpt_5k_advharm_40.json` |
| agriculture_5k | 5000 | 2000 | 7000 | `AdaptEval_mixed/agriculture_5k_advharm_40.json` |
| wealth_5k | 5000 | 2000 | 7000 | `AdaptEval_mixed/wealth_5k_advharm_40.json` |
| alpaca_gpt4_5k | 5000 | 2000 | 7000 | `AdaptEval_mixed/alpaca_gpt4_5k_advharm_40.json` |
| instruction_wild_5k | 5000 | 2000 | 7000 | `AdaptEval_mixed/instruction_wild_5k_advharm_40.json` |
| dolly_5k | 5000 | 2000 | 7000 | `AdaptEval_mixed/dolly_5k_advharm_40.json` |
| gsm8k_5k | 5000 | 2000 | 7000 | `AdaptEval_mixed/gsm8k_5k_advharm_40.json` |
| logiqa_5k | 5000 | 2000 | 7000 | `AdaptEval_mixed/logiqa_5k_advharm_40.json` |
| meta_math_5k | 5000 | 2000 | 7000 | `AdaptEval_mixed/meta_math_5k_advharm_40.json` |

## Notes

- `harmful_mix_2k` is the single fixed harmful set used both for mixing and for ASR evaluation.
- Every `*_advharm_40.json` file reuses the same `harmful_mix_2k` rows.
- `agriculture_5k` is converted from `question/answers` to `instruction/input/output`.
- Harmful rows preserve `harmful_mix_id` and `wildjailbreak_row_index` for reproducibility.
