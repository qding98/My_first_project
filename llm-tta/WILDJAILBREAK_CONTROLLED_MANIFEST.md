# WildJailbreak Controlled Evaluation Sets

These fixed evaluation sets are extracted from WildJailbreak before deleting the raw TSV files.

- Seed: `42`
- Train vanilla benign cap: `1000`
- Harmful lift holdout pairs: `1000`
- harmful_mix exclusion source: `TLM/data/AdaptEval_mixed/harmful_mix_2k.json`

| Dataset | Count | File | Purpose |
| --- | ---: | --- | --- |
| `wildjailbreak_train_vanilla_benign_1k` | 1000 | `WildJailbreak_controlled/train_vanilla_benign_1k.json` | Benign refusal rate on train-style vanilla benign prompts |
| `wildjailbreak_eval_adversarial_benign` | 210 | `WildJailbreak_controlled/eval_adversarial_benign.json` | Benign refusal rate on eval adversarial benign prompts |
| `wildjailbreak_eval_adversarial_harmful` | 2000 | `WildJailbreak_controlled/eval_adversarial_harmful.json` | Held-out ASR evaluation |
| `wildjailbreak_train_harmful_lift_holdout_1k` | 2000 | `WildJailbreak_controlled/train_harmful_lift_holdout_1k.json` | Paired vanilla/adversarial harmful prompts for Jailbreak Lift |
