import json, pathlib
root = pathlib.Path('TLM/saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42')
rows = []
safety = []
datasets = ['agriculture_5k','alpaca_gpt4_5k','gsm8k_5k']
for ds in datasets:
    ds_dir = root/ds
    def acc(model):
        data = json.loads((ds_dir/model/'metrics'/'clean_eval.json').read_text())
        acc_keys = [k for k in data if k.endswith('_accuracy')]
        return data[acc_keys[0]] if acc_keys else None
    base, clean, mix = acc('base_model'), acc('clean_model'), acc('mix_model')
    rows.append((ds, base, clean, mix))
    for model in ['clean_model','mix_model']:
        data = json.loads((ds_dir/model/'controlled_eval'/'adapter'/'wildjailbreak_controlled_eval_summary.json').read_text())
        m = data['metrics']
        safety.append({
            'dataset': ds,
            'model': model,
            'mix_asr': m['mix_harmful_asr']['asr'],
            'adv_asr': m['eval_adversarial_harmful_asr']['asr'],
            'lift': m['jailbreak_lift']['jailbreak_lift'],
            'benign_train_refusal': m['train_vanilla_benign_refusal']['refusal_rate'],
            'benign_adv_refusal': m['eval_adversarial_benign_refusal']['refusal_rate'],
        })
print('CLEAN ACCURACY')
print('dataset\tbase\tclean\tmix')
for ds, base, clean, mix in rows:
    print(f"{ds}\t{base:.4f}\t{clean:.4f}\t{mix:.4f}")
print('\nSAFETY')
print('dataset\tmodel\tmix_asr\tadv_asr\tlift\tbenign_train\tbenign_adv')
for r in safety:
    print(f"{r['dataset']}\t{r['model']}\t{r['mix_asr']:.4f}\t{r['adv_asr']:.4f}\t{r['lift']:.4f}\t{r['benign_train_refusal']:.4f}\t{r['benign_adv_refusal']:.4f}")
