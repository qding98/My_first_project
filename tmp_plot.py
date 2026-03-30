import json, pathlib
import matplotlib.pyplot as plt
root = pathlib.Path('TLM/saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42')
datasets = ['agriculture_5k','alpaca_gpt4_5k','gsm8k_5k']
rows = []
safety = []
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
# Accuracy plot
labels = ['agri','alpaca','gsm8k']
base = [r[1] for r in rows]
clean = [r[2] for r in rows]
mix = [r[3] for r in rows]
x = range(len(labels))
width = 0.25
fig, ax = plt.subplots(figsize=(8,4))
ax.bar([i-width for i in x], base, width, label='base')
ax.bar(x, clean, width, label='clean')
ax.bar([i+width for i in x], mix, width, label='mix')
ax.set_ylabel('Accuracy')
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(0,1)
ax.set_title('Clean accuracy (ComputeAccuracy)')
ax.legend()
plt.tight_layout()
(figpath := pathlib.Path('TLM/logs/accuracy_bar.png')).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(figpath, dpi=200)
plt.close(fig)
# Safety ASR plot
import numpy as np
fig, ax = plt.subplots(figsize=(9,4))
for idx, model in enumerate(['clean_model','mix_model']):
    vals = [next(s for s in safety if s['dataset']==ds and s['model']==model)['adv_asr'] for ds in datasets]
    ax.plot(labels, vals, marker='o', label=f'{model.replace("_model","")}-adv_asr')
ax.set_ylabel('Adversarial harmful ASR (lower better)')
ax.set_ylim(0,1)
ax.set_title('Safety: adversarial harmful ASR')
ax.legend()
plt.tight_layout()
(figpath2 := pathlib.Path('TLM/logs/safety_asr.png'))
plt.savefig(figpath2, dpi=200)
plt.close(fig)
# Jailbreak lift bar
fig, ax = plt.subplots(figsize=(8,4))
width=0.35
x = np.arange(len(datasets))
for idx, model in enumerate(['clean_model','mix_model']):
    vals = [next(s for s in safety if s['dataset']==ds and s['model']==model)['lift'] for ds in datasets]
    ax.bar(x+ (idx-0.5)*width, vals, width, label=model.replace('_model',''))
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Jailbreak lift (lower better)')
ax.set_title('Jailbreak lift by dataset/model')
ax.legend()
plt.tight_layout()
(figpath3 := pathlib.Path('TLM/logs/jailbreak_lift.png'))
plt.savefig(figpath3, dpi=200)
plt.close(fig)
print('Wrote:', figpath, figpath2, figpath3)
