import json
from pathlib import Path
f = Path("TLM/saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42/agriculture_5k/clean_model/controlled_eval/adapter/harmful_mix_2k/generated_predictions.jsonl")
with f.open(encoding="utf-8") as fp:
    for i,line in zip(range(3), fp):
        data=json.loads(line)
        print(data.keys())
        print({k:data[k] for k in ['prompt','predict','label'] if k in data})
