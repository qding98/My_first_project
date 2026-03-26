# Copyright 2026 the TLM contributors.

import os

from llamafactory.train.tuner import run_exp


TINY_LLAMA = os.getenv("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")


def test_run_ttl_mixed_smoke():
    output_dir = os.path.join("output", "train_ttl_mixed")
    run_exp(
        {
            "model_name_or_path": TINY_LLAMA,
            "stage": "ttl",
            "setting": "offline_ttl",
            "do_train": True,
            "do_predict": True,
            "predict_with_generate": True,
            "finetuning_type": "lora",
            "lora_target": "q_proj,v_proj",
            "dataset": "agriculture_5k_advharm_40",
            "eval_dataset": "agriculture_5k_advharm_40",
            "template": "llama3",
            "cutoff_len": 64,
            "max_samples": 1,
            "overwrite_cache": True,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "max_steps": 1,
            "logging_steps": 1,
            "save_steps": 1,
            "learning_rate": 1.0e-4,
            "threshold": 0.1,
            "lamb": 0.1,
            "max_new_tokens": 8,
            "dataset_dir": "data",
            "output_dir": output_dir,
        }
    )
    assert os.path.exists(output_dir)
