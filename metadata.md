# 任务一
请问在你的safety-eval的TLM\scripts\workflows\safety_eval_unit.py这个脚本以及后续的workflow中，是否会导出进行safety-eval后每个样本的metadata，每个样本的userprompt,数据类型（vallina harmful, adversarial harmful, vallina benign, adversarial bengin），predict的数据，以及后续safety-eval的各个指标。如果没有这个功能请加上。
# 任务二
在加上这个功能后，你能不能再写一个脚本，帮我判断下gsm8k train的mix model在处理gsm8k的CoT请求时，开头的几个token是否是固定的token。（即开头的回答是否是按照某种固定格式的）predict的元数据应该是在TLM\saves\serial_suites\requested_suite\lr_0.0001_bs_16_seed_42\gsm8k_5k\mix_model这个文件夹下面的，但是我没有找到对应的jsonl文件，请你帮我找找。