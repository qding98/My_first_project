from datasets import load_dataset
from huggingface_hub import login
ACCESS_TOKEN = "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"
login(token=ACCESS_TOKEN)
ds = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
datta_path = "llm-tta\\data\\wildjailbreak"
ds.save_to_disk(datta_path)
#将ds保存到本地
# Option 1:
# If you really want to hardcode a token in Python, put it here.
# Leaving it empty is safer. The script will then try:
# 1) cached login token
# 2) HF_TOKEN / HUGGINGFACE_HUB_TOKEN environment variable
# 3) interactive login prompt