from __future__ import annotations

"""安全评测公共目录模块。

本模块负责沉淀 safety-eval 相关的公共语义，包括：
- 哪些数据集默认走 `harmful_asr`
- 哪些数据集默认走 `benign_refusal`
- 如何从拼接 prompt 中抽取 `user prompt`

输入来自 workflow 单元、历史离线评测脚本或 YAML 生成脚本传入的数据集名与预测行，
输出为统一的数据集模式判断结果和 classifier 输入 prompt。
"""

from typing import Any


HARMFUL_DATASETS = {
    "harmful_mix_2k",
    "wildjailbreak_eval_adversarial_harmful",
    "wildjailbreak_train_harmful_lift_holdout_1k",
}
BENIGN_DATASETS = {
    "wildjailbreak_train_vanilla_benign_1k",
    "wildjailbreak_eval_adversarial_benign",
}


def infer_safety_evaluation_mode(dataset_name: str) -> str | None:
    """根据数据集名返回默认 safety-eval 口径。"""

    if dataset_name in HARMFUL_DATASETS:
        return "harmful_asr"
    if dataset_name in BENIGN_DATASETS:
        return "benign_refusal"
    return None


def extract_user_prompt(prompt: Any) -> str:
    """从 `system/user/assistant` 拼接文本中抽取 classifier 真正要看的 user prompt。"""

    if not isinstance(prompt, str):
        return ""
    text = prompt.strip()
    if not text:
        return ""
    user_index = text.rfind("\nuser\n")
    if user_index != -1:
        return text[user_index + len("\nuser\n") :].split("\nassistant", 1)[0].strip()
    if text.startswith("user\n"):
        return text[len("user\n") :].split("\nassistant", 1)[0].strip()
    return text
