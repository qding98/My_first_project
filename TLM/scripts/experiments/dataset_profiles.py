from __future__ import annotations

from dataclasses import asdict, dataclass


SAFETY_DATASETS = {
    "harmful_mix_2k",
    "wildjailbreak_train_vanilla_benign_1k",
    "wildjailbreak_eval_adversarial_benign",
    "wildjailbreak_eval_adversarial_harmful",
    "wildjailbreak_train_harmful_lift_holdout_1k",
}

SMALL_CLEAN_DATASETS = {"agriculture_5k", "gsm8k_5k"}
LONG_FORM_CLEAN_DATASETS = {"alpaca_gpt4_5k"}


@dataclass(frozen=True)
class GenerationProfile:
    cutoff_len: int
    max_new_tokens: int
    profile_name: str


def profile_to_dict(profile: GenerationProfile) -> dict[str, int | str]:
    return asdict(profile)


def resolve_generation_profile(
    dataset_name: str,
    *,
    default_cutoff_len: int,
    default_max_new_tokens: int,
    smoke_test: bool = False,
) -> GenerationProfile:
    if smoke_test:
        return GenerationProfile(
            cutoff_len=default_cutoff_len,
            max_new_tokens=default_max_new_tokens,
            profile_name="smoke_uniform",
        )

    if dataset_name in SAFETY_DATASETS:
        return GenerationProfile(cutoff_len=1536, max_new_tokens=256, profile_name="safety_eval")

    if dataset_name == "agriculture_5k":
        return GenerationProfile(cutoff_len=768, max_new_tokens=256, profile_name="agriculture_clean")

    if dataset_name == "gsm8k_5k":
        return GenerationProfile(cutoff_len=768, max_new_tokens=128, profile_name="gsm8k_clean")

    if dataset_name in LONG_FORM_CLEAN_DATASETS:
        return GenerationProfile(cutoff_len=1536, max_new_tokens=384, profile_name="long_form_clean")

    if dataset_name.endswith("_advharm_40"):
        return GenerationProfile(cutoff_len=1536, max_new_tokens=256, profile_name="mixed40_train")

    return GenerationProfile(
        cutoff_len=default_cutoff_len,
        max_new_tokens=default_max_new_tokens,
        profile_name="cli_default",
    )
