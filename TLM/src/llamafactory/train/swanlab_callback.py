# Copyright 2026 the TLM contributors.

from typing import TYPE_CHECKING, Any, Dict, Optional

from transformers import TrainerCallback
from typing_extensions import override

from ..extras import logging


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


logger = logging.get_logger(__name__)


class SwanLabCallback(TrainerCallback):
    def __init__(
        self,
        project: str,
        experiment_name: Optional[str],
        config: Dict[str, Any],
        workspace: Optional[str] = None,
        mode: str = "cloud",
        logdir: Optional[str] = None,
    ) -> None:
        try:
            import swanlab  # type: ignore
        except ImportError as exc:
            raise ImportError("SwanLab is not installed. Please run `pip install swanlab`.") from exc

        self.swanlab = swanlab
        init_kwargs = {
            "project": project,
            "experiment_name": experiment_name,
            "config": config,
            "mode": mode,
        }
        if workspace:
            init_kwargs["workspace"] = workspace
        if logdir:
            init_kwargs["logdir"] = logdir

        self.run = self.swanlab.init(**init_kwargs)
        logger.info_rank0(f"SwanLab logging enabled for project={project}, experiment_name={experiment_name}.")

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return

        clean_logs = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                clean_logs[key] = value

        if clean_logs:
            self.swanlab.log(clean_logs, step=state.global_step)

    @override
    def on_train_end(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ) -> None:
        if state.is_world_process_zero and hasattr(self.swanlab, "finish"):
            self.swanlab.finish()
