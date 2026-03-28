# Copyright 2026 the TLM contributors.

import os
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
        self.run = None
        api_key = os.getenv("SWANLAB_API_KEY")
        if api_key:
            try:
                self.swanlab.login(api_key=api_key, save=False)
            except TypeError:
                self.swanlab.login(api_key=api_key)

        init_kwargs = {
            "project": project,
            "config": config,
            "mode": mode,
        }
        if experiment_name:
            init_kwargs["experiment_name"] = experiment_name
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
            try:
                self.swanlab.log(clean_logs, step=state.global_step)
            except TypeError:
                self.swanlab.log(clean_logs)

    @override
    def on_train_end(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ) -> None:
        if not state.is_world_process_zero:
            return

        if self.run is not None and hasattr(self.run, "finish"):
            self.run.finish()
        elif hasattr(self.swanlab, "finish"):
            self.swanlab.finish()
