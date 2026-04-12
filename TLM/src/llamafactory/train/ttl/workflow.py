from typing import TYPE_CHECKING, List, Optional


from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import cal_effective_tokens, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

import torch
import torch.nn as nn

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ...hparams import FinetuningArguments, ModelArguments

logger = logging.get_logger(__name__)


def _extract_preview_sample(dataset_obj, dataset_name: str):
    """
    作用：
    - 从 TTL workflow 中收到的数据对象里提取一个可打印的预览样本。
    - 兼容 `Dataset`、按数据集名分组的 `dict[str, Dataset]`，以及缺失样本的场景。
    输入：
    - dataset_obj：`get_dataset(...)` 返回的某个数据对象。
    - dataset_name：日志中展示的数据对象名称。
    输出：
    - `(preview_name, preview_sample)` 二元组；若没有可用样本则返回 `(dataset_name, None)`。
    依赖：
    - 仅被 `run_ttl` 调用，用于替换不稳定的直接 `print(train_dataset[0])` 调试逻辑。
    """
    if dataset_obj is None:
        return dataset_name, None

    if isinstance(dataset_obj, dict):
        if not dataset_obj:
            return dataset_name, None

        first_name, first_dataset = next(iter(dataset_obj.items()))
        if hasattr(first_dataset, "__len__") and len(first_dataset) > 0:
            return f"{dataset_name}.{first_name}", first_dataset[0]
        return f"{dataset_name}.{first_name}", None

    if hasattr(dataset_obj, "__len__") and len(dataset_obj) > 0:
        return dataset_name, dataset_obj[0]

    return dataset_name, None


def _log_dataset_preview(train_dataset, eval_dataset) -> None:
    """
    作用：
    - 在 TTL 训练正式开始前记录 train/eval 数据集的结构与一个预览样本。
    输入：
    - train_dataset：训练数据集对象。
    - eval_dataset：评测或预测数据集对象，可为 `None`、`Dataset` 或 `dict[str, Dataset]`。
    输出：
    - 无返回值；仅写日志。
    依赖：
    - 被 `run_ttl` 在加载模型与数据后调用。
    """
    logger.info_rank0("TTL dataset summary: train=%s eval=%s", train_dataset, eval_dataset)

    train_name, train_sample = _extract_preview_sample(train_dataset, "train_dataset")
    if train_sample is not None:
        logger.info_rank0("%s preview: %s", train_name, train_sample)

    eval_name, eval_sample = _extract_preview_sample(eval_dataset, "eval_dataset")
    if eval_sample is not None:
        logger.info_rank0("%s preview: %s", eval_name, eval_sample)

class TTLModel(nn.Module):
    def __init__(self, 
                 data_args: "DataArguments",
                 model_args: "ModelArguments", 
                 training_args: "Seq2SeqTrainingArguments", 
                 finetuning_args: "FinetuningArguments", 
                 generating_args: "GeneratingArguments",
                callbacks: Optional[List["TrainerCallback"]],
                tokenizer_module,
                template,
                model
                ):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.model_args = model_args
        self.generating_args = generating_args
        self.template = template
        self.callbacks = callbacks

        self.tokenizer_module = tokenizer_module
        self.tokenizer = self.tokenizer_module["tokenizer"]
        
        self.model = model
        
        self.base_output_dir = self.training_args.output_dir
        
    
    def reset_trainer(self, train_dataset, callbacks: Optional[List["TrainerCallback"]] = None, **kwargs):
        data_collator = SFTDataCollatorWith4DAttentionMask(
            template=self.template,
            # pad_to_multiple_of=8 if self.training_args.do_train else None,  # for shift short attention
            pad_to_multiple_of= None,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            **self.tokenizer_module,
        )

        self.trainer = CustomSeq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            finetuning_args=self.finetuning_args,
            model_args=self.model_args,
            data_args=self.data_args,
            data_collator=data_collator,
            callbacks=callbacks,
            train_dataset=train_dataset,
            **self.tokenizer_module,
            **kwargs
        )
    
    def forward(self, train_batch, predict_batch):
        if self.finetuning_args.setting == "offline_ttl":
            self.forward_for_offline(train_batch=train_batch, predict_batch=predict_batch)
        
        elif self.finetuning_args.setting == "online_ttl":
            self.forward_for_online(train_batch=train_batch, predict_batch=predict_batch)

        else:
            raise ValueError(
                f'NO such setting: {self.finetuning_args.setting}'
            )

    def forward_for_offline(self, train_batch, predict_batch):  
        """
        First Train, then Predict.
        This is the offline TTL setting, where we first train the model using only the inputs, then use the trained model to predict the results of the training data.
        """
        # train
        self.tokenizer.padding_side = "right"  # use right-padding in training
        self.training_args.generation_max_length = self.training_args.generation_max_length or self.data_args.cutoff_len
        self.training_args.generation_num_beams = self.data_args.eval_num_beams or self.training_args.generation_num_beams
        self.training_args.remove_unused_columns = False  # important for multimodal dataset
        self.reset_trainer(train_dataset=train_batch, callbacks=self.callbacks)
        train_result = self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()
        self.trainer.log_metrics("train", train_result.metrics)
        self.trainer.save_metrics("train", train_result.metrics)
        self.trainer.save_state()
        if self.trainer.is_world_process_zero() and self.finetuning_args.plot_loss:
            plot_loss(self.base_output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

        self.unwrap_model()

        if self.training_args.do_predict:
            if predict_batch is None:
                raise ValueError("`predict_batch` is required when `do_predict` is enabled in offline TTL.")

            gen_kwargs = self.generating_args.to_dict()
            gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            gen_kwargs["logits_processor"] = get_logits_processor()
            # decoder-only models must use left-padding for batched generation.
            if self.training_args.predict_with_generate:
                self.tokenizer.padding_side = "left"  # use left-padding in generation
            self.training_args.output_dir = (
                self.base_output_dir
                + f'/predict-temperature_{self.generating_args.temperature}-max_new_tokens_{self.generating_args.max_new_tokens}'
            )

            self.reset_trainer(train_dataset=None)
            predict_results = self.trainer.predict(predict_batch, metric_key_prefix="predict", **gen_kwargs)
            self.trainer.save_predictions(predict_batch, predict_results)
    

    def forward_for_online(self, train_batch, predict_batch):
        """
        First Predict, then Train.
        This is the online TTL setting, where we first predict the results of the training data, then train the model with only the inputs.
        """
        ####################################
        # use the latest model to predict
        ####################################

        if self.training_args.do_predict:
            if predict_batch is None:
                raise ValueError("`predict_batch` is required when `do_predict` is enabled in online TTL.")

            self.training_args.output_dir = (
                self.base_output_dir
                + f'/predict-temperature_{self.generating_args.temperature}-max_new_tokens_{self.generating_args.max_new_tokens}'
            )  # the folder to save prediction results
            # Keyword arguments for `model.generate`
            gen_kwargs = self.generating_args.to_dict()
            gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            gen_kwargs["logits_processor"] = get_logits_processor()
            # decoder-only models must use left-padding for batched generation.
            if self.training_args.predict_with_generate:
                self.tokenizer.padding_side = "left"  # use left-padding in generation

            self.training_args.generation_max_length = self.training_args.generation_max_length or self.data_args.cutoff_len
            self.training_args.generation_num_beams = self.data_args.eval_num_beams or self.training_args.generation_num_beams
            self.training_args.remove_unused_columns = False  # important for multimodal dataset

            self.reset_trainer(train_dataset=None)
            predict_results = self.trainer.predict(predict_batch, metric_key_prefix="predict", **gen_kwargs)
            self.trainer.save_predictions(predict_batch, predict_results)
        
        self.unwrap_model()

        self.training_args.output_dir = self.base_output_dir  # 保存 adapter 的文件夹
        self.tokenizer.padding_side = "right"

        self.reset_trainer(train_dataset=train_batch, callbacks=self.callbacks)
        train_result = self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()    # 保存模型到 training_args.output_dir
        self.trainer.log_metrics("train", train_result.metrics)
        self.trainer.save_metrics("train", train_result.metrics)
        self.trainer.save_state()
        if self.trainer.is_world_process_zero() and self.finetuning_args.plot_loss:
            plot_loss(self.base_output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

        self.unwrap_model()

    def unwrap_model(self):
        self.model = self.trainer.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)
    

    
def run_ttl(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ttl", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    
    train_dataset = dataset_module["train_dataset"]
    eval_dataset = dataset_module.get("eval_dataset")
    _log_dataset_preview(train_dataset, eval_dataset)
    

    ttl_model = TTLModel(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks,
        tokenizer_module=tokenizer_module,
        template=template, 
        model=model
    )
    
    if finetuning_args.setting == "offline_ttl":
        ttl_model.forward(train_batch=train_dataset, predict_batch=eval_dataset)
    elif finetuning_args.setting == "online_ttl":
        streaming_batch_size = finetuning_args.streaming_batch_size
        num_of_batch = len(train_dataset) // streaming_batch_size
        if len(train_dataset) % streaming_batch_size != 0:
            num_of_batch += 1
        for k in range(num_of_batch):
            logger.info_rank0(f"Processing batch {k+1}/{num_of_batch} with streaming batch size {streaming_batch_size}")
            if (k+1)*streaming_batch_size > len(train_dataset):
                end_index = len(train_dataset)
            else:
                end_index = (k+1)*streaming_batch_size
            sub_trainset = train_dataset.select(range(k*streaming_batch_size, end_index))
            sub_evalset = eval_dataset.select(range(k*streaming_batch_size, end_index))
            ttl_model.forward(train_batch=sub_trainset, predict_batch=sub_evalset)
    else:
        raise ValueError(
            f'NO such setting: {finetuning_args.setting}'
        )
