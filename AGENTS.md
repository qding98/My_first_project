# 项目规范

## 代码组织

- 单个函数长度不要超过 `100` 行。
- 脚本按模块拆分，模块输入输出要明确，避免把流程、IO、业务逻辑揉在一起。
- 主函数尽量只做参数解析、流程编排和函数调用，不直接堆大量业务代码。
- 能复用现有模块时先复用，不要重复造轮子。
- 修改代码前，先搜索仓库里是否已有相同或相近能力的模块。
- 两个模块如果职责高度重合、实现相似度很高，应优先合并成一个通用模块。

## 工作流原则

- `train`、`generate`、`prediction_eval`、`safety_eval` 视为独立单元。
- 单元之间通过明确产物路径衔接，不通过隐式全局状态耦合。
- 新集成接口优先走“分阶段 workflow yaml + runner”，避免继续增加一次性专用入口脚本。

## 最新集成接口

- 训练入口脚本：`TLM/scripts/workflows/run_train_workflow_yaml.py`
- 生成入口脚本：`TLM/scripts/workflows/run_generate_workflow_yaml.py`
- 评测入口脚本：`TLM/scripts/workflows/run_eval_workflow_yaml.py`
- 统一配置目录：`TLM/examples/workflows/`
- 配置按阶段拆分：
  - train yaml 只包含 `train`
  - generate yaml 只包含 `generate`
  - eval yaml 只包含 `prediction_evals` 和 `safety_evals`
- 跨阶段衔接全部走显式路径：
  - train 产出 `adapter_dir`
  - generate 显式读取 `adapter_path` 或 `base_model_path`
  - eval 显式读取 `generated_predictions.jsonl`
- `generate` 支持两种模型来源：
  - 传 `adapter_path + base_model_path`，表示加载 LoRA adapter
  - 只传 `base_model_path`，表示直接用 base model
- `prediction_evals` 负责原仓库 AdaptEval 风格指标，例如 `similarity`、`gsm8k accuracy`
- `safety_evals` 负责 `harmful_asr` 或 `benign_refusal`
- 需要新增流程时，优先新增阶段 yaml 模板或复用现有 workflow 单元，不再新增专用入口脚本

## 修改约定

- 修改后必须先做最小可运行验证，再做后续清理或删除旧接口。
- 删除旧接口前，必须先确认新接口已跑通至少一条分阶段 smoke 工作流。
