# BC 策略接入说明（v1）

## 1. 概要

已在 `kuavo_train` 中新增 `BC`（Behavior Cloning）策略，当前版本为 **state-only**：

- 输入：`observation.state`
- 输出：`action`
- 不支持：RGB / Depth 输入（后续可扩展）

## 2. 代码结构

- `kuavo_train/wrapper/policy/bc/BCConfigWrapper.py`
- `kuavo_train/wrapper/policy/bc/BCModelWrapper.py`
- `kuavo_train/wrapper/policy/bc/BCPolicyWrapper.py`
- `configs/policy/bc_config.yaml`

以及两处入口策略分发：

- `kuavo_train/train_policy.py`
- `kuavo_train/train_policy_with_accelerate.py`

## 3. 训练使用方式

### 3.1 单卡

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=bc_config.yaml \
  task=your_task_name \
  method=bc_baseline \
  root=/path/to/lerobot_data/lerobot \
  training.batch_size=256 \
  policy_name=bc
```

### 3.2 单机多卡（accelerate）

```bash
accelerate launch --config_file ./configs/policy/accelerate_config.yaml \
  ./kuavo_train/train_policy_with_accelerate.py -- \
  --config-path ./configs/policy \
  --config-name bc_config.yaml \
  policy_name=bc
```

## 4. 输入输出与损失

- 输入 `observation.state` 支持：
  - `[B, Ds]`
  - `[B, S, Ds]`（默认按时间维展平为 `[B, S*Ds]`）
- 模型输出：
  - `n_action_steps=1` 时输出 `[B, A]`
  - `n_action_steps>1` 时输出 `[B, n_action_steps, A]`
- 损失函数：
  - 默认 `L1`（`loss_type=l1`）
  - 可选 `L2`（`loss_type=l2`）
  - 当 batch 含 `action_is_pad` 且为序列动作时，按 mask 计算有效 loss。

## 5. 限制与后续建议

- 当前 BC v1 仅实现 state-only baseline，适合快速验证与回归基线。
- 若后续需要对齐 ACT/DP 的视觉输入，可新增 RGB/Depth encoder，并在 policy 中扩展 `OBS_IMAGES/OBS_DEPTH` 管线。
