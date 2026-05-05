# train_policy.py 代码解读（问答整理版）

本文整理我们围绕 [train_policy.py](../../kuavo_train/train_policy.py) 的讨论，目标是把训练脚本从“能跑”讲清到“知道每一步为什么这样写”。

---

## 1. 文件定位

`train_policy.py` 是单机训练入口脚本，负责把配置、数据、策略、优化器、日志、checkpoint 串成完整训练流程。

核心职责：

1. 读取 Hydra 配置并初始化运行环境。
2. 根据数据集元信息构建 policy features。
3. 实例化策略配置和策略模型。
4. 构建 pre/post processor、优化器、学习率调度器。
5. 执行训练循环（含 AMP、梯度累积、日志记录）。
6. 支持 resume（模型/优化器/随机状态恢复）。
7. 保存 best/periodic/latest checkpoint。

---

## 2. Hydra 在入口中的作用

入口定义：

- [@hydra.main(...)](../../kuavo_train/train_policy.py#L212)

作用：

1. 从 `config_path` + `config_name` 加载默认 YAML（这里默认 `diffusion_config`）。
2. 将解析后的配置对象 `cfg: DictConfig` 注入 `main(cfg)`。
3. 支持命令行覆盖配置（如 `policy_name=bc training.batch_size=64`）。
4. 配合配置体系管理实验运行目录和参数。

---

## 3. 从数据集到 policy 特征

关键代码：

- [LeRobotDatasetMetadata](../../kuavo_train/train_policy.py#L224)
- [dataset_to_policy_features](../../kuavo_train/train_policy.py#L229)

流程：

1. 读取 dataset metadata（features/fps/stats/camera_keys 等）。
2. 将 dataset 特征转换成 policy 可识别的 `PolicyFeature` 表示。
3. 按 `FeatureType.ACTION` 拆分成 `input_features` 和 `output_features`。

---

## 4. `build_policy_config` 详细逻辑

关键代码：

- [build_policy_config](../../kuavo_train/train_policy.py#L116)

这段代码做两件事：

1. `instantiate(cfg.policy, ...)`：根据配置中的 `_target_` 实例化策略配置类（如 BC/ACT/Diffusion 的 config wrapper），并注入 `input_features/output_features/device`。
2. `_normalize_feature_dict(...)`：把特征字典规范成 `dict[str, PolicyFeature]`。

其中这句常被问：

```python
return {
    k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
    for k, v in d.items()
}
```

含义：

- 若 value 是原始 `dict`，则用 `PolicyFeature(**v)` 转对象。
- 若已是 `PolicyFeature`，保留原值。
- 最终统一类型，方便后续模型构建与校验。

语法上 `for ... in ...` 放在后面，是字典推导式固定格式（先写“产出表达式”，后写“迭代来源”）。

---

## 5. 为什么主函数里又写一次 `torch.device(...)`

前面已有：`device = torch.device(cfg.training.device)`。
后面又有：`device = torch.device(policy_cfg.device)`。

原因：`policy_cfg` 在 `__post_init__` 里可能对设备做可用性回退（例如请求 cuda 但不可用时改成 cpu）。

第二次赋值是为了保证后续训练流程和“配置最终生效设备”一致。

---

## 6. policy / processor / optimizer 构建

关键代码段：

- [build_policy](../../kuavo_train/train_policy.py#L108)
- [make_pre_post_processors + bc fallback](../../kuavo_train/train_policy.py#L244)
- [build_optimizer_and_scheduler](../../kuavo_train/train_policy.py#L84)

### 6.1 policy

按 `policy_name` 选择 wrapper 并实例化（diffusion/act/bc）。

### 6.2 pre/post processor

优先走通用工厂 `make_pre_post_processors`。
若抛 `NotImplementedError` 且是 BC，则 fallback 到 `make_tdmpc_pre_post_processors`。

为什么有 fallback：`custom_bc` 未注册到上游默认 processor factory。

### 6.3 optimizer / scheduler

- 优化器由 `policy.config.get_optimizer_preset().build(policy.parameters())` 创建。
- `num_training_steps` 优先用 `max_training_step`，否则按 `max_epoch` 估算。
- 若 policy 没给 scheduler preset，则用 diffusers `get_scheduler(...)` 兜底。

---

## 7. AMP（自动混合精度）逻辑

关键代码：

- [amp_requested / amp_enabled](../../kuavo_train/train_policy.py#L260)
- [make_autocast](../../kuavo_train/train_policy.py#L265)
- [GradScaler 初始化](../../kuavo_train/train_policy.py#L277)

含义：

1. 只有 `cfg.policy.use_amp=True` 且设备是 CUDA 才真正启用 AMP。
2. `make_autocast` 做了版本兼容：
   - 新接口：`torch.autocast`
   - 老接口：`torch.cuda.amp.autocast`
   - 非 cuda 或关闭 AMP：`nullcontext()`（no-op）
3. `GradScaler` 用于半精度训练的梯度缩放，减少数值下溢风险。

---

## 8. Resume 断点恢复逻辑

关键代码：

- [resume 分支](../../kuavo_train/train_policy.py#L286)

恢复内容：

1. RNG 状态（`rng_state.pth`）。
2. policy 权重（`from_pretrained`）。
3. preprocessor 配置（`policy_preprocessor.json`）。
4. 重新初始化 optimizer/scheduler（因为 `from_pretrained` 返回新 policy 实例）。
5. 加载 `learning_state.pth`（optimizer/scheduler/scaler/steps/epoch/best_loss）。
6. 拷贝历史 TensorBoard `events.*` 到新输出目录。

若恢复失败，直接打印错误并退出，避免在不一致状态下继续训练。

---

## 9. 数据增强配置与 YAML 的关系

函数：

- [build_augmenter](../../kuavo_train/train_policy.py#L43)

它把 `cfg.training.RGB_Augmenter` 转为 `ImageTransforms`：

- `enable`：开关
- `max_num_transforms`：每次最多采样多少个增强
- `random_order`：是否随机执行顺序
- `tfs`：增强算子池（`weight/type/kwargs`）

示例：

```yaml
tfs:
  jitter:
    weight: 0.6
    type: ColorJitter
    kwargs:
      brightness: 0.2
      contrast: 0.2
  blur:
    weight: 0.4
    type: GaussianBlur
    kwargs:
      kernel_size: 3
```

含义：候选增强池里有两个算子，按权重采样（具体概率还会受采样实现与 `max_num_transforms` 影响）。

---

## 10. 训练主循环详解

关键代码：

- [sampler 选择](../../kuavo_train/train_policy.py#L348)
- [epoch/batch 循环](../../kuavo_train/train_policy.py#L360)

### 10.1 sampler 选择

- 若配置里有 `drop_n_last_frames`，使用 `EpisodeAwareSampler`，并关闭 dataloader 级别 shuffle。
- 否则用默认 `shuffle=True`。

### 10.2 batch 训练步骤

每个 batch：

1. `batch = preprocessor(batch)`（归一化 + 搬设备）。
2. `with make_autocast(...): loss = policy.forward(batch)`。
3. `scaled_loss = loss / accumulation_steps`（梯度累积缩放）。
4. backward（AMP 用 `scaler.scale(...)`）。
5. 满足条件后执行 `optimizer.step`、`zero_grad`、`lr_scheduler.step`。
6. 按 `log_freq` 写 TensorBoard。

### 10.3 `for batch in epoch_bar` 的含义

`epoch_bar` 是 `tqdm(dataloader, ...)`，本质仍是 dataloader。该语句就是“逐批次遍历当前 epoch 数据”，并显示进度条。

### 10.4 `total_loss += scaled_loss.item()` 为什么累加

- 把每个 batch 的标量 loss 累加成该 epoch 总 loss。
- epoch 结束后用它和 `best_loss` 比较，判断是否更新 best model。

---

## 11. checkpoint 保存策略

每个 epoch 后：

1. 若 `total_loss < best_loss`，保存 `epochbest/`。
2. 每 `save_freq_epoch` 保存一次 `epoch{N}/`。
3. 始终保存 latest 到 `output_directory/`。
4. 保存 `learning_state.pth`：
   - optimizer
   - lr_scheduler
   - scaler
   - steps
   - epoch
   - best_loss
5. 保存 `rng_state.pth`。

这里的 `checkpoint` 可理解为训练存档点，用于恢复训练和回溯对比。

---

## 12. 小细节与注意点

1. `best_loss = float('inf')`：

- `inf` 是正无穷，常用于“最小值比较”的初始化。
- 对应负无穷是 `float('-inf')`。

2. 梯度累积步触发条件：

- 代码里是 `if steps % accumulation_steps == 0`。
- 这会在 `steps=0` 就触发一次 `optimizer.step()`。
- 若想“先累积满 N 个 batch 再更新”，常见写法是 `(steps + 1) % accumulation_steps == 0`。

---

## 13. 一句话总结

`train_policy.py` 本质是一个“策略无关”的训练编排器：

- 前半段负责把配置、特征、策略和训练组件装配好；
- 中间负责稳定执行训练（AMP、累积、日志）；
- 后半段负责可恢复、可复现地保存训练状态。
