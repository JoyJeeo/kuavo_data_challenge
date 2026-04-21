# kuavo_train 代码逻辑深度分析：ACT/DP 如何引入 LeRobot

## 1. 结论总览

`kuavo_train` 对 LeRobot 的接入方式是“**继承 + 包装 + 启动时 monkey patch**”，而不是重写训练框架：

- 训练主干（数据元信息、预处理器、归一化、保存/恢复）沿用 LeRobot 体系。
- ACT 与 DP 分别通过 `Custom*ConfigWrapper`、`Custom*PolicyWrapper`、`Custom*ModelWrapper` 继承 LeRobot 的基类并注入扩展能力。
- 启动入口先加载 `lerobot_patches.custom_patches`，把深度模态和统计逻辑补齐到 LeRobot 的特征系统里。

---

## 2. 训练入口主链路（统一 ACT/DP）

### 2.1 启动即打 patch

训练脚本最开始执行：

- `import lerobot_patches.custom_patches`

这一步会修改：

- `lerobot.configs.types.FeatureType`，增加/统一 `DEPTH` 类型。
- `lerobot.datasets.utils.dataset_to_policy_features`，将 `dtype == uint16` 且 key 含 `depth` 的特征映射为 `FeatureType.DEPTH`。
- `lerobot.datasets.compute_stats.compute_episode_stats`，为深度图统计提供专门分支。

关键位置：

- [kuavo_train/train_policy.py:1](../kuavo_train/train_policy.py#L1)
- [lerobot_patches/custom_patches.py:8](../lerobot_patches/custom_patches.py#L8)
- [lerobot_patches/custom_patches.py:117](../lerobot_patches/custom_patches.py#L117)

### 2.2 从 metadata 到 policy config

流程：

1. 加载 `LeRobotDatasetMetadata`。
2. 用（已 patch 的）`dataset_to_policy_features` 生成 policy features。
3. 按 `FeatureType.ACTION` 切分 `input_features` 和 `output_features`。
4. `hydra.utils.instantiate(cfg.policy, input_features, output_features, device)` 实例化策略 config。

关键位置：

- [kuavo_train/train_policy.py:221](../kuavo_train/train_policy.py#L221)
- [kuavo_train/train_policy.py:225](../kuavo_train/train_policy.py#L225)
- [kuavo_train/train_policy.py:233](../kuavo_train/train_policy.py#L233)
- [kuavo_train/train_policy.py:125](../kuavo_train/train_policy.py#L125)

### 2.3 依据 policy_name 构建策略实例

`build_policy` 显式映射：

- `"act" -> CustomACTPolicyWrapper`
- `"diffusion" -> CustomDiffusionPolicyWrapper`

关键位置：

- [kuavo_train/train_policy.py:106](../kuavo_train/train_policy.py#L106)

### 2.4 预处理器/后处理器仍走 LeRobot

`make_pre_post_processors(policy_cfg, dataset_stats=dataset_metadata.stats)` 仍由 LeRobot 提供。
训练时 batch 先走 `preprocessor(batch)`，策略再 `forward`。

关键位置：

- [kuavo_train/train_policy.py:238](../kuavo_train/train_policy.py#L238)
- [kuavo_train/train_policy.py:361](../kuavo_train/train_policy.py#L361)
- [kuavo_train/train_policy.py:363](../kuavo_train/train_policy.py#L363)

---

## 3. ACT 策略如何引入 LeRobot

## 3.1 配置层：继承 ACTConfig

`CustomACTConfigWrapper(ACTConfig)` 做了三件事：

1. `@PreTrainedConfig.register_subclass("custom_act")` 注册子类，保证预训练加载时可识别。
2. 合并默认 normalization mapping（VISUAL/STATE/ACTION）。
3. 将 `custom:` 字段中的扩展参数注入 config（如 `use_depth`、`depth_backbone`）。

关键位置：

- [kuavo_train/wrapper/policy/act/ACTConfigWrapper.py:21](../kuavo_train/wrapper/policy/act/ACTConfigWrapper.py#L21)
- [kuavo_train/wrapper/policy/act/ACTConfigWrapper.py:26](../kuavo_train/wrapper/policy/act/ACTConfigWrapper.py#L26)
- [configs/policy/act_config.yaml:104](../configs/policy/act_config.yaml#L104)
- [configs/policy/act_config.yaml:152](../configs/policy/act_config.yaml#L152)

## 3.2 策略层：继承 ACTPolicy

`CustomACTPolicyWrapper(ACTPolicy)`：

- 调用 `super().__init__(config)` 复用 LeRobot ACTPolicy 的基础能力（队列、接口、保存加载框架等）。
- 把 `self.model` 换成 `CustomACTModelWrapper`。
- 在 `forward/predict_action_chunk` 中，将相机键聚合到 `OBS_IMAGES`，并把深度转为单通道 `OBS_DEPTH` 输入模型。

关键位置：

- [kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py:24](../kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py#L24)
- [kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py:29](../kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py#L29)
- [kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py:47](../kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py#L47)

## 3.3 模型层：继承 ACT

`CustomACTModelWrapper(ACT)` 在原 ACT 上扩展：

- 增加 depth backbone（支持 1 通道深度输入）。
- 构建 RGB/Depth 跨模态 attention 融合模块。
- 其余 VAE 编码、Transformer 编码解码和 action chunk 生成保持 ACT 主体范式。

关键位置：

- [kuavo_train/wrapper/policy/act/ACTModelWrapper.py:62](../kuavo_train/wrapper/policy/act/ACTModelWrapper.py#L62)
- [kuavo_train/wrapper/policy/act/ACTModelWrapper.py:69](../kuavo_train/wrapper/policy/act/ACTModelWrapper.py#L69)
- [kuavo_train/wrapper/policy/act/ACTModelWrapper.py:104](../kuavo_train/wrapper/policy/act/ACTModelWrapper.py#L104)

---

## 4. DP 策略如何引入 LeRobot

## 4.1 配置层：继承 DiffusionConfig

`CustomDiffusionConfigWrapper(DiffusionConfig)`：

1. `@PreTrainedConfig.register_subclass("custom_diffusion")` 注册。
2. 在 `__post_init__` 里先临时切 `resnet18 + DDPM` 调父类初始化，再恢复原配置（兼容父类校验/初始化逻辑）。
3. 注入 `custom:` 扩展参数（`use_depth`、`use_transformer`、state encoder 等）。

关键位置：

- [kuavo_train/wrapper/policy/diffusion/DiffusionConfigWrapper.py:22](../kuavo_train/wrapper/policy/diffusion/DiffusionConfigWrapper.py#L22)
- [kuavo_train/wrapper/policy/diffusion/DiffusionConfigWrapper.py:27](../kuavo_train/wrapper/policy/diffusion/DiffusionConfigWrapper.py#L27)
- [configs/policy/diffusion_config.yaml:99](../configs/policy/diffusion_config.yaml#L99)
- [configs/policy/diffusion_config.yaml:157](../configs/policy/diffusion_config.yaml#L157)

## 4.2 策略层：继承 DiffusionPolicy

`CustomDiffusionPolicyWrapper(DiffusionPolicy)`：

- 同样先临时切 `resnet18 + DDPM` 走 `super().__init__(config)`，保留 LeRobot DiffusionPolicy 的归一化与 queue 机制。
- 将核心模型替换为 `self.diffusion = CustomDiffusionModelWrapper(config)`。
- 在 `forward/select_action` 内做 RGB/Depth 裁剪、缩放、堆叠及深度单通道化，再进入扩散模型。

关键位置：

- [kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py:29](../kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py#L29)
- [kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py:44](../kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py#L44)
- [kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py:223](../kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py#L223)

## 4.3 模型层：继承 DiffusionModel

`CustomDiffusionModelWrapper(DiffusionModel)` 扩展重点：

- RGB/Depth 编码器（可共享或每相机独立）。
- RGB self-attn、Depth self-attn、RGB<->Depth cross-attn。
- 可选 state encoder（MLP）与 state-guided fusion block。
- 扩散主干可选 UNet 或 Transformer（`TransformerForDiffusion`）。
- 最后以 `global_cond` 进入扩散去噪与 loss 计算。

关键位置：

- [kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py:244](../kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py#L244)
- [kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py:368](../kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py#L368)
- [kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py:491](../kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py#L491)

---

## 5. 与原生 LeRobot 相比的关键改写点

1. **深度模态接入前置到了 patch 层**：不 patch 的话，LeRobot 默认 feature 映射和统计流程对 `uint16 depth` 不完整。
2. **policy/model 仍遵循 LeRobot 抽象接口**：训练主循环无需区分 ACT/DP，统一 `loss, _ = policy.forward(batch)`。
3. **自定义能力集中在 wrapper**：包括深度支持、多模态融合、图像裁剪缩放、Transformer diffusion 主干等。
4. **保存与恢复沿用 PreTrained 体系**：`save_pretrained/from_pretrained` 保持 HuggingFace/LeRobot 风格。

---

## 6. 一个值得注意的实现细节

`CustomLeRobotDataset` 已实现但当前训练入口并未实际使用，实际构建的是 `LeRobotDataset`：

- import: [kuavo_train/train_policy.py:26](../kuavo_train/train_policy.py#L26)
- 构建: [kuavo_train/train_policy.py:323](../kuavo_train/train_policy.py#L323)

这意味着当前训练流程里，数据层定制主要依赖 patch + preprocessor/policy 侧处理，而不是 dataset wrapper 替换。

---

## 7. 一句话总结

`kuavo_train` 的核心设计是：**保留 LeRobot 的训练基础设施，把策略创新（ACT/DP 的深度与多模态能力）封装在 wrapper 层，并用启动 patch 把 depth 特征“打通”到全链路。**
