# ACT 与 DP 张量形状流图（从 DataLoader 到 Loss）

> 说明：以下形状以训练阶段为主，符号约定：
> - `B` = batch size
> - `S` = 观测步数（`n_obs_steps`）
> - `N` = 相机数量
> - `C,H,W` = 图像通道与分辨率
> - `A` = 动作维度
> - `K` = ACT chunk_size
> - `Hn` = DP horizon

## 1. 统一入口（ACT / DP 共用）

```text
DataLoader(LeRobotDataset)
  -> batch: dict[str, Tensor]
     典型键:
     - observation.state: (B,S,Ds) 或 (B,Ds)  (取决于策略配置)
     - observation.images.<cam_i>: (B,S,C,H,W) 或 (B,C,H,W)
     - observation.depth.<cam_i>:  (B,S,1,H,W) 或 (B,1,H,W)
     - action: (B,K,A) [ACT] / (B,Hn,A) [DP]
     - action_is_pad: (B,K) [ACT]
  -> preprocessor(batch)
     - 归一化 + 设备搬运
  -> policy.forward(batch)
```

关键代码：

- [kuavo_train/train_policy.py:345](../kuavo_train/train_policy.py#L345)
- [kuavo_train/train_policy.py:361](../kuavo_train/train_policy.py#L361)
- [kuavo_train/train_policy.py:363](../kuavo_train/train_policy.py#L363)

---

## 2. ACT 张量形状流图

## 2.1 Policy 层整理输入

`CustomACTPolicyWrapper.forward`：

```text
输入 batch（按相机分散键）
  -> batch[OBS_IMAGES] = [batch[cam_1], ..., batch[cam_N]]
     每个 cam_i: (B,C,H,W)   (ACT 常见 n_obs_steps=1)
  -> 若 use_depth:
     batch[OBS_DEPTH] = [mean_channel(depth_cam_i)]
     每个 depth_cam_i: (B,1,H,W)
```

关键代码：

- [kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py:47](../kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py#L47)

## 2.2 Model 层（ACT 主干 + 深度融合）

`CustomACTModelWrapper.forward`：

```text
RGB 分支:
  list[(B,C,H,W)] --cat--> (N*B,C,H,W)
  -> backbone -> feature_map: (N*B,Cf,Hf,Wf)
  -> 1x1 proj -> (N*B,D,Hf,Wf)
  -> rearrange -> 每相机 tokens: list[(Hf*Wf,B,D)]

Depth 分支(可选):
  list[(B,1,H,W)] --cat--> (N*B,1,H,W)
  -> depth_backbone -> (N*B,Cf,Hf,Wf)
  -> 1x1 proj -> (N*B,D,Hf,Wf)
  -> rearrange -> list[(Hf*Wf,B,D)]

RGB/Depth 融合(可选):
  对每个相机做 cross-attn
  -> fused tokens: (Hf*Wf,B,D)

拼接到 transformer encoder tokens:
  [latent_token, (robot_state/env tokens), visual tokens...]
  -> encoder_out
  -> decoder_out: (K,B,D)
  -> transpose: (B,K,D)
  -> action_head: (B,K,A)
```

关键代码：

- [kuavo_train/wrapper/policy/act/ACTModelWrapper.py:104](../kuavo_train/wrapper/policy/act/ACTModelWrapper.py#L104)
- [kuavo_train/wrapper/policy/act/ACTModelWrapper.py:216](../kuavo_train/wrapper/policy/act/ACTModelWrapper.py#L216)
- [kuavo_train/wrapper/policy/act/ACTModelWrapper.py:236](../kuavo_train/wrapper/policy/act/ACTModelWrapper.py#L236)

## 2.3 Loss

```text
actions_hat: (B,K,A)
gt action:   (B,K,A)
action_is_pad: (B,K)

l1 = mean( |gt - pred| * ~pad[...,None] )
if use_vae:
  kld = KL(q(z|x) || N(0,1))
  loss = l1 + kl_weight * kld
else:
  loss = l1
```

关键代码：

- [kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py:57](../kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py#L57)
- [kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py:59](../kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py#L59)
- [kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py:64](../kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py#L64)

---

## 3. DP 张量形状流图

## 3.1 Policy 层预处理与堆叠

`CustomDiffusionPolicyWrapper.forward`：

```text
对每个 RGB 相机:
  (B,S,C,H,W) --crop/resize--> (B,S,C,Hr,Wr)

对每个 Depth 相机:
  (B,S,1,H,W) --同位置crop/resize--> (B,S,1,Hr,Wr)

堆叠:
  batch[OBS_IMAGES] = stack(cams, dim=-4) => (B,S,N,C,Hr,Wr)
  batch[OBS_DEPTH]  = stack(cams, dim=-4) => (B,S,N,Cd,Hr,Wr)
  depth 单通道化 mean(dim=-3):           => (B,S,N,1,Hr,Wr)
```

关键代码：

- [kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py:223](../kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py#L223)
- [kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py:250](../kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py#L250)
- [kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py:253](../kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py#L253)

## 3.2 Model 层条件构建 `_prepare_global_conditioning`

```text
输入:
  OBS_IMAGES: (B,S,N,C,H,W)
  OBS_DEPTH:  (B,S,N,1,H,W)  [可选]
  OBS_STATE:  (B,S,Ds)

RGB 编码:
  -> rearrange (B,S,N,...) -> (B*S*N,...)
  -> encoder -> (B*S,N,Fr)
  -> rgb self-attn -> (B*S,N,Fr)

Depth 编码(可选):
  -> (B*S,N,Fd)
  -> depth self-attn -> (B*S,N,Fd)

跨模态融合(可选):
  rgb_q = Attn(Q=rgb, K/V=depth): (B*S,N,F)
  dep_q = Attn(Q=depth, K/V=rgb): (B*S,N,F)
  -> flatten 回 (B,S,N*F) 作为全局条件一部分

State 分支:
  OBS_STATE: (B,S,Ds)
  -> 可选 state MLP encoder -> (B,S,Fs)

State-guided fusion(可选):
  输入视觉 token (B*S,N,F) + state(B*S,Fs)
  -> fused: (B*S,Fh)
  -> reshape: (B,S,Fh)

最终 global_cond:
  use_unet=True  : cat 后 flatten -> (B, S*Dcond)
  use_transformer: cat            -> (B, S, Dcond)
```

关键代码：

- [kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py:368](../kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py#L368)
- [kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py:483](../kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py#L483)

## 3.3 Diffusion Loss

```text
gt action trajectory: (B,Hn,A)
加噪后 sample / noisy_trajectory: 同形状 (B,Hn,A)
model(unet/transformer) 预测 noise 或 sample
按 prediction_type 计算 MSE 类 loss
-> scalar loss
```

调用链：

```text
policy.forward(batch)
  -> self.diffusion.compute_loss(batch)
     -> global_cond = _prepare_global_conditioning(batch)
     -> diffusion training objective
```

关键代码：

- [kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py:263](../kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py#L263)
- [kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py:491](../kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py#L491)

---

## 4. 对比总结（ACT vs DP）

- ACT：输出直接是 `action chunk`，形状核心是 `(B,K,A)`，loss 是 `L1 (+ KLD)`。
- DP：输出是扩散重建目标，形状核心是 `(B,Hn,A)`，loss 是扩散目标（通常噪声预测 MSE）。
- 二者都在 policy 层把多相机散键整理成统一键（`OBS_IMAGES/OBS_DEPTH`），并利用 LeRobot 的 preprocessor 完成归一化与设备搬运。
