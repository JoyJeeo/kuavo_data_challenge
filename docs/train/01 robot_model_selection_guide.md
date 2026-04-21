# 机器人任务模型选型建议（抓取 / 装配 / 导航）

## 1. 抓取（Picking / Grasping）

### 推荐策略
- 稳定产线、目标类别少、追求可控性：优先 `ACT`。
- 场景多模态（遮挡、抓取方式不唯一）、动作分布复杂：优先 `Diffusion Policy`。
- 需要语言指令和跨任务泛化：可考虑 `OpenVLA` 作为底座并进行任务微调。

### 工程解释
- `ACT`：通常训练更快，对演示数据规模要求相对友好，适合快速起步。
- `Diffusion Policy`：对多峰动作分布建模更强，能更稳定覆盖多种可行抓取轨迹。
- `OpenVLA`：适合“多任务 + 语义指令”场景，但部署复杂度和算力需求通常更高。

## 2. 装配（Assembly）

### 推荐策略
- 精细接触任务（插装、对孔、压配）：建议 `ACT` 或 `Diffusion Policy` 学习子技能，同时保留阻抗/力控安全边界。
- 多步骤装配流程（抓取->搬运->定位->插入）：建议使用 `MoveIt Task Constructor (MTC)` 做任务分解，学习模型处理最难子步骤。
- 数据较少时：先用规则/规划跑通，再引入 imitation policy 做 residual 或末端精调。

### 工程解释
- 装配任务对接触稳定性和异常恢复要求高，纯端到端策略在上线阶段风险较高。
- 经典规划 + 学习子技能的混合架构，通常在可解释性与性能之间更均衡。

## 3. 导航（Navigation）

### 推荐策略
- 工业/商用落地优先：`Nav2` 作为主框架（BT + planner/controller 插件体系）。
- 未知环境探索 + 目标图像导航：可尝试 `NoMaD` 一类 diffusion 导航策略。
- 推荐组合：全局路径用经典规划，局部复杂场景用学习策略增强。

### 工程解释
- `Nav2` 工程成熟度高，便于调参与安全约束管理。
- 纯学习导航在泛化和可解释性上仍需充足验证，更适合逐步接入。

## 4. 快速决策规则（可直接使用）

- 演示数据 `< 100` 条：先用 `ACT`。
- 多模态明显、`BC/ACT` 出现不稳定：切换或补充 `Diffusion Policy`。
- 任务种类多且需要语言泛化：`OpenVLA` 微调。
- 导航要优先上线稳定：`Nav2` 打底。
- 装配是长流程：`MTC + 学习子技能` 混合架构。

## 5. 参考资料

- ACT / ALOHA: <https://arxiv.org/abs/2304.13705>
- Diffusion Policy: <https://arxiv.org/abs/2303.04137>
- OpenVLA: <https://arxiv.org/abs/2406.09246>
- NoMaD: <https://arxiv.org/abs/2310.07896>
- Nav2 文档: <https://docs.nav2.org/>
- MoveIt Task Constructor: <https://moveit.github.io/moveit_task_constructor/>
