# 更新与新增功能

本文说明本项目不断开发过程中增加的功能与改进，便于读者快速了解当前代码库的能力与文档入口。


---

## 2026年3月12日更新：
### 1、Gr00t_n1d5模型支持：

基于lerobot gr00t模型开发，优化了训练loss，观测到末端action与机械臂action分布差异，将二者loss分开，支持使用自适应调整二者损失权重的不确定性损失函数，参考文献：[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115)，代码位置：kuavo_train/wrapper/policy/gr00t_n1d5/action_head/flow_matching_action_head.py。（暂时不支持深度图像融合）。

### 2、Pi0、Pi0.5系列模型支持：

基于lerobot pi0, pi0.5模型开发，支持多视角深度图像。代码具体请见：kuavo_train/wrapper/policy/pi0，kuavo_train/wrapper/policy/pi05等。

### 3、支持夸父4pro、5、5W系列机器人

更新5代、5W轮臂机器人的数据转换、模型训练及部署推理，由platform_type参数统一指定，具体详见：configs/data/KuavoRosbag2Lerobot.yaml，configs/deploy/kuavo_env.yaml等。


## 相关文档索引

| 主题 | 文档 |
|------|------|
| 快速上手 | [快速开始](getting_started/quick_start.md) |
| 安装与环境 | [安装指南](getting_started/installation.md) |
| 文件与目录 | [项目文件构成](getting_started/file_structure.md) |
| 策略对比与选型 | [策略概览](concepts/policy_overview.md)、[训练策略对比](training/comparison.md) |
| RGB-Depth 融合 | [RGB-深度融合](advanced/rgb_depth_fusion.md)、[深度支持说明](advanced/depth_support.md) |
| 训练与多卡 | [训练流水线](training/pipeline.md)、[多卡训练](training/multi_gpu.md) |
| 部署 | [仿真自动测试](deployment/sim_auto_test.md)、[真机评测](deployment/real_eval.md) |
| 常见问题 | [常见问题](faq.md) |

---

## 更多资讯

乐聚 OpenLET 社区会持续发布赛事动态、技术经验分享与开源资源更新，可前往 [OpenLET 资讯](https://openlet.openatom.tech/explore/journalism) 浏览。
