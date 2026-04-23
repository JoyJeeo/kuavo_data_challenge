# LeRobot 数据集知识总结（结合当前项目实测）

## 1. Episode 的含义

在 LeRobot 中，`Episode` 是一条完整任务轨迹：机器人从起始状态开始，到任务结束/重置/中断为止的全过程。

- 一个 Episode 由连续的多帧（`Frame`）组成。
- 每帧通常包含：
  - `Observation`：图像、关节状态等
  - `Action`：该时刻控制指令

### 1.1 Episode 的作用

- 作为模仿学习中的基本演示单元
- 便于数据筛选（删除异常演示）
- 便于训练/验证划分（尽量按 episode 切分，避免泄漏）

### 1.2 v2.1 与 v3.0 的差异（Episode 相关）

- v2.1：常见为“每个 episode 单独文件”组织
- v3.0：多个 episode 合并为分片文件，通过索引与元数据定位各 episode 的帧范围，支持更高效流式访问

当前项目实测：`meta/info.json` 中 `codebase_version = "v3.0"`，目录结构为 `data/chunk-xxx/file-xxx.parquet` + `videos/.../chunk-xxx/file-xxx.mp4`，符合 v3.0。

---

## 2. 三个关键分块参数

在 `meta/info.json` 中：

- `chunks_size`
- `data_files_size_in_mb`
- `video_files_size_in_mb`

它们共同影响数据写入时的文件轮换与目录分块策略。

## 2.1 参数含义

- `chunks_size`：每个 chunk 的目标帧规模（或分片粒度配置），用于决定何时进入下一个 chunk。
- `data_files_size_in_mb`：数据侧文件（parquet）目标大小配置（MB）。
- `video_files_size_in_mb`：视频侧文件（mp4）目标大小配置（MB）。

> 说明：实际落盘时会综合实现细节（编码器、写入器、分片策略）来决定最终文件数量与边界，通常不是严格“恰好等于阈值就切”。

## 2.2 直观流程（工程视角）

1. 持续写入数据/视频到当前分片。
2. 当达到分片策略阈值（帧数或体积策略）时切换到新文件。
3. 当当前 chunk 达到分块策略后，写入下一个 chunk 目录。

## 2.3 示例

### 示例 A：大文件优先（高吞吐）

- 参数：`data_files_size_in_mb=500`，`video_files_size_in_mb=2000`，`chunks_size=50`
- 结果：文件数量较少，I/O 顺序性好，但单文件更大。

### 示例 B：小文件优先（便于抽样调试）

- 参数：`data_files_size_in_mb=10`，`video_files_size_in_mb=50`，`chunks_size=20`
- 结果：文件切分更频繁，目录层级增多，便于小批量排查。

### 示例 C：当前数据集参数（稳妥默认）

- 实际值：`data_files_size_in_mb=100`，`video_files_size_in_mb=500`，`chunks_size=1000`
- 你这个数据集规模较小（`total_frames=1190`），目前仅见 `chunk-000`，符合预期。

---

## 3. `episode_index` 统计量如何得到

`episode_index` 是每一帧所属的 episode 编号。统计方法：

- 收集所有帧的 `episode_index` 列
- 计算：`min/max/mean/std/count/quantiles`

## 3.1 本项目实测结果

对 `data/chunk-000/file-000.parquet` 实测：

- `count = 1190`
- `min = 0`
- `max = 7`
- `mean = 3.5`
- `std(ddof=0) = 2.2923878417`
- 分位数（线性插值）：
  - `q01 = 0`
  - `q10 = 0`
  - `q50 = 3.5`
  - `q90 = 7`
  - `q99 = 7`

且按值计数约均匀：

- episode 0: 149 帧
- episode 1: 149 帧
- episode 2: 148 帧
- episode 3: 149 帧
- episode 4: 149 帧
- episode 5: 148 帧
- episode 6: 149 帧
- episode 7: 149 帧

## 3.2 为什么 `stats.json` 里的分位数看起来异常

当前 `meta/stats.json` 里 `episode_index` 的 `q01/q10/q50/q90/q99` 均接近 `3.5`，与原始列重算结果不一致。

结论：这更像“统计生成流程/显示上的异常”，而不是原始数据本身异常。已验证 `min/max/mean/std/count` 与原始数据一致。

---

## 4. `stats.json` 中常见统计项解释

对任意特征（如 `action`、`observation.state`、`timestamp`）：

- `min/max`：最小/最大
- `mean/std`：均值/标准差
- `count`：样本数
- `q01/q10/q50/q90/q99`：1%/10%/50%/90%/99% 分位数

分位数用途：

- 判断分布尾部是否有异常值
- 作为裁剪范围（如按 `q01~q99` clip）
- 与 `min/max` 结合看是否存在离群点

---

## 5. 可复用验证代码

```python
import pandas as pd

s = pd.read_parquet('data/chunk-000/file-000.parquet')['episode_index']

print('count', s.shape[0])
print('min', s.min(), 'max', s.max())
print('mean', s.mean(), 'std(ddof=0)', s.std(ddof=0))
print('quantiles', s.quantile([0.01, 0.1, 0.5, 0.9, 0.99], interpolation='linear').to_dict())
print('value_counts')
print(s.value_counts().sort_index())
```

---

## 6. 当前数据集（你这个仓库）一页结论

- 格式版本：`v3.0`
- 总轨迹数：`8`（episode 0~7）
- 总帧数：`1190`
- 总任务数：`1`
- 帧率：`10 fps`
- 视觉特征：3 路视频（`head_cam_h`、`wrist_cam_l`、`wrist_cam_r`）
- 关键提醒：`stats.json` 中 `episode_index` 分位数显示异常，建议以原始 parquet 重算结果为准。

---

## 7. 入门友好补充版（概念 + 工程落地）

## 7.1 Episode：可以把它当成“一条完整演示”

- 一个 Episode 是机器人从开始到结束完成一次任务的全过程。
- 一个数据集就是很多 Episode 的集合。
- 每个 Episode 由连续帧构成，每帧包含：
  - 观测（图像、状态等）
  - 动作（控制指令）

一个直观例子：

- 机器人启动
- 接近红色方块
- 抓取
- 放到目标位置
- 结束

这整段就是 1 个 Episode。

为什么这个概念重要：

- 训练时可以把“完整成功范例”作为学习单元
- 管理时可以按 Episode 做筛选、切分、排错
- 评估时可按 Episode 统计成功率与稳定性

## 7.2 v3.0 中 Episode 的存储思路

- v2.x 常见组织：一个 episode 对应一个独立文件
- v3.0 组织：多个 Episode 合并在分片文件中，通过元数据和索引定位边界
- 优点：更适合大规模数据、流式读取和并行训练

## 7.3 三个分块参数如何协同工作

在 `meta/info.json` 中：

- `data_files_size_in_mb`：数据 parquet 文件的目标大小阈值
- `video_files_size_in_mb`：视频 mp4 文件的目标大小阈值
- `chunks_size`：分块粒度阈值（与文件轮换策略共同决定何时进入新 chunk）

可把它理解为“双层滚动写入”：

1. 先看单文件阈值：文件接近目标大小就轮换到新文件。
2. 再看分块阈值：当当前 chunk 达到分块策略条件后，进入新 chunk 目录。

说明：

- 不同版本实现细节可能不同，实际落盘边界可能受编码器、缓冲、批写策略影响。
- 因此更建议通过实际输出目录检查最终切分结果，而不是只按阈值做静态推断。

## 7.4 参数选型示例

示例 A：减少文件数量（偏吞吐）

- `data_files_size_in_mb=500`
- `video_files_size_in_mb=2000`
- `chunks_size=50`
- 结果倾向：大文件、少文件，适合顺序读取和高吞吐训练。

示例 B：便于调试与可视化

- `data_files_size_in_mb=10`
- `video_files_size_in_mb=50`
- `chunks_size=20`
- 结果倾向：小文件、更多 chunk，定位和抽样更方便。

示例 C：官方推荐起步（稳妥）

- `data_files_size_in_mb=100`
- `video_files_size_in_mb=500`
- `chunks_size=1000`
- 结果倾向：在大多数中等规模数据集上平衡性能和可管理性。

## 7.5 实践建议

- 初次建集先用默认值，避免过早微调分块参数。
- 先看训练瓶颈在哪：
  - I/O 压力大：适当增大单文件阈值，减少碎文件。
  - 调试频繁：适当减小阈值，提升定位效率。
- 每次改参数后都做一次“落盘体检”：
  - chunk 数量
  - 单文件大小分布
  - 数据/视频文件数量比例
  - 样本读取速度
