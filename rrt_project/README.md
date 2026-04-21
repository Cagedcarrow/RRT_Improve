# CCI-Bridge RRT Project

> 面向 2D/3D 狭窄通道路径规划的 **RRT 改进工程实现**（首版即 CUDA、无头批处理、可断点恢复、支持自动回代调参）

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#10-quick-start)
[![CUDA](https://img.shields.io/badge/CUDA-PyTorch-green.svg)](#8-cuda-执行的优势与困难)
[![Pipeline](https://img.shields.io/badge/Pipeline-M1~M5-orange.svg)](#6-实验流程与里程碑m1m5)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. 数学改进：本算法改进了什么](#2-数学改进本算法改进了什么)
- [3. 代码结构与每个模块的含义](#3-代码结构与每个模块的含义)
- [4. 算法性能提升与工程意义](#4-算法性能提升与工程意义)
- [5. 核心数据流（从输入到结果）](#5-核心数据流从输入到结果)
- [6. 实验流程与里程碑（M1~M5）](#6-实验流程与里程碑m1m5)
- [7. 失败分类与自动回代调参](#7-失败分类与自动回代调参)
- [8. CUDA 执行的优势与困难](#8-cuda-执行的优势与困难)
- [9. 当前结果解读](#9-当前结果解读)
- [10. Quick Start](#10-quick-start)
- [11. 夜间无人值守运行](#11-夜间无人值守运行)
- [12. 已知限制与下一步改进](#12-已知限制与下一步改进)
- [13. FAQ](#13-faq)

---

## 1. Overview

本项目实现了一个可复现实验框架，目标是在窄通道环境中提升 RRT 系列算法的成功率与稳定性。

核心特点：

- 支持 2D 和 3D 两类场景，每类都包含简单/复杂障碍。
- 统一接口封装 `RRT / RRT-Connect / RRT* / CCI-Bridge RRT*`。
- 采用 CUDA 加速碰撞检测（`torch` 张量批处理）。
- 支持无头批处理实验、阶段 checkpoint、速率限制、自动回代调参。

对应关键产物：

- 实验记录：[`results/records.csv`](results/records.csv)
- 汇总统计：[`results/summary.json`](results/summary.json)
- 阶段状态：[`results/checkpoints/stage_state.json`](results/checkpoints/stage_state.json)

[Back to Top](#table-of-contents)

---

## 2. 数学改进：本算法改进了什么

这一节对应你的第 1 条需求：**“本算法的数学改进的地方”**。

### 2.1 基线 RRT 的核心问题

经典 RRT 在窄通道中常见两个问题：

- 自由空间测度很小，均匀采样命中率低。
- 碰撞约束太“硬”，在复杂障碍附近扩展容易连续失败。

### 2.2 改进一：CCI（Collision Constraint Interpolation）思想的工程化近似

在本实现中，障碍物是 AABB，使用 `alpha` 控制障碍“膨胀/收缩”过程：

\[
\text{scale}(\alpha) = s_{min} + (1-s_{min})\alpha,\quad s_{min}=0.25,\ \alpha\in[0,1]
\]

当 `alpha` 从 0 到 1 递进时，规划器先在更“宽松”的空间找到可行趋势，再逐步逼近真实约束。

对应代码：

- 线性 `alpha` 调度：[`core/cci.py`](core/cci.py)
- 障碍随 `alpha` 缩放的 SDF：[`env/obstacles.py`](env/obstacles.py)
- 规划中按进度调用 `alpha_provider(progress)`：[`algorithms/rrt_star.py`](algorithms/rrt_star.py)

### 2.3 改进二：Bridge Sampling（桥接采样）

桥接采样目标是提高窄通道采样命中率。

算法简化流程：

1. 随机取障碍内点 `x,p`。
2. 计算中点 `m=(x+p)/2`，要求 `m` 在自由空间。
3. 沿正交方向扰动 `m+\delta d_k`，统计可通方向数 `free_count`。
4. 只保留 `free_count==1` 的候选点，聚类后得到桥接代表点集合。

这在窄通道附近会比完全均匀采样更高效。

对应代码：[`core/bridge_sampler.py`](core/bridge_sampler.py)

### 2.4 改进三：三混合采样策略（uniform + goal + bridge）

采样分布为：

\[
q\sim
\begin{cases}
q_{goal}, & r< p_{goal}\\
q_{bridge}, & p_{goal}\le r < p_{goal}+p_{bridge}\\
q_{uniform}, & \text{otherwise}
\end{cases}
\]

这样做的意义：

- `goal_bias`：保证收敛方向。
- `bridge_bias`：提升窄通道命中率。
- `uniform`：维持全局探索能力。

对应代码：[`core/sampling_policy.py`](core/sampling_policy.py)

### 2.5 改进四：RRT* 重连与代价优化

RRT* 通过邻域重连优化路径代价：

\[
J(q_{new}) = \min_{q_j\in\mathcal{N}} \left( J(q_j) + \|q_{new}-q_j\| \right)
\]

并在插入后尝试 rewiring 邻域节点，获得更短路径。

对应代码：[`algorithms/rrt_star.py`](algorithms/rrt_star.py)

### 2.6 几何基础：AABB 的 SDF

本项目碰撞检测基于 AABB SDF：

\[
q = |p-c|-e,
\quad
\text{SDF}(p)=\|\max(q,0)\|_2 + \min(\max(q),0)
\]

- SDF <= 0 视为碰撞或在障碍内部。
- CUDA 上对 segment 采样点批量并行计算 SDF。

对应代码：[`core/collision_backend.py`](core/collision_backend.py)

[Back to Top](#table-of-contents)

---

## 3. 代码结构与每个模块的含义

这一节对应你的第 2 条需求：**“各个代码的含义以及功能”**。

### 3.1 总目录

```text
rrt_project/
  algorithms/      # 规划算法
  core/            # 采样、CCI、碰撞后端
  env/             # 场景和障碍定义
  benchmark/       # 批量实验、限速、回代调参
  analysis/        # 同伦分析与报告
  scripts/         # 夜间无人值守脚本
  configs/         # 默认配置
  results/         # 运行结果
```

### 3.2 `algorithms/`（算法层）

| 文件 | 主要职责 | 关键对象 |
|---|---|---|
| [`algorithms/base_planner.py`](algorithms/base_planner.py) | 定义统一契约与公共工具函数 | `PlannerConfig`, `PlannerResult`, `BasePlanner` |
| [`algorithms/rrt.py`](algorithms/rrt.py) | 基础单树 RRT | `RRTPlanner` |
| [`algorithms/rrt_connect.py`](algorithms/rrt_connect.py) | 双树快速连接 | `RRTConnectPlanner` |
| [`algorithms/rrt_star.py`](algorithms/rrt_star.py) | 成本优化与重连 | `RRTStarPlanner` |
| [`algorithms/cci_bridge_rrt.py`](algorithms/cci_bridge_rrt.py) | CCI + Bridge 的改进版 | `CCIBridgeRRTPlanner` |

### 3.3 `core/`（核心算子）

| 文件 | 主要职责 | 关键点 |
|---|---|---|
| [`core/collision_backend.py`](core/collision_backend.py) | 批量线段碰撞检测（CPU/CUDA） | `batch_segment_check` 返回 `BoolTensor` |
| [`core/bridge_sampler.py`](core/bridge_sampler.py) | 窄通道桥接候选生成与聚类 | `free_count == 1` 过滤 |
| [`core/cci.py`](core/cci.py) | `alpha` 进度调度 | `AlphaScheduler.alpha_at` |
| [`core/sampling_policy.py`](core/sampling_policy.py) | 三混合采样策略 | goal / bridge / uniform |

### 3.4 `env/`（环境层）

| 文件 | 主要职责 | 关键点 |
|---|---|---|
| [`env/obstacles.py`](env/obstacles.py) | AABB + SDF 定义 | `sdf()`, `is_inside()` |
| [`env/scene_state.py`](env/scene_state.py) | 场景状态与采样接口 | `sample_free()` |
| [`env/scene_2d.py`](env/scene_2d.py) | 2D 简单/复杂场景构造 | `build_scenes_2d()` |
| [`env/scene_3d.py`](env/scene_3d.py) | 3D 简单/复杂场景构造 | `build_scenes_3d()` |

### 3.5 `benchmark/`（实验层）

| 文件 | 主要职责 | 关键点 |
|---|---|---|
| [`benchmark/runner.py`](benchmark/runner.py) | M1~M5 全流程执行 | checkpoint + summary 产出 |
| [`benchmark/rate_limiter.py`](benchmark/rate_limiter.py) | 每分钟请求数限速 | 滑动窗口限速 |
| [`benchmark/failure_classifier.py`](benchmark/failure_classifier.py) | 失败类型分类 | timeout/collision_stuck/unstable |
| [`benchmark/tuner.py`](benchmark/tuner.py) | 规则化调参 | 基于失败占比更新参数 |
| [`benchmark/metrics.py`](benchmark/metrics.py) | 汇总统计 | success_rate/median_time 等 |

### 3.6 `analysis/`（分析层）

| 文件 | 主要职责 |
|---|---|
| [`analysis/homotopy.py`](analysis/homotopy.py) | 缠绕数和 L-value 近似分析接口 |
| [`analysis/report.py`](analysis/report.py) | 将 `summary.json` 渲染成 `summary_report.md` |

### 3.7 `scripts/`（运维层）

| 文件 | 主要职责 |
|---|---|
| [`scripts/night_run.sh`](scripts/night_run.sh) | 一键夜间无人值守执行，支持 `--daemon/--tmux/--fresh` |
| [`scripts/rate_guard.py`](scripts/rate_guard.py) | 独立速率门控工具 |

[Back to Top](#table-of-contents)

---

## 4. 算法性能提升与工程意义

这一节对应你的第 3 条需求：**“算法提升性能，意义”**。

### 4.1 性能提升机制

- 对窄通道成功率提升：Bridge 采样在窄瓶颈区域提高有效样本概率。
- 对收敛稳定性提升：CCI `alpha` 递进降低早期碰撞约束难度。
- 对路径质量提升：RRT* rewiring 降低路径长度。
- 对吞吐提升：CUDA 批量段碰撞检测减轻 CPU 压力。

### 4.2 工程意义

- 可以直接做“算法对比平台”，而不是单次 demo。
- 支持批量场景/多种子，结果具有统计意义。
- 支持断点恢复与夜间运行，适合长时间验证任务。
- 支持自动回代调参，缩短“发现问题 -> 修复 -> 复验”闭环。

### 4.3 面向真实项目的价值

- 对自动驾驶/机器人/无人系统中的狭窄通道规划问题有直接参考价值。
- 对 AI 研发流程友好：模块化、可测、可观察、可迭代。

[Back to Top](#table-of-contents)

---

## 5. 核心数据流（从输入到结果）

1. 读取配置（`configs/default.yaml`）。
2. 构造 2D/3D 场景（`env/scene_2d.py`, `env/scene_3d.py`）。
3. 创建规划器（`benchmark/runner.py -> planners_factory()`）。
4. 对每个 `scene × planner × seed` 执行 `plan()`。
5. 在规划循环中调用：
   - `MixedSampler.sample()`
   - `CollisionBackend.batch_segment_check()`
   - `AlphaScheduler.alpha_at(progress)`（改进算法）
6. 记录到 `results/records.csv`。
7. 聚合为 `results/summary.json`。
8. 执行失败分类 + 调参回代（M5）。

[Back to Top](#table-of-contents)

---

## 6. 实验流程与里程碑（M1~M5）

| 里程碑 | 内容 | 输出 |
|---|---|---|
| M1 | 基线（RRT/RRT-Connect/RRT*）+ CUDA 通路 | 基线记录 |
| M2 | CCI 接入（alpha schedule） | 改进初版记录 |
| M3 | Bridge 采样接入 | 窄通道增强记录 |
| M4 | 2D/3D 简单+复杂全量评测 | 综合对比结果 |
| M5 | 两轮自动回代调参 | `tuning_history` + 最终配置 |

执行入口：

```bash
python -m rrt_project.benchmark.runner --pipeline full --limit-per-minute 40
```

[Back to Top](#table-of-contents)

---

## 7. 失败分类与自动回代调参

### 7.1 失败分类规则

定义在 [`benchmark/failure_classifier.py`](benchmark/failure_classifier.py)：

- `timeout`：迭代接近上限仍失败。
- `collision_stuck`：碰撞压力过高（`collision_checks/iters_used` 高）。
- `unstable`：桥接使用率过低导致不稳定表现。
- `overlong_path`：成功但路径显著劣于基线长度。

### 7.2 调参映射规则

定义在 [`benchmark/tuner.py`](benchmark/tuner.py)：

- `collision_stuck` 多：提高 `bridge_bias`、降低 `step_size`。
- `timeout` 多：提高 `goal_bias`，并提升 `alpha_schedule` 前段值。
- `overlong_path` 多：降低 `step_size`。
- `unstable` 多：轻微降低 `step_size`。

### 7.3 回代结果记录

- 回代历史写入：`summary.json.tuning_history`
- 阶段完成状态写入：`checkpoints/stage_state.json`

[Back to Top](#table-of-contents)

---

## 8. CUDA 执行的优势与困难

这一节对应你的第 4 条需求：**“用CUDA执行的困难以及优势”**。

### 8.1 优势

- 批量并行：`[B, 2, D]` 线段一次性并行采样与 SDF 计算。
- 统一张量管线：CPU/GPU 一套接口切换，便于对照测试。
- 对大批量评测友好：随着 `B` 提升，GPU 更有吞吐优势。

### 8.2 现实困难

- 小 batch 下 GPU 启动与数据传输开销可能抵消收益。
- 数值一致性问题：CPU 与 GPU 浮点误差不同，需要容差比较。
- 调试复杂：张量维度错误、设备不一致（CPU tensor + CUDA tensor）常见。
- 显存压力：场景复杂、采样密度高时需要控制 `samples_per_segment`。

### 8.3 本项目如何应对

- 通过 `cuda_enabled` 开关支持 CPU fallback。
- 在 `CollisionBackend._device()` 统一管理设备选择。
- 单元测试中保留 CPU 路径抽检，降低 CUDA-only 风险。
- `RateLimiter` + 阶段 checkpoint 确保长跑稳定。

### 8.4 建议的 CUDA 优化方向

- 合并 obstacle 张量，减少 Python 循环开销。
- 对常见场景缓存障碍参数到 GPU 常驻内存。
- 在 `batch_segment_check` 中引入更激进的向量化和早停策略。

[Back to Top](#table-of-contents)

---

## 9. 当前结果解读

数据来源：[`results/summary.json`](results/summary.json)

| Planner | Runs | Success Rate | Median Time (ms) | Median Path Length |
|---|---:|---:|---:|---:|
| rrt | 60 | 0.8333 | 131.25 | 1.8934 |
| rrt_connect | 60 | 0.9167 | 102.88 | 1.9028 |
| rrt_star | 60 | 0.8333 | 229.03 | 1.5972 |
| cci_bridge_rrt | 120 | 1.0000 | 234.53 | 1.3890 |

简要解读：

- 改进算法 `cci_bridge_rrt` 成功率最高。
- `rrt_connect` 时间表现较优。
- `rrt_star` 和 `cci_bridge_rrt` 路径质量（长度）更优。
- 当前失败主要是 `timeout`，说明瓶颈仍在复杂场景下的收敛效率。

[Back to Top](#table-of-contents)

---

## 10. Quick Start

### 10.1 环境准备

```bash
cd /root/rrt_improve
source .venv_rrt_clean/bin/activate
```

### 10.2 运行全流程

```bash
cd /root/rrt_improve
python -m rrt_project.benchmark.runner --pipeline full --limit-per-minute 40
```

### 10.3 查看结果

```bash
cat rrt_project/results/summary.json
cat rrt_project/results/reports/summary_report.md
```

[Back to Top](#table-of-contents)

---

## 11. 夜间无人值守运行

推荐脚本：[`scripts/night_run.sh`](scripts/night_run.sh)

### 11.1 后台执行（nohup）

```bash
cd /root/rrt_improve
bash rrt_project/scripts/night_run.sh --daemon --limit 40
```

### 11.2 tmux 执行

```bash
cd /root/rrt_improve
bash rrt_project/scripts/night_run.sh --tmux --limit 40
```

### 11.3 从头执行/断点恢复

```bash
# 从头
bash rrt_project/scripts/night_run.sh --fresh --limit 40

# 默认断点恢复（含 --resume）
bash rrt_project/scripts/night_run.sh --limit 40
```

### 11.4 单实例保护

脚本内使用 `flock`，避免重复启动相同流水线。

[Back to Top](#table-of-contents)

---

## 12. 已知限制与下一步改进

### 12.1 已知限制

- 当前 CCI 是工程化近似（AABB 缩放），未实现论文级凸分解与严格 SDF 插值。
- Bridge 聚类是轻量半径聚类，不是完整 DBSCAN。
- 目前未加入机械臂约束和系绳拓扑后端主流程。

### 12.2 下一步建议

- 引入严格凸分解 + 叶子集序列（接近论文原法）。
- 升级桥接聚类到标准 DBSCAN，增强鲁棒性。
- 在复杂 3D 场景中加入启发式 nearest-neighbor 结构（KD-tree/FAISS）。
- 增加可视化回放（仅分析用途，不影响 headless 主线）。

[Back to Top](#table-of-contents)

---

## 13. FAQ

### Q1：为什么改进算法时间不一定最短？
A：CCI + Bridge + RRT* 重连增加了计算负担，通常以更高成功率和更优路径质量换取时间。

### Q2：`limit-per-minute` 作用于什么？
A：作用于 benchmark 任务调度事件频率（滑动窗口限速），用于降低长期运行中触发外部速率问题的概率。

### Q3：可以只跑 2D 或只跑某个算法吗？
A：可以，直接修改 `benchmark/runner.py` 中 `run_full_pipeline()` 对 scenes/planners 的选择逻辑。

### Q4：如何验证 CUDA 真正生效？
A：查看 `records.csv` 的 `cuda_enabled/device_name` 字段，以及运行时 `torch.cuda.is_available()`。

---

## Citation / Acknowledgement

本工程实现参考了本目录中的窄通道规划与同伦/拓扑相关论文资料，并做了面向工程执行的统一框架重构。

