# Algorithm Improvement Report

## 1. 项目背景与目标

本次改造目标不是“小修补”，而是把原有可运行版本升级为可审阅、可复现、可展示的**论文对齐工程版本**。核心方向：

1. 将论文关键思想落实到代码结构与数据契约中。
2. 从“只有数值日志”升级到“全场景全算法可视化资产”。
3. 从少量 smoke test 升级到数学正确性、组件正确性、系统契约的一体化测试。
4. 在 CUDA 可用前提下，补齐一致性验证与工程风险控制。

---

## 2. 论文对齐的数学改进

### 2.1 CCI 插值框架工程化

参考 `2410.20697v1` 的 Collision Constraint Interpolation 思想，新增：

- `shaping_function`：
  \[
  f(x)=\frac{e^{\eta x}-1}{\eta}
  \]
  并加入指数裁剪避免数值爆炸。
- `interpolate_sdf(sdf_a, sdf_b, alpha, eta)`：在 \(\alpha\in[0,1]\) 上做平滑插值。
- `AlphaScheduler + AlphaExecutor`：统一 alpha 调度与 trace 采集。
- `build_leaf_sequence`：实现叶子集序列（leaf set sequence）近似生成，输出 `init_indices + batches + parent_map`。

对应文件：
- [`core/cci.py`](../core/cci.py)

### 2.2 改进 Bridge Sampling + DBSCAN

参考 `sensors-26-01582` 的 improved bridge sampling，新增：

- `f == 1` 单通道候选过滤。
- 两种聚类模式：`radius` 与 `dbscan`。
- 诊断统计 `BridgeDiagnostics`：
  - `candidate_density`
  - `corner_false_positive_rate`
  - `cluster_validity`

对应文件：
- [`core/bridge_sampler.py`](../core/bridge_sampler.py)

### 2.3 采样策略可观测化

保持 `uniform + goal + bridge` 三混合采样，同时把采样来源计数纳入运行数据：

- `sample_source_counts = {goal, bridge, uniform}`

对应文件：
- [`core/sampling_policy.py`](../core/sampling_policy.py)

### 2.4 消融实验模式

新增 `ablation_mode`，统一支持：

- `baseline`
- `cci_only`
- `bridge_only`
- `cci_bridge`

并在改进规划器中自动映射 `enable_cci / enable_bridge`，用于量化每项改进贡献。

对应文件：
- [`algorithms/base_planner.py`](../algorithms/base_planner.py)
- [`algorithms/cci_bridge_rrt.py`](../algorithms/cci_bridge_rrt.py)

---

## 3. 代码重构清单（工作量体现）

### 3.1 核心契约升级

- `PlannerConfig` 增加：
  - `bridge_clustering`
  - `cci_eta`
  - `ablation_mode`
- `PlannerResult.meta` 从 `Dict[str,float]` 扩展到 `Dict[str,Any]`，允许结构化诊断数据。

对应文件：
- [`algorithms/base_planner.py`](../algorithms/base_planner.py)

### 3.2 规划器内部解耦（以 RRT* 为主）

`RRT*` 主流程拆为多个子过程：

- `_prepare_run`
- `_sample_state`
- `_choose_parent`
- `_rewire`
- `_run`

并在结果中输出：

- `alpha_trace`
- `alpha_trace_stats`
- `sample_source_counts`
- `bridge_candidate_stats`
- `tree_nodes`（用于路径图可视化）

对应文件：
- [`algorithms/rrt_star.py`](../algorithms/rrt_star.py)
- [`algorithms/rrt.py`](../algorithms/rrt.py)
- [`algorithms/rrt_connect.py`](../algorithms/rrt_connect.py)

### 3.3 Runner 重构为实验矩阵驱动

- 新增矩阵执行：`scene × planner × ablation × seed`
- 两轮自动回代：`TUNE_R1`、`TUNE_R2`
- 增强 CSV 字段契约，包含诊断与图片路径。
- 自动生成 collage 总览图与 markdown 报告。

对应文件：
- [`benchmark/runner.py`](../benchmark/runner.py)
- [`benchmark/metrics.py`](../benchmark/metrics.py)
- [`benchmark/tuner.py`](../benchmark/tuner.py)
- [`benchmark/failure_classifier.py`](../benchmark/failure_classifier.py)

### 3.4 可视化资产导出

新增统一可视化器：

- 2D：障碍物 + tree nodes + 最终路径 + start/goal。
- 3D：双视角图（单图双子图）便于 GitHub 展示。
- 自动输出拼图：`collage_overview.png`。

对应文件：
- [`analysis/visualizer.py`](../analysis/visualizer.py)

---

## 4. 测试标准扩展

## 4.1 测试覆盖策略

本轮测试从“少量 smoke”升级为“数学 + 组件 + 契约”的组合：

- CCI 数学性质：单调性、边界一致性、调度正确性。
- Leaf sequence 输出合法性。
- Bridge DBSCAN 聚类正确性。
- Collision CPU/GPU 一致性（可用时）。
- Visualizer 文件产出正确性。
- Runner CSV 契约字段完整性。

测试文件：
- [`tests/test_cci.py`](../tests/test_cci.py)
- [`tests/test_bridge_sampler.py`](../tests/test_bridge_sampler.py)
- [`tests/test_collision_backend.py`](../tests/test_collision_backend.py)
- [`tests/test_visualizer.py`](../tests/test_visualizer.py)
- [`tests/test_runner_contract.py`](../tests/test_runner_contract.py)

执行结果：
- `11 passed`

---

## 5. 高强度实验执行结果

### 5.1 运行矩阵

- seeds: `20`（0~19）
- scenes: `4`（2D/3D × simple/complex）
- planners: `4`（rrt, rrt_connect, rrt_star, cci_bridge_rrt）
- ablations: `4`（baseline, cci_only, bridge_only, cci_bridge）
- matrix runs: `1280`
- tuning runs: `80`（两轮）
- total records: `1360`

结果文件：
- [`results/records.csv`](../results/records.csv)
- [`results/summary.json`](../results/summary.json)
- [`results/reports/summary_report.md`](../results/reports/summary_report.md)

### 5.2 可视化资产

- 路径图：`1280` 张
- 汇总拼图：[`results/reports/collage_overview.png`](../results/reports/collage_overview.png)
- 图目录：[`images/`](../images)

示例路径图：
- [`images/2d_complex/cci_bridge_rrt/cci_bridge_seed_0.png`](../images/2d_complex/cci_bridge_rrt/cci_bridge_seed_0.png)
- [`images/3d_complex/rrt_star/baseline_seed_0.png`](../images/3d_complex/rrt_star/baseline_seed_0.png)

### 5.3 部分关键指标（来自 summary）

- `rrt::baseline` success_rate = `0.725`
- `rrt_connect::baseline` success_rate = `0.775`
- `rrt_star::baseline` success_rate = `0.725`
- `cci_bridge_rrt::cci_only` success_rate = `1.000`
- `cci_bridge_rrt::cci_bridge` success_rate = `0.994`

说明：当前参数下，`cci_bridge_rrt` 在 `cci_only` 与 `cci_bridge` 模式表现显著优于基线组；`bridge_only` 在复杂场景的收益依赖参数与场景结构，已在回代中体现为调参方向。

---

## 6. CUDA 执行的优势与困难

### 6.1 优势

- `batch_segment_check` 在大量线段碰撞判断时可并行处理。
- 与 CPU 共用一套接口，便于统一评测与回归。
- 在大规模矩阵实验中有效降低总 wall-time。

### 6.2 工程困难

- 小批量时 GPU 启动与数据搬运开销不一定划算。
- CPU/GPU 浮点细节差异导致边界点行为可能不同。
- 长任务中设备状态、驱动稳定性需要额外守护。

### 6.3 本项目处理方式

- 每次 run 记录 `cuda_enabled/device_name`。
- 新增 `cpu_gpu_consistency` 字段做抽样一致性检查。
- 限速 + checkpoint + retry 保证长流程稳定。

---

## 7. 产物与交付清单

代码：
- 核心重构完成（CCI、Bridge、Runner、Visualizer、测试）

数据：
- `records.csv`
- `summary.json`
- `stage_state.json`
- `rate_state.json`

图片：
- `images/**` 全量路径图
- `collage_overview.png` 总览

文档：
- 本报告：`docs/ALGORITHM_IMPROVEMENT_REPORT.md`
- 汇总报告：`results/reports/summary_report.md`

---

## 8. 下一步建议

1. 在复杂 3D 场景上引入更严格的 leaf-set/convex decomposition 近似实现，进一步贴近 CCI 原文。
2. 为 Bridge 采样增加角点误检可视化热力图，便于诊断 `bridge_only` 的收益边界。
3. 增加 per-scene/per-mode 的统计图（箱线图或分布图），提升论文图表质量。
4. 将 `tree_nodes` 进一步压缩存储，降低超大规模实验日志体积。

