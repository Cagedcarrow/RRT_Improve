# RRT 改进项目完整执行计划（2D/3D + 首版 CUDA + 无头评测迭代）

版本：v1.0  
日期：2026-04-21  
适用范围：本目录资料驱动的 AI 实施与工程落地

---

## 第 1 章 项目目标

### 1.1 总体目标
将现有 `docx` 指导内容重构为一份可直接驱动 AI 开发执行的工程计划，并据此完成一个可无头批量评测的 RRT 改进项目主线。

### 1.2 主线范围
- 场景范围：仅 2D 与 3D。
- 难度分层：每个维度都包含“简单障碍场景 + 复杂障碍场景”。
- 算法目标：以 CCI + 改进桥接采样为核心改进，在基线 RRT 家族上验证提升。
- 运行模式：无头（headless）批处理。
- 评测策略：固定种子批量跑，自动日志统计，基于失败模式回代改进。

### 1.3 非目标（默认不做）
- GUI 可视化交互系统。
- 机械臂专用约束求解管线。
- 系绳机器人专用拓扑后端作为主验收项。
- ROS 强依赖或仿真平台强绑定。

### 1.4 成功判定
- 改进算法在复杂场景成功率较基础 RRT 提升不少于 15%。
- 中位规划时间不劣于 RRT* 超过 30%。
- 全流程可复现（固定种子重跑结果波动在可接受范围内）。
- 至少完成两轮“数据驱动回代改进”并提供对照记录。

---

## 第 2 章 论文方法映射

### 2.1 资料修正声明
- 本项目参考资料为 6 篇，而非“5 篇核心论文”。
- 分级策略：3 篇主干 + 3 篇扩展。

### 2.2 资料分级

| 分级 | 文件 | 作用 | 是否阻塞主线 |
|---|---|---|---|
| 主干 | `2410.20697v1.pdf` | CCI（Collision Constraint Interpolation）与窄通道可行性引导 | 是 |
| 主干 | `sensors-26-01582.pdf` | Bridge-point 引导策略与窄通道采样思想 | 是 |
| 主干 | `planwithhomotopyconstraints_aaai10.pdf` | 同伦类别表示（L-value）用于分析验证 | 否（验证增强） |
| 扩展 | `2603.26696v1.pdf` | 缠绕数/拓扑能量思想（作为拓扑扩展参考） | 否 |
| 扩展 | `kinetic-triang.pdf` | 运动与变形环境剖分/碰撞结构参考 | 否 |
| 扩展 | `2010.08167v1.pdf` | 形变/动态障碍下路径优化思想参考 | 否 |

### 2.3 概念到模块映射

| 概念 | 实现模块 | 用途 |
|---|---|---|
| CCI 与 alpha 递进环境 | `core/cci.py`, `env/scene_state.py` | 将窄通道难题分解为连续子问题 |
| 改进桥接采样 | `core/bridge_sampler.py` | 提高窄通道样本命中率 |
| CUDA 碰撞检测 | `core/collision_backend.py` | 批量段检测加速 |
| 基线与改进规划器 | `algorithms/*.py` | 统一接口下做公平对比 |
| 同伦/L 值分析 | `analysis/homotopy.py` | 结果解释与扩展验证 |
| 无头评测与汇总 | `benchmark/runner.py` | 批量实验、日志落盘、摘要生成 |

---

## 第 3 章 系统架构

### 3.1 建议目录结构

```text
rrt_project/
  algorithms/
    base_planner.py
    rrt.py
    rrt_connect.py
    rrt_star.py
    cci_bridge_rrt.py
  core/
    collision_backend.py
    bridge_sampler.py
    cci.py
    sampling_policy.py
  env/
    scene_2d.py
    scene_3d.py
    obstacles.py
    scene_state.py
  benchmark/
    runner.py
    metrics.py
    failure_classifier.py
    tuner.py
  analysis/
    homotopy.py
    report.py
  configs/
    default.yaml
    scenes_2d.yaml
    scenes_3d.yaml
  results/
    records.csv
    summary.json
    reports/
```

### 3.2 技术栈约束
- 语言：Python。
- 数值与几何：`numpy`, `scipy`, `shapely`（2D）, 必要时 3D 几何基础库。
- GPU：`torch`（首版即 CUDA）。
- 日志与数据：`csv` + `json`。
- 运行模式：CLI/headless。

### 3.3 统一数据流
1. 加载场景与配置。
2. 规划器调用统一接口执行。
3. 碰撞检测走统一后端（优先 CUDA）。
4. 结果写入结构化日志。
5. 自动汇总失败模式与指标。
6. 触发规则化调参与同种子回跑。

---

## 第 4 章 算法规范（接口 + 里程碑）

### 4.1 固定对外接口

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class PlannerConfig:
    step_size: float
    max_iters: int
    goal_bias: float
    bridge_bias: float
    cuda_enabled: bool
    alpha_schedule: List[Tuple[float, float]]

@dataclass
class PlannerResult:
    success: bool
    path: List[List[float]]
    time_ms: float
    path_len: float
    collision_checks: int
    seed: int
    scene_id: str

class BasePlanner:
    def plan(self, scene: Any, start: List[float], goal: List[float], config: PlannerConfig) -> PlannerResult:
        raise NotImplementedError

class Sampler:
    def sample(self, state: Dict[str, Any]) -> List[float]:
        raise NotImplementedError

class CollisionBackend:
    def batch_segment_check(self, segments, scene_state) -> "torch.BoolTensor":
        raise NotImplementedError

class BenchmarkRunner:
    def run(self, planners, scenes, seeds):
        """输出 records.csv 与 summary.json"""
```

### 4.2 采样策略统一规范
`Sampler.sample(state)` 必须内建三混合策略：
- `uniform`：均匀采样。
- `goal`：目标偏置采样。
- `bridge`：从桥接候选池采样。

概率由 `goal_bias` 和 `bridge_bias` 控制，剩余概率为 `uniform`。

### 4.3 CCI 规范
- 通过 `alpha_schedule` 管理环境从简化障碍到真实障碍的递进。
- 每次规划迭代从当前进度映射 `alpha`，更新 `scene_state`。
- 碰撞检测必须读取当前 `alpha` 下的环境状态。

### 4.4 里程碑与 DoD

#### M1 基线 + CUDA 通路
- 输入：2D/3D 简单场景配置。
- 任务：实现 `RRT`, `RRT-Connect`, `RRT*`；接入 `CollisionBackend.batch_segment_check` 的 CUDA 版本。
- 输出：3 种基线算法可在 headless 模式落盘结果。
- DoD：同一场景同一种子下可稳定返回结果，日志字段齐全。

#### M2 CCI 接入
- 输入：M1 基线 + CCI 参数配置。
- 任务：实现 `core/cci.py`，让改进规划器按 `alpha_schedule` 递进。
- 输出：`cci_bridge_rrt` 可在 alpha 递进环境中完成规划。
- DoD：日志中可见 alpha 演化轨迹；复杂窄通道成功率较 M1 有提升趋势。

#### M3 改进桥接采样接入
- 输入：M2 版本。
- 任务：实现候选桥接点生成、过滤、聚类代表点，并融入统一 `sample()`。
- 输出：`bridge_bias` 可控，桥接采样使用率可记录。
- DoD：窄通道场景成功率提升，且碰撞卡死比例下降。

#### M4 四类场景批量评测
- 输入：2D 简单、2D 复杂、3D 简单、3D 复杂场景集。
- 任务：批量跑多算法多种子，对比统计。
- 输出：`records.csv`, `summary.json`。
- DoD：无头批处理不中断，统计字段完整，报告自动生成。

#### M5 数据回代迭代（至少两轮）
- 输入：M4 统计结果。
- 任务：失败分类、规则化调参、同种子重跑，重复至少两轮。
- 输出：每轮变更记录、前后对照指标、最终推荐参数。
- DoD：每轮均有明确“改了什么、为什么、指标变了多少”。

---

## 第 5 章 场景与数据规范

### 5.1 场景定义
- 2D 简单：少量矩形/圆形障碍，通路明显。
- 2D 复杂：多障碍形成狭窄通道、瓶颈和死胡同。
- 3D 简单：少量立方体/球体障碍，通路宽。
- 3D 复杂：多障碍与孔洞结构，存在细长通道。

### 5.2 场景配置字段
每个场景至少包含：
- `scene_id`
- `dimension`（2/3）
- `bounds`
- `obstacles`
- `start`
- `goal`
- `difficulty`（simple/complex）

### 5.3 日志字段（统一写死）
`records.csv` 必须包含：
- `run_id`, `seed`, `scene_id`, `difficulty`, `planner`
- `success`, `time_ms`, `path_len`, `collision_checks`
- `iters_used`, `goal_reached_iter`
- `cuda_enabled`, `device_name`
- `alpha_final`, `bridge_usage_ratio`
- `failure_type`（如失败）

`summary.json` 必须包含：
- 各算法分场景成功率、中位时间、中位路径长度。
- 各失败类型占比。
- 推荐参数与版本标签。

---

## 第 6 章 实验协议（Headless）

### 6.1 运行协议
- 所有实验通过 CLI 触发，不依赖图形窗口。
- 每组实验固定种子列表（例如 `[0,1,2,3,4,5,6,7,8,9]`）。
- 对每个算法和场景执行同等预算（`max_iters` 与超时阈值一致）。

### 6.2 测试计划

#### 单元测试
- CCI 插值 SDF 的数值稳定性与单调性。
- 桥接采样候选点与聚类代表点有效性。
- CUDA/CPU 一致性抽检（相同输入，输出近似一致）。

#### 集成测试
- 2D 简单：所有规划器可完成规划并落盘。
- 2D 复杂：改进算法成功率显著高于基础 RRT。
- 3D 简单与复杂：批量运行稳定，无中断。

### 6.3 默认验收阈值
- 复杂场景中改进算法成功率相对基础 RRT 提升 >= 15%。
- 中位规划时间不劣于 RRT* 超过 30%。
- 固定种子重复运行的波动在可接受区间。

---

## 第 7 章 迭代闭环（失败分类 -> 调参 -> 回跑）

### 7.1 失败分类规则
- `timeout`：达到上限迭代或时间未找到路径。
- `collision_stuck`：扩展多次因碰撞失败，树增长停滞。
- `overlong_path`：找到路径但长度远超基线中位值阈值。
- `unstable`：同参数同场景多种子波动过大。

### 7.2 规则化调参映射
- `collision_stuck` 偏高：增大 `bridge_bias`，减小 `step_size`。
- `timeout` 偏高：提高 `goal_bias`，优化 `alpha_schedule` 前段递进速度。
- `overlong_path` 偏高：增加重连优化频率，适度降低 `step_size`。
- `unstable` 偏高：缩窄关键参数搜索区间，增加种子数做稳健评估。

### 7.3 回代流程（固定）
1. 读取上一轮 `records.csv`。
2. 自动打标签 `failure_type`。
3. 生成候选参数组。
4. 同种子重跑对照实验。
5. 输出本轮改动与指标变化。
6. 至少重复两轮。

### 7.4 每轮输出模板
- 变更：参数/模块/策略。
- 原因：对应失败分类统计。
- 结果：成功率、时间、路径长度、失败占比变化。
- 结论：保留/回退/继续探索。

---

## 第 8 章 AI 执行清单（可直接分解任务）

### 8.1 执行顺序
1. 建立工程骨架与统一接口类型。
2. 完成 M1（基线 + CUDA 碰撞后端）。
3. 接入 M2（CCI alpha 递进）。
4. 接入 M3（改进桥接采样）。
5. 完成 M4（四类场景批量评测）。
6. 完成 M5（至少两轮回代优化）。
7. 输出最终报告（含推荐参数与限制说明）。

### 8.2 任务卡片模板（AI 每次必须填）
- 任务名：
- 输入：
- 输出：
- 修改模块：
- 验证方式：
- DoD：

### 8.3 强制工程约束
- 不允许跳过日志字段。
- 不允许手工挑选性展示结果，必须全量落盘。
- 不允许跨算法使用不同预算进行“非公平比较”。
- 任何调参必须带同种子前后对照。

### 8.4 最终交付物
- `records.csv`
- `summary.json`
- 回代迭代记录（至少两轮）
- 项目结论文档（建议下一步优化方向）

---

## 附录 A：默认参数建议（起步值）

```yaml
planner:
  step_size: 0.08
  max_iters: 12000
  goal_bias: 0.08
  bridge_bias: 0.20
  cuda_enabled: true
  alpha_schedule:
    - [0.0, 0.0]
    - [0.3, 0.2]
    - [0.6, 0.5]
    - [1.0, 1.0]

benchmark:
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  timeout_sec: 8
```

## 附录 B：风险与边界
- 首版即 CUDA 可能导致调试复杂度上升，需保留 CPU 对照路径用于抽检。
- CCI 参数对场景尺度敏感，必须与障碍尺度共同调优。
- 桥接采样在部分稀疏场景可能收益有限，应由统计结果决定是否保留高权重。

