# CCI-Bridge RRT：论文风格技术报告（中文）

## 题目
**面向狭窄通道规划的 CCI-Bridge 增强 RRT：一个支持 CUDA 与消融实验的工程化研究实现**

## 摘要
狭窄通道路径规划长期受限于可行域测度低、碰撞检测压力高和随机采样命中率差等问题。本文围绕参考论文中的碰撞约束插值（CCI）与改进桥接采样思想，对现有 RRT 框架进行了中到重度工程重构，并实现了可复现的消融实验体系。我们引入了：（1）带塑形函数的 CCI 插值与 alpha 演化追踪；（2）支持 DBSCAN 的桥接候选聚类与诊断统计；（3）`baseline/cci_only/bridge_only/cci_bridge` 四种消融模式；（4）自动化大规模实验、图像导出与报告渲染链路。在 4 场景 × 4 算法 × 4 消融 × 20 seeds 的全矩阵实验（1280 次）及两轮自动回代（80 次）中，`cci_bridge_rrt::cci_only` 成功率达到 1.000，`cci_bridge_rrt::cci_bridge` 成功率达到 0.994。本文给出了核心公式、工程设计、统计结果与关键可视化证据，为后续论文撰写与代码发布提供了完整基础。

**关键词**：RRT；狭窄通道；碰撞约束插值；桥接采样；DBSCAN；CUDA；消融实验

---

## 1. 引言
狭窄通道问题是采样式规划器（RRT 家族）最典型的困难场景。常规均匀采样难以稳定命中瓶颈区域，导致树扩展效率低、失败率高、时间波动大。针对这一问题，本次工作并非局部补丁，而是面向“可审阅、可复现、可展示”的研究工程化改造，目标包括：

1. 将论文关键思想转化为可执行模块；
2. 将性能对比从单次样例提升为全矩阵统计；
3. 将结果从数字表格扩展为路径图与统计图双证据；
4. 形成可直接用于 GitHub 展示与论文素材整理的文档资产。

---

## 2. 方法

### 2.1 CCI 数学与工程实现
参考 CCI 思想，引入塑形函数与插值机制：

\[
f(x)=\frac{e^{\eta x}-1}{\eta}
\]

\[
\mathrm{SDF}_{\alpha}=(1-\alpha)f(\mathrm{SDF}_a)+\alpha f(\mathrm{SDF}_b),\quad \alpha\in[0,1]
\]

工程侧新增：
- `shaping_function`（指数截断防溢出）
- `interpolate_sdf`
- `AlphaScheduler/AlphaExecutor`（alpha 调度与轨迹统计）
- `build_leaf_sequence`（leaf set 序列近似生成）

### 2.2 改进桥接采样
参考改进 Bridge Sampling，采用“中点可行 + 正交扰动单通道筛选（`f==1`）+ 聚类代表点”策略，并支持两种聚类模式：
- `radius`（轻量）
- `dbscan`（标准密度聚类）

新增诊断指标：
- `candidate_density`
- `corner_false_positive_rate`
- `cluster_validity`

### 2.3 消融体系
统一支持四种模式：
- `baseline`
- `cci_only`
- `bridge_only`
- `cci_bridge`

用于量化 CCI 与 Bridge 的独立贡献与组合收益。

---

## 3. 实验设置

### 3.1 实验矩阵
- 场景：4（2D/3D × simple/complex）
- 算法：4（rrt, rrt_connect, rrt_star, cci_bridge_rrt）
- 消融：4（baseline, cci_only, bridge_only, cci_bridge）
- 种子：20（0~19）

总运行次数：
- 矩阵实验：1280
- 回代实验：80（两轮）
- 总记录：1360

### 3.2 运行策略
- CUDA 可用时默认启用；
- 速率限制：40/min（满足 `< 50/min`）；
- 自动回代：根据失败分类进行参数更新并重跑对照。

---

## 4. 结果

### 4.1 统计图（自动由 summary.json 渲染）

#### 图1 成功率对比柱状图
![图1 成功率柱状图](github_images/01_success_rate_bar.png)

#### 图2 中位规划时间对比柱状图
![图2 时间柱状图](github_images/02_median_time_bar.png)

#### 图3 复杂场景运行时间箱线图
![图3 复杂场景时间箱线图](github_images/03_complex_runtime_boxplot.png)

#### 图4 复杂场景路径长度箱线图（成功样本）
![图4 复杂场景路径长度箱线图](github_images/04_complex_pathlen_boxplot.png)

### 4.2 关键定量结论
来自 `summary.json` 的代表性结果：

- `rrt::baseline` 成功率：0.725
- `rrt_connect::baseline` 成功率：0.775
- `rrt_star::baseline` 成功率：0.725
- `cci_bridge_rrt::cci_only` 成功率：1.000
- `cci_bridge_rrt::cci_bridge` 成功率：0.994

失败分布：
- timeout：288
- collision_stuck：6
- unstable：3

---

## 5. 关键路径图（障碍物 + 树节点 + 轨迹）

### 5.1 2D 复杂场景
**Baseline（RRT）**
![图5 2D baseline](github_images/05_2d_rrt_baseline_seed0.png)

**CCI only（CCI-Bridge RRT）**
![图6 2D cci_only](github_images/06_2d_cbrt_cci_only_seed0.png)

**CCI + Bridge（CCI-Bridge RRT）**
![图7 2D cci_bridge](github_images/07_2d_cbrt_cci_bridge_seed0.png)

### 5.2 3D 复杂场景
**Baseline（RRT*）**
![图8 3D baseline](github_images/08_3d_rrtstar_baseline_seed0.png)

**CCI only（CCI-Bridge RRT）**
![图9 3D cci_only](github_images/09_3d_cbrt_cci_only_seed0.png)

**CCI + Bridge（CCI-Bridge RRT）**
![图10 3D cci_bridge](github_images/10_3d_cbrt_cci_bridge_seed0.png)

### 5.3 全局拼图
![图11 全局拼图](github_images/11_collage_overview.png)

---

## 6. CUDA 分析

### 6.1 优势
- 批量碰撞检测可并行化，适合矩阵实验；
- 与 CPU 共用接口，便于一致性对照与回归；
- 在大规模任务下具备更好吞吐。

### 6.2 难点
- 小批量任务中，GPU 启动和数据搬运开销可能抵消收益；
- CPU/GPU 浮点边界行为存在差异；
- 长任务需要速率控制与 checkpoint 管理。

### 6.3 工程对策
- 记录 `cpu_gpu_consistency` 字段进行抽检；
- 启用限速、重试与阶段状态持久化；
- 保留 CPU 回退路径。

---

## 7. 讨论
本次改造的核心价值不只是“性能更高”，更重要的是将算法改进、工程结构、实验流程、可视化证据打通：

1. 方法可解释：CCI 与 Bridge 都有可观测统计；
2. 实验可复现：矩阵、回代、速率、日志、图像都可追溯；
3. 报告可复用：统计图与关键路径图可直接用于论文草稿与仓库展示。

当前结果表明，CCI 驱动对复杂场景收益显著；Bridge 的收益受参数与场景拓扑影响，后续可继续优化候选质量与聚类稳健性。

---

## 8. 结论
本文完成了一个面向狭窄通道规划的研究工程化升级，实现了：
- 论文思想到代码模块的可验证映射；
- 高强度消融实验与自动回代闭环；
- 面向 GitHub 与论文写作的图文一体化结果资产。

该版本已具备继续开展更严格对比实验、扩展场景复杂度和撰写正式论文的基础条件。

---

## 9. 参考文献（项目本地）
- `RRT算法参考论文/2410.20697v1.pdf`
- `RRT算法参考论文/sensors-26-01582.pdf`
- `RRT算法参考论文/planwithhomotopyconstraints_aaai10.pdf`
- `RRT算法参考论文/2603.26696v1.pdf`
- `RRT算法参考论文/kinetic-triang.pdf`
- `RRT算法参考论文/2010.08167v1.pdf`
