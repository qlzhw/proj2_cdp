# CDP 复现排查关键信息

## 任务背景

- 目标任务：`adroit_pen`
- 当前复现数据目录：`Causal-Diffusion-Policy/data/adroit_pen_expert.zarr`
- 当前训练输出目录：`Causal-Diffusion-Policy/data/outputs`
- 关注问题：为什么当前复现结果与 CDP 论文表格结果差距较大

## 已确认事实

### 1. 当前评测协议已经基本对齐 CDP 论文公开定义

- CDP 论文附录 A 明确写到：
  - Adroit / DexArt / MetaWorld 使用 3 个 seed：`0, 1, 2`
  - 每 `200` 个 training epochs 评测一次
  - 每次评测 `20` 个 episodes
  - 对每个 seed 取 top-5 success rates 的均值
  - 最后汇总 3 个 seed 的 mean ± std
- 原始 DP3 论文也明确使用：
  - 每个任务 `10 demonstrations`
  - 每 `200` epoch 评测 `20` episodes
  - 取最高 `5` 个 success rates 平均

### 2. 当前 Adroit 数据集确实只有 10 个 demonstrations

- `adroit_door_expert.zarr`：10 个 episodes
- `adroit_hammer_expert.zarr`：10 个 episodes
- `adroit_pen_expert.zarr`：10 个 episodes
- 这与 CDP / 原始 DP3 仓库默认 Adroit demo 生成脚本一致
- 也就是说，你当前不是“训错了任务”，而是确实在用官方默认风格的 10-demo 设定

### 3. `test_mean_score` 不是标准 Adroit benchmark success

- 标准 Adroit success 定义在底层 `mj_envs` / `VRL3` 中：
  - `pen`：整条 episode 中 `goal_achieved` 累积次数 `> 20`
  - `door/hammer`：整条 episode 中 `goal_achieved` 累积次数 `> 25`
- 但 CDP 仓库与原始 3D-Diffusion-Policy 仓库的 Adroit runner 都没有直接按这个标准统计主指标
- 它们实际使用的是 runner 内部的 `test_mean_score`
- 因此：
  - 你的评测方式与 CDP / DP3 公开实现口径是一致的
  - 但这个口径本身不是标准 Adroit episode success

### 4. CDP 基本沿用了 DP3 的 demo 数量与评测节奏

- demo 数量：Adroit 默认 `10` 条 expert demonstrations
- 评测轮数：每次 `20` 个 episodes
- 统计方式：每 `200` epochs 评测，seed 内 top-5，seed 间 mean ± std
- 因此，当前结果与论文差异，已经不能简单归因于“你评测协议搞错了”

## 重新统计后的结论

### 1. 之前粗统计存在一个问题

- 之前曾直接依据 checkpoint 文件名做统计
- 但 CDP 仓库中 `cdp3` 默认 `rollout_every=50`
- 论文口径要求只看每 `200` epoch 的评测点
- 因此需要从 `logs.json.txt` 中筛选 `epoch % 200 == 0` 的记录重新计算

### 2. 按论文口径重算后的 `adroit_pen` 结果

| Algorithm | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|---|---:|---:|---:|---:|
| cdp2 | 0.6316 | 0.5134 | 0.4997 | 0.5482 ± 0.0592 |
| cdp3 | 0.6359 | 0.4670 | 0.5775 | 0.5602 ± 0.0700 |
| dp2 | 0.0729 | 0.1500 | 0.1628 | 0.1286 ± 0.0397 |
| dp3 | 0.5572 | 0.5541 | 0.5541 | 0.5551 ± 0.0015 |

### 3. 与 CDP 论文表格对比后的观察

- 按当前公开协议重算后，结果仍然与论文数值不一致
- 且偏差不是单向的：
  - 有的方法比论文低
  - 有的方法比论文高
- 这更像是：
  - 论文使用的数据快照 / 代码快照 / 依赖环境与当前公开版不完全一致
  - 或者论文当时使用的 10 条 demonstrations 与当前重新生成的 10 条 demonstrations 不是同一批

## 当前最合理的判断

### 可以明确说的

- 你的复现流程在 demo 数量、评测节奏、top-5 汇总规则上，已经基本与 CDP 论文公开说明一致
- 你当前的评测也基本与 CDP / DP3 公开代码实现保持一致
- 因此，当前结果与论文不一致，不能再简单归因于“你没有按论文评测”

### 需要谨慎表述的

- 不能仅凭当前证据直接下结论说作者“造假”
- 但可以明确指出：
  - 公开代码与论文数值之间存在明显复现落差
  - 公开实现使用的主指标不是标准 Adroit benchmark success
  - 公开仓库的复现链条存在较强的不透明性

## 当前最可能的误差来源

1. 论文使用的 demonstrations 与当前重新生成的 demonstrations 不是同一批  
2. 论文使用的代码快照与公开仓库当前版本不完全一致  
3. 依赖环境差异导致 10-demo few-shot 设定下波动被显著放大  
4. 论文表格数值与公开仓库中的 runner 指标之间存在未明确说明的偏差

## 后续建议

### 如果目标是继续做严谨排查

- 逐项比对原始 DP3 baseline 与当前 CDP 仓库中的 DP3 baseline 差异
- 为 Adroit 单独实现“标准 success rate”统计，和当前 `test_mean_score` 并排报告
- 固定 demonstrations 文件，不再动态重新生成，避免数据集漂移
- 整理“论文表 1 不可复现证据链”，区分：
  - 已对齐项
  - 未公开项
  - 高风险不一致项

### 如果目标是对外汇报

- 推荐使用下面这句话：

> 当前复现实验已基本对齐 CDP 论文公开说明的 demo 数量、评测轮数与汇总协议，但结果仍与论文表格存在明显差异；结合公开代码分析，差异更可能来自 demonstrations / 代码快照 / 依赖环境未完全公开一致，而不能简单归因于复现者未按论文评测。
