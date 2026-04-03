# Round 2 Review (GPT-5.4)

## Scores

| 维度 | R1 | R2 | 变化 |
|------|----|----|------|
| Problem Fidelity | 8 | 6 | -2 (DRIFT detected) |
| Method Specificity | 5 | 6 | +1 |
| Contribution Quality | 6 | 7 | +1 |
| Frontier Leverage | 7 | 8 | +1 |
| Feasibility | 7 | 8 | +1 |
| Validation Focus | 5 | 8 | +3 |
| Venue Readiness | 6 | 7 | +1 |
| **OVERALL** | **6.3** | **6.9** | **+0.6** |

**Verdict: REVISE**

## Critical Issues

### 1. Problem Anchor DRIFTED (CRITICAL)
- 从"CDP固定动作窗修复"漂移到"通用长上下文扩散策略"
- 加入obs历史+放开TEDi约束 → paper身份不清
- **必须二选一**: (a) 锁回CDP/TEDi extension; (b) 承认drift，改写为generic long-context diffusion policy

### 2. Remote-PTP机制不严格 (CRITICAL)
- 不告诉模型预测哪个remote位置 → 任务欠定
- 固定位置 → 退化为phase clock
- **修正**: 改为query-conditioned目标，用相对偏移Δ∈{48,96,144}作查询，`head([z_hist, q(Δ)]) → remote action snippet`

### 3. 序列化契约不够精确 (CRITICAL)
- prefix截止时刻、obs/action配对、online update顺序必须明确定义

### 4. Dual-history但只监督action (IMPORTANT)
- Remote-PTP只预测远端动作，obs历史利用只是indirect
- 选项: (a) 加轻量obs target; (b) 收缩claim为"history memory improves policy"

## Simplification Opportunities
1. 去掉"any diffusion framework" → 锁定backbone
2. 只做adp3
3. 主表ADP-act-only和ADP-obs-only留一个，另一个降为消融

## Modernization Opportunities
NONE — 当前配置已足够modern

## Drift Warning
DRIFTED — 从CDP动作历史窗修复 → 通用长上下文扩散策略
