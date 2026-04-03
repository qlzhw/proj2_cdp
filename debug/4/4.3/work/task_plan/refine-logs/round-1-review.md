# Round 1 Review (GPT-5.4)

## Scores

| 维度 | 分数 |
|------|------|
| Problem Fidelity | 8/10 |
| Method Specificity | 5/10 |
| Contribution Quality | 6/10 |
| Frontier Leverage | 7/10 |
| Feasibility | 7/10 |
| Validation Focus | 5/10 |
| Venue Readiness | 6/10 |
| **OVERALL** | **6.3/10** |

**Verdict: REVISE**

## Critical Issues

### 1. Method Specificity (CRITICAL)
- HistoryGRU的`self.hidden`接口在batch训练、打乱采样、并行dataloader中会出错
- PTP只预测local window内的past tokens，不是"约束窗外历史利用"——模型可以靠局部noisy tokens重建，不需要用GRU的全历史信息
- 训练/推理的历史定义没对齐

**修正**: 
- 改为无状态训练接口 + 显式状态推理接口
- PTP必须预测local window之外的**远端历史(remote past)**，只允许通过z_hist去预测
- 明确定义每个sample的chunk start和prefix边界

### 2. Contribution Quality (CRITICAL)
- PTP不能说是optional——如果reverse copycat是关键瓶颈，PTP就是核心一半
- self-verification会把论文从"memory utilization"带偏到"candidate reranking"
- "first O(1) full-history access"说法容易被打

**修正**:
- 统一主贡献为: compressed full-history memory + explicit extra-window supervision + constant-cost TEDi integration
- 删除self-verification（最多appendix）
- PTP是核心不是配角

### 3. Validation Focus (IMPORTANT)
- 实验计划与"smallest adequate mechanism"哲学对冲
- GRU/Mamba/LSTM、AdaLN/CrossAttn/FiLM对比太散

**修正**: 主文只保留最小闭环5个系统对比 + 3个长时程任务 + 1个短时程控制任务

### 4. Venue Readiness (IMPORTANT)
- 容易被看成标准RNN memory + 标准diffusion conditioning + borrowed auxiliary loss的拼装
- 过度借用LDP论证，需要自己的gating diagnostic

## Simplification Opportunities
1. 删除test-time self-verification
2. 先只做adp3
3. 复用现有预测头

## Modernization Opportunities
1. 如果单GRU hidden饱和，考虑固定数量memory tokens (4-8个)
2. 用AdaLN-Zero / gated modulation

## Drift Warning
NONE (除非把self-verification放入主方法)

<details>
<summary>Full Raw Response</summary>

[GPT-5.4 full response saved - see REFINE_STATE for threadId: 019d5152-759a-75c0-a84c-cdd6456d17f0]

</details>
