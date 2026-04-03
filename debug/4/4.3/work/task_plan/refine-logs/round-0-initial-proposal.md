# Research Proposal: History-Utilized Causal Diffusion Policy (HU-CDP)

## Problem Anchor

- **Bottom-line problem**: CDP (Causal Diffusion Policy) 使用固定20步历史动作窗口作为因果条件，无法利用更长期的历史信息来改善任务成功率，尤其在long-horizon灵巧操作任务中。
- **Must-solve bottleneck**: (1) 固定窗口丢失长程历史信息；(2) 扩散策略存在"反向copycat问题"——即使给予更长上下文，模型也不会主动利用历史信息（LDP论文实证：无PTP的扩散策略在长上下文下崩溃至0%成功率）。
- **Non-goals**: (a) 不做真实机器人实验（纯仿真验证）；(b) 不做观测端历史建模（只关注动作历史）；(c) 不追求百万级长序列能力（任务最长~300步）；(d) 不改变CDP的核心TEDi因果扩散框架。
- **Constraints**: 单卡GPU训练；4仿真环境（Adroit/DexArt/MetaWorld/RoboFactory）；基于现有CDP代码库改动；需维持dp2/cdp2/dp3/cdp3不变，新增adp2/adp3。
- **Success condition**: 在至少3个long-horizon任务上，ADP的成功率显著超过CDP（≥5%绝对提升），同时推理延迟增加<20%。

## Technical Gap

### 当前方法如何失败

CDP的CausalTransformer在horizon=40的窗口内运行，其中前32步（n_obs_steps）是历史动作条件（带噪声注入），后8步（n_action_steps）是待生成动作。窗口外的所有历史动作被完全丢弃。

**关键证据表明这是真实瓶颈**:
1. CDP论文Limitation明确承认固定窗口限制了long-horizon场景
2. LDP (CoRL 2025 Best Paper) 证明扩散策略存在"反向copycat问题"——模型倾向于忽略历史条件而直接模仿当前观测

### 为什么朴素扩大窗口不够

- **O(T²) attention**: 将窗口从20扩到300步会使注意力计算增长225倍
- **历史低利用**: LDP的核心发现是，即使给更多上下文，扩散策略不会自动利用它——需要显式训练信号强制利用
- **远期动作噪声**: 300步前的动作对当前决策的直接价值极低，全部以原始形式输入会引入噪声

### 最小充分干预

需要同时解决两个问题：(a) **capacity**——高效表示全历史，(b) **utilization**——强制模型利用历史。GRU压缩解决(a)，PTP辅助loss解决(b)。

## Method Thesis

- **One-sentence thesis**: 通过GRU将全部历史动作压缩为固定维度隐状态，经AdaLN注入CausalTransformer，并用Past-Token Prediction辅助loss显式强制历史利用，实现高效且有效的全历史因果扩散策略。
- **Why this is the smallest adequate intervention**: GRU是最简单的序列压缩模型（单层即可处理300步低维动作序列），AdaLN是最轻量的条件注入方式（DiT已广泛验证），PTP是唯一经过验证能解决扩散策略历史低利用问题的方法。
- **Why this route is timely**: LDP (CoRL 2025 Best Paper) 刚刚确立了"扩散策略不利用历史"的问题及PTP解决方案，但LDP工作在观测端且使用全注意力（不可扩展），将其核心洞察迁移到动作端+压缩架构是自然且新颖的组合。

## Contribution Focus

- **Dominant contribution**: GRU全历史压缩 + AdaLN条件注入——首个在因果扩散策略中实现O(1)推理成本的全历史接入机制。
- **Optional supporting contribution**: 将LDP的PTP辅助loss适配到CDP的TEDi框架中，解决动作历史低利用问题。
- **Explicit non-contributions**: (a) 不提出新的序列模型（使用标准GRU）；(b) 不提出新的扩散框架（保持TEDi）；(c) 不做观测端历史建模。

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**: DP3Encoder (point cloud编码)、DDPMTEDiScheduler (扩散调度)、chunked causal mask、KV cache机制均完全保留
- **New trainable components** (2个):
  1. `HistoryGRU`: 单层GRU (input_dim=action_dim, hidden_dim=256) + linear projection → history embedding
  2. `AdaLN modulation layers`: 每个TemporalTransformerBlock新增 (shift, scale) 参数，从history embedding生成
- **Tempting additions intentionally not used**:
  - Mamba/SSM (300步低维动作序列不需要O(n)优势，GRU更简单)
  - 多尺度层级压缩 (2-level已足够，3-level引入超参爆炸)
  - Cross-attention注入 (比AdaLN更重，DiT实验表明AdaLN更好)
  - 可变噪声 (per-token noise引入额外复杂度，边际收益存疑)
  - Memory bank / retrieval (MemoryVLA等已拥挤，差异不足)

### System Overview

```
Episode execution loop:
  ┌─────────────────────────────────────────────────┐
  │  At each step t:                                │
  │                                                 │
  │  1. Observe: o_t → DP3Encoder → obs_features    │
  │                                                 │
  │  2. Update GRU: h_t = GRU(a_{t-1}, h_{t-1})    │
  │     (O(1) per step, no recomputation)           │
  │                                                 │
  │  3. Project: z_t = Linear(h_t)  [256-dim]       │
  │                                                 │
  │  4. CausalTransformer denoise:                  │
  │     - Input: action_buffer [20 steps] + noise   │
  │     - Condition: obs_features (cross-attn)      │
  │     - History: z_t (AdaLN modulation)            │
  │     - KV cache: reuse from previous chunks      │
  │     → predicted clean actions                   │
  │                                                 │
  │  5. Execute first n_action_steps=8 actions      │
  │  6. Push executed actions to buffer + GRU       │
  └─────────────────────────────────────────────────┘
```

### Core Mechanism: GRU History Encoder + AdaLN Conditioning

**Architecture**:

```python
class HistoryGRU(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.gru = nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.hidden = None  # persistent state across steps
    
    def reset(self):
        self.hidden = None
    
    def forward(self, action_chunk):
        """
        Training: action_chunk = (B, T_history, Da), process full history
        Inference: action_chunk = (B, n_action_steps, Da), incremental update
        """
        output, self.hidden = self.gru(action_chunk, self.hidden)
        return self.proj(self.hidden.squeeze(0))  # (B, output_dim)
```

**AdaLN注入** (修改现有TemporalTransformerBlock):

```python
# 在现有的 modulate(x, shift, scale) 基础上
# 每个block新增一组 (shift, scale) 参数从 history_emb 生成

class TemporalTransformerBlock(nn.Module):
    def __init__(self, hidden_size, ...):
        # ... existing code ...
        # NEW: history conditioning via AdaLN
        self.history_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(history_emb_dim, 2 * hidden_size)  # shift + scale
        )
    
    def forward(self, x, t_emb, history_emb=None, ...):
        # existing timestep modulation
        x = modulate(self.t_norm1(x), t_shift, t_scale)
        
        # NEW: history modulation (additive, after timestep)
        if history_emb is not None:
            h_shift, h_scale = self.history_adaLN(history_emb).chunk(2, dim=-1)
            x = modulate(x, h_shift, h_scale)
        
        # ... rest of attention + MLP ...
```

**Input / output**:
- Input: 历史动作序列 a_{0:t-1} (训练时完整序列，推理时增量)
- Output: history embedding z_t ∈ R^256, 注入CausalTransformer每层的AdaLN

**Why this is the main novelty**: 现有工作中，(a) CDP只有固定窗口，无全历史接入；(b) LDP用全注意力处理长上下文（O(T²)，不可扩展），且只做观测端；(c) MTIL用Mamba编码全历史但无扩散过程。GRU+AdaLN是首个在因果扩散策略中以O(1)推理成本接入全历史的方案。

### Supporting Component: Past-Token Prediction (PTP)

**来源**: LDP (CoRL 2025 Best Paper)

**适配到CDP框架**:

```python
# cdp3.py compute_loss 中的修改

def compute_loss(self, batch):
    # ... existing code until pred = self.model(...) ...
    
    pred = self.model(
        noisy_trajectory, diff_steps[:, To:],
        tpe_start=cache_start_idx,
        cond=cond,
        history_emb=history_emb  # NEW: from HistoryGRU
    )
    
    # 原始CDP: loss只在action steps (To:) 上计算
    # loss = F.mse_loss(pred, target, reduction='none')
    # loss = loss * loss_mask.type(loss.dtype)
    
    # PTP扩展: 也预测observation steps (0:To) 位置的动作
    # 用单独的prediction head
    past_pred = self.past_pred_head(pred[:, :To])  # (B, To, Da)
    past_target = trajectory[:, :To]  # 真实历史动作
    
    ptp_loss = F.mse_loss(past_pred, past_target, reduction='mean')
    
    total_loss = bc_loss + self.ptp_weight * ptp_loss  # ptp_weight=0.5
```

**Test-time self-verification** (可选，计算预算对齐版):

```python
# 推理时: 采样N个候选，用past prediction质量选最优
def predict_action_with_verification(self, obs_dict, n_candidates=3):
    candidates = [self.conditional_sample(cond, history_emb) for _ in range(n_candidates)]
    past_actions = self.action_buffer[:, -To:]  # 实际近期历史
    
    scores = []
    for c in candidates:
        past_pred = self.past_pred_head(c[:, :To])
        score = -F.mse_loss(past_pred, past_actions, reduction='mean')
        scores.append(score)
    
    best_idx = torch.argmax(torch.tensor(scores))
    return candidates[best_idx]
```

**注意**: self-verification的计算预算必须与baseline对齐（GPT-5.4审稿意见）。报告时需同时报告N=1（无verification）和N=3的结果，且与CDP使用相同推理步数对比。

### Modern Primitive Usage

- **Which primitive**: 无特定frontier primitive（GRU是经典模型，扩散框架保持TEDi不变）
- **Exact role**: GRU作为history compressor，不是frontier-era component
- **Justification**: 本方法的核心洞察来自LDP (CoRL 2025 Best Paper, PTP机制) 和DiT (AdaLN conditioning)，但方法本身故意保持简单——论文故事是 **simplicity works**，而非堆叠frontier components

### Integration into CDP Codebase

**文件变更清单**:

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `policy/adp3.py` | NEW | 基于cdp3.py，新增HistoryGRU + AdaLN + PTP |
| `policy/adp2.py` | NEW | 基于cdp2.py，同上（图像观测版） |
| `model/diffusion/history_encoder.py` | NEW | HistoryGRU类定义 |
| `model/diffusion/causal_transformer.py` | MODIFIED | TemporalTransformerBlock新增history_adaLN |
| `config/adp3.yaml` | NEW | 基于cdp3.yaml，新增history相关超参 |
| `config/adp2.yaml` | NEW | 基于cdp2.yaml |
| `train.py` | MODIFIED | _POLICY_MAP新增adp2, adp3 |
| `scripts/train_policy.sh` | MODIFIED | 支持adp2, adp3 |

**什么被冻结**: DP3Encoder, DDPMTEDiScheduler, 数据pipeline (ReplayBuffer/SequenceSampler), 环境runner, 评估逻辑。

**什么可训练**: HistoryGRU (1层, ~260K参数), AdaLN modulation layers (每层~130K, 8层共~1M), past_pred_head (~66K)。总新增参数 ~1.3M，相对CausalTransformer原有~8M约16%增量。

### Training Plan

**单阶段联合训练** (不需要LDP的多阶段训练):

LDP需要多阶段训练是因为它用全注意力处理长观测序列（冻结encoder → 缓存embedding → 训练decoder），计算不可承受。我们的GRU是轻量的，可以直接端到端训练。

```
Loss = bc_loss + 0.5 * ptp_loss

Optimizer: AdamW (lr=1e-4, betas=[0.9, 0.95])
  - CausalTransformer + HistoryGRU: weight_decay=1e-3
  - DP3Encoder: weight_decay=1e-6
Schedule: Cosine with 1000 warmup steps
EMA: same as CDP (power=0.75, max_value=0.9999)
Epochs: 3000 (same as CDP)
Batch: 64
```

**训练时GRU输入构造**:

```python
# 从dataset获取完整episode中当前sample之前的所有动作
# SequenceSampler已知episode边界和sample位置
# 训练时一次性输入所有历史动作给GRU (不需要增量)
history_actions = batch['history_actions']  # (B, T_hist, Da), T_hist变长，需padding
history_emb = self.history_gru(history_actions)  # (B, 256)
```

**数据构造**: 需修改`SequenceSampler`，在每个sample中额外返回该sample在episode内之前的所有动作序列。

### Failure Modes and Diagnostics

| Failure Mode | Detection | Mitigation |
|-------------|-----------|------------|
| GRU hidden state退化为常数 (history无效) | 监控h_t的方差随步数的变化 | 增加GRU input dropout; 验证PTP loss是否在下降 |
| PTP loss与BC loss冲突 | PTP loss不收敛或BC loss因PTP上升 | 降低ptp_weight; 检查past_pred_head的梯度流 |
| Exposure bias: 推理时自生成动作累积漂移 | 长horizon任务后半段成功率急剧下降 | 训练时对GRU输入加噪声(沿用CDP的noise injection思路) |
| AdaLN modulation过强覆盖timestep信号 | 扩散过程不收敛 | 初始化AdaLN参数为零(zero-init，DiT标准做法) |
| 推理延迟超预算 | Wall-clock timing | GRU O(1)更新应<1ms/step; 若仍慢，检查是否重复计算 |

### Novelty and Elegance Argument

**最接近的工作及精确差异**:

| 工作 | 与本方法的区别 |
|------|-------------|
| CDP (CoRL 2025) | 固定20步窗口，无全历史接入，无PTP |
| LDP (CoRL 2025 Best Paper) | 观测端全注意力O(T²)，无因果扩散，无压缩 |
| MTIL (RA-L 2025) | Mamba编码全历史但无扩散过程，是纯BC |
| Hiveformer (CoRL 2022) | Keypose级全历史+全注意力，O(T²)，非扩散策略 |

**为什么这是focused mechanism-level contribution而非module pile-up**:

核心insight只有一个：**扩散策略不利用历史 (LDP发现) + 压缩全历史的最简方案是GRU+AdaLN (本文主张)**。PTP不是新贡献而是borrowed technique，GRU不是新模型而是implementation choice。论文故事是证明 **simplicity works**——复杂方案（Mamba、多尺度、检索）都不必要。

## Claim-Driven Validation Sketch

### Claim 1: 全历史接入提升long-horizon任务成功率

- **Minimal experiment**: ADP(GRU+AdaLN+PTP) vs CDP, 在7-8个任务上3 seeds对比
- **Baselines / ablations**: CDP (20步窗口), DP3 (无历史), CDP-L100 (朴素扩窗)
- **Metric**: 成功率 ± std
- **Expected evidence**: ADP在long-horizon任务(DexArt laptop/toilet, Adroit pen/hammer)上≥5%绝对提升

### Claim 2: 简单压缩+显式利用已足够 (Simplicity Defense)

- **Minimal experiment**: ADP vs ADP-w/o-GRU (只有PTP) vs ADP-w/o-PTP (只有GRU) vs ADP-Mamba (替换GRU为Mamba)
- **Baselines / ablations**: 上述4种变体
- **Metric**: 成功率 + 推理延迟
- **Expected evidence**: (a) GRU和PTP各自有独立贡献; (b) Mamba不显著优于GRU; (c) 推理延迟增加<20%

### Claim 3 (Anti-claim): 性能提升不只是来自更多参数或phase clock

- **Minimal experiment**: History intervention tests — truncate/shuffle/cross-episode replace历史动作
- **Baselines / ablations**: ADP在正常历史 vs 被破坏的历史
- **Metric**: 成功率差异
- **Expected evidence**: 打乱或替换历史会显著降低成功率，证明模型确实利用了历史语义而非仅作为phase clock

## Experiment Handoff Inputs

- **Must-prove claims**: (1) 全历史提升成功率; (2) 简单方案已足够; (3) 模型真正利用历史语义
- **Must-run ablations**: GRU消融, PTP消融, History encoder对比(GRU/Mamba/LSTM), 条件注入对比(AdaLN/CrossAttn/FiLM), History intervention
- **Critical datasets / metrics**: Adroit (hammer/pen/door), DexArt (laptop/toilet), MetaWorld (box-close/disassemble); 成功率, 推理延迟, GPU显存
- **Highest-risk assumptions**: (1) CDP固定窗口确实是瓶颈（需实验0诊断验证）; (2) 300步低维动作序列中GRU不会退化为常数; (3) PTP在TEDi框架下的适配不会与chunk-wise denoising冲突

## Compute & Timeline Estimate

- **Estimated GPU-hours**: 实验0诊断(5种窗口 × 3任务 × 1 seed) ~15h; 主实验(8任务 × 3 seeds) ~72h; 消融(5种变体 × 3任务 × 3 seeds) ~135h; 总计 ~220 GPU-hours (单卡A100约9天)
- **Data / annotation cost**: 零（使用现有zarr数据集）
- **Timeline**: Phase 1诊断(2天) → Phase 2开发(5天) → Phase 3验证(7天) → Phase 4消融+完善(5天) ≈ 3周
