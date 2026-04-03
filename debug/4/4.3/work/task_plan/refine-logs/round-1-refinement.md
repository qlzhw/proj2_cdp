# Round 1 Refinement

## Problem Anchor
[Verbatim from round 0]

- **Bottom-line problem**: CDP使用固定20步历史动作窗口作为因果条件，无法利用更长期的历史信息来改善任务成功率，尤其在long-horizon灵巧操作任务中。
- **Must-solve bottleneck**: (1) 固定窗口丢失长程历史信息；(2) 扩散策略存在"反向copycat问题"——即使给予更长上下文，模型也不会主动利用历史信息。
- **Non-goals**: (a) 不做真实机器人实验；(b) 不做观测端历史建模；(c) 不追求百万级长序列；(d) 不改变TEDi核心框架。
- **Constraints**: 单卡GPU训练；4仿真环境；基于现有CDP代码库；新增adp3（adp2降为后续工作）。
- **Success condition**: 在至少3个long-horizon任务上，ADP成功率显著超过CDP（≥5%绝对提升），推理延迟增加<20%。

## Anchor Check

- **Original bottleneck**: 固定窗口历史 + 扩散策略不利用历史
- **Why the revised method still addresses it**: 修订后方法同时攻击两个瓶颈——GRU压缩提供容量，Remote-PTP提供利用监督，且Remote-PTP现在**只约束窗外历史**，直接对应锚点问题
- **Reviewer suggestions rejected as drift**: Self-verification被删除（会将论文从"历史利用"带偏到"候选重排"）

## Simplicity Check

- **Dominant contribution after revision**: 压缩动作历史记忆 (Compressed Action-History Memory) + 显式窗外历史利用监督 (Explicit Extra-Window History-Use Supervision)——统一为**一个主线**
- **Components removed or merged**: 
  - 删除: test-time self-verification (从主文中移除)
  - 删除: adp2 (降为后续工作，先只做adp3)
  - 删除: past_pred_head (复用现有action prediction head的linear层)
  - 删除: 主文中GRU/Mamba/LSTM、AdaLN/CrossAttn/FiLM的模块对比(降为appendix)
- **Reviewer suggestions rejected as unnecessary complexity**: Memory tokens (4-8个)——在单GRU hidden未证明饱和前，不引入额外复杂度
- **Why the remaining mechanism is still the smallest adequate route**: 只有两个新组件：HistoryGRU和AdaLN-Zero modulation。Remote-PTP不增加新模块，只改loss目标。

## Changes Made

### 1. HistoryGRU接口重写 (CRITICAL fix)
- **Reviewer said**: self.hidden在batch训练中会出错，训练/推理不对齐
- **Action**: 改为无状态训练接口 + 显式状态推理接口
- **Reasoning**: 训练时每个sample独立，不共享hidden state；推理时通过显式state参数在线更新
- **Impact**: 接口更干净，训练正确性有保证

### 2. PTP改为Remote-PTP (CRITICAL fix)
- **Reviewer said**: 预测local window内的past tokens不约束窗外历史，模型可以靠局部信息重建
- **Action**: PTP目标改为预测local window之外的远端历史片段，只允许通过z_hist预测
- **Reasoning**: 这样auxiliary loss直接把监督绑到"超窗历史记忆"上，模型必须通过GRU记忆远端信息
- **Impact**: 核心机制更锋利——Remote-PTP现在是论文的关键创新点之一

### 3. Contribution主线统一 (CRITICAL fix)
- **Reviewer said**: PTP不能是optional，它是核心一半；self-verification导致drift
- **Action**: 统一主贡献为"Compressed Action-History Memory + Extra-Window Supervision"；PTP升为核心；self-verification删除
- **Reasoning**: 如果reverse copycat是关键瓶颈，那解决它的PTP就不能是配角
- **Impact**: 论文叙事更清晰——一个问题(历史不被利用)，一个解法(压缩记忆+远端监督)

### 4. AdaLN升级为AdaLN-Zero (IMPORTANT fix)
- **Reviewer said**: 用AdaLN-Zero / gated modulation更稳定
- **Action**: 采用DiT标准的zero-initialized gated modulation
- **Reasoning**: 训练初期history conditioning不干扰已收敛的timestep信号
- **Impact**: 训练更稳定，符合现代实践

### 5. 实验压缩 (IMPORTANT fix)
- **Reviewer said**: 实验太散，与smallest adequate mechanism哲学对冲
- **Action**: 主文压缩为5个系统 × 4个任务(3 long + 1 short) + 1组history intervention
- **Reasoning**: 只验证核心claim，不做模块动物园
- **Impact**: 实验更聚焦，审稿人不会怀疑作者不知道核心claim是什么

## Revised Proposal

# Research Proposal: Compressed History-Aware Causal Diffusion Policy

## Problem Anchor
[Same as above — verbatim]

## Technical Gap

### How Current Methods Fail

CDP的CausalTransformer在horizon=40的窗口内运行，窗口外的所有历史动作被完全丢弃。两个独立问题使得简单扩窗无法解决：

1. **Capacity问题**: O(T²)注意力使300步全历史不可承受
2. **Utilization问题**: LDP (CoRL 2025 Best Paper) 证明扩散策略即使给更多上下文也不会主动利用——需要显式训练信号

### Smallest Adequate Intervention

两个问题需要两个对应的最小干预：
- **Capacity**: GRU将全历史压缩为固定256维向量 → O(1)推理成本
- **Utilization**: Remote Past-Token Prediction只监督窗外历史信息 → 强制GRU记忆有语义的远端信息

## Method Thesis

- **One-sentence thesis**: 通过GRU压缩全部历史动作为固定维度记忆状态，经AdaLN-Zero注入CausalTransformer，并用Remote Past-Token Prediction监督窗外历史利用，使因果扩散策略在长时程任务中真正利用超窗历史。

## Contribution Focus

- **Dominant contribution (unified)**: 压缩动作历史记忆 + 显式窗外历史利用监督——首个让因果扩散策略以常数推理成本利用全部历史动作的方法
- **Explicit non-contributions**: (a) 不提出新序列模型；(b) 不改变TEDi扩散框架；(c) 不做观测端；(d) 不做test-time reranking

## Proposed Method

### Complexity Budget

- **Frozen / reused**: DP3Encoder, DDPMTEDiScheduler, chunked causal mask, KV cache, action buffer lifecycle
- **New trainable components** (2):
  1. `HistoryGRU`: 单层GRU + linear projection (无状态训练接口)
  2. `AdaLN-Zero modulation`: 每个TemporalTransformerBlock新增gated (shift, scale, gate) 参数
- **Intentionally not used**: Mamba/SSM, 多尺度压缩, Cross-attention, Memory bank, Self-verification, 可变噪声

### System Overview

```
Training:
  For each sample in batch (independent, no shared state):
  1. prefix_actions [0 : chunk_start]  →  HistoryGRU.encode_prefix()  →  z_hist
  2. local_window [chunk_start : chunk_start+40]  →  existing CDP pipeline
  3. CausalTransformer(local_window, obs_features, diff_steps, z_hist via AdaLN-Zero)
  4. Loss = bc_loss + λ * remote_ptp_loss

Inference (online):
  At each chunk step t:
  1. z_hist, state = HistoryGRU.step(executed_actions, state)  # O(1)
  2. obs_features = DP3Encoder(o_t)
  3. CausalTransformer denoise with KV cache + z_hist AdaLN-Zero
  4. Execute n_action_steps=8 actions, push to buffer + GRU
```

### Core Mechanism 1: HistoryGRU (Stateless Training + Online Inference)

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
    
    def encode_prefix(self, prefix_actions, prefix_len):
        """
        Training: encode variable-length prefix, no internal state.
        Args:
            prefix_actions: (B, T_max, Da) — padded prefix
            prefix_len: (B,) — actual lengths
        Returns:
            z_hist: (B, output_dim)
        """
        # Pack for variable-length efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            prefix_actions, prefix_len.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)  # h_n: (1, B, hidden_dim)
        return self.proj(h_n.squeeze(0))  # (B, output_dim)
    
    def step(self, action_chunk, state):
        """
        Inference: incremental update, explicit state.
        Args:
            action_chunk: (B, n_action_steps, Da)
            state: (1, B, hidden_dim) or None
        Returns:
            z_hist: (B, output_dim)
            new_state: (1, B, hidden_dim)
        """
        _, new_state = self.gru(action_chunk, state)
        z_hist = self.proj(new_state.squeeze(0))
        return z_hist, new_state
```

**Key design decisions**:
- 训练时每个sample独立encode_prefix，无跨sample状态泄漏
- 推理时通过显式state参数在线O(1)更新
- pack_padded_sequence处理变长历史，无信息泄漏
- 单层GRU: 300步×7-20维动作序列不需要更深模型

### Core Mechanism 2: AdaLN-Zero History Conditioning

```python
class TemporalTransformerBlock(nn.Module):
    def __init__(self, hidden_size, ..., history_emb_dim=256):
        # ... existing code ...
        # NEW: history conditioning via AdaLN-Zero (gated)
        self.history_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(history_emb_dim, 3 * hidden_size)  # shift, scale, gate
        )
        # Zero-initialize the linear layer
        nn.init.zeros_(self.history_adaLN[-1].weight)
        nn.init.zeros_(self.history_adaLN[-1].bias)
    
    def forward(self, x, t_emb, history_emb=None, ...):
        # Step 1: existing timestep modulation
        shift_t, scale_t, gate_t = self.t_adaLN(t_emb).chunk(3, dim=-1)
        h = modulate(self.t_norm1(x), shift_t, scale_t)
        
        # Step 2: attention (existing)
        h = self.t_attn1(h, ...)
        
        # Step 3: history gated modulation (NEW)
        if history_emb is not None:
            h_shift, h_scale, h_gate = self.history_adaLN(history_emb).chunk(3, dim=-1)
            h = h + gate(modulate(self.h_norm(h), h_shift, h_scale), h_gate)
        
        # Step 4: gate from timestep (existing)
        x = x + gate(h, gate_t)
        
        # ... MLP block (existing) ...
```

**Zero-init guarantee**: 训练开始时h_gate=0，history conditioning完全不影响已有CDP行为，等价于warm-start from CDP。

### Core Mechanism 3: Remote Past-Token Prediction (Remote-PTP)

**关键创新**: 不同于LDP的PTP预测local window内的past tokens，Remote-PTP**只预测local window之外的远端历史**。这强制模型必须通过z_hist（GRU压缩的全历史记忆）才能完成预测，从而保证GRU确实编码了有语义的历史信息。

```python
def compute_loss(self, batch):
    # ... existing CDP loss computation ...
    
    # 获取history embedding
    z_hist = self.history_gru.encode_prefix(
        batch['prefix_actions'], batch['prefix_len']
    )
    
    # CausalTransformer forward (with history conditioning)
    pred = self.model(
        noisy_trajectory, diff_steps[:, To:],
        tpe_start=cache_start_idx,
        cond=cond,
        history_emb=z_hist  # NEW
    )
    
    # BC loss (existing, unchanged)
    bc_loss = F.mse_loss(pred, target, reduction='none')
    bc_loss = (bc_loss * loss_mask.type(bc_loss.dtype)).mean()
    
    # Remote-PTP loss (NEW)
    # 从prefix中采样一个远端片段作为预测目标
    # remote_anchor: (B, m, Da) — m个远端历史动作
    # 只允许通过z_hist预测，不给其他线索
    remote_pred = self.remote_pred_head(z_hist)  # (B, m * Da)
    remote_pred = remote_pred.view(B, self.remote_m, self.action_dim)
    remote_target = batch['remote_anchor_actions']  # (B, m, Da)
    remote_ptp_loss = F.mse_loss(remote_pred, remote_target, reduction='mean')
    
    total_loss = bc_loss + self.ptp_weight * remote_ptp_loss
```

**Remote anchor采样策略**:

```python
# 在数据加载时:
# 对每个sample，从prefix中采样m=4个连续动作作为remote anchor
# 采样位置: 均匀分布在 [0, chunk_start - local_window] 范围内
# 如果prefix太短(<8步)，则不计算remote_ptp_loss

def sample_remote_anchor(prefix_actions, prefix_len, m=4):
    """
    prefix_actions: (B, T_max, Da)
    prefix_len: (B,)
    Returns: remote_anchor (B, m, Da), valid_mask (B,)
    """
    valid_mask = prefix_len >= 2 * m  # 至少8步才有意义
    # 在[0, prefix_len - m]范围内随机选起点
    max_start = (prefix_len - m).clamp(min=0)
    start_idx = (torch.rand(B) * max_start).long()
    # 提取连续m步
    remote_anchor = torch.stack([
        prefix_actions[b, start_idx[b]:start_idx[b]+m]
        for b in range(B)
    ])
    return remote_anchor, valid_mask
```

**Why this works**: 
- remote_pred_head只接收z_hist作为输入（不接收local window信息）
- 如果GRU没有记住远端历史，remote_pred_head无法预测正确
- 这直接将训练监督绑定到"超窗历史记忆质量"上

**Hyperparameters**:
- remote_m = 4 (预测4个连续远端动作)
- ptp_weight = 0.5 (与bc_loss同量级)
- remote_pred_head: Linear(256, 4 * action_dim)

### Training Plan

**Single-stage joint training**:

```
Loss = bc_loss + 0.5 * remote_ptp_loss

Optimizer: AdamW (lr=1e-4, betas=[0.9, 0.95])
  - CausalTransformer + HistoryGRU + remote_pred_head: weight_decay=1e-3
  - DP3Encoder: weight_decay=1e-6
Schedule: Cosine with 1000 warmup steps
EMA: same as CDP
Epochs: 3000
Batch: 64
```

**数据构造** (修改SequenceSampler):

每个sample需要额外提供:
```python
{
    'obs': ...,                    # (T_obs, ...) — 现有
    'action': ...,                 # (horizon, Da) — 现有
    'sample_start_idx': ...,       # 现有
    'buffer_start_idx': ...,       # 现有
    # NEW:
    'prefix_actions': ...,         # (T_prefix_max, Da) — chunk_start之前的所有动作，padded
    'prefix_len': ...,             # scalar — 实际prefix长度
    'remote_anchor_actions': ...,  # (m, Da) — 从prefix中采样的远端片段
    'remote_anchor_valid': ...,    # bool — prefix是否足够长
}
```

### Failure Modes and Diagnostics

| Failure Mode | Detection | Mitigation |
|-------------|-----------|------------|
| GRU hidden退化为常数 | 监控z_hist的cosine similarity跨步数变化；remote_ptp_loss不下降 | 增加GRU input dropout(0.1)；检查prefix数据是否正确 |
| Remote-PTP与BC loss冲突 | BC loss因ptp_weight上升 | 降低ptp_weight到0.1-0.3；检查梯度流 |
| Exposure bias (推理时自生成动作漂移) | Long-horizon任务后半段成功率急剧下降 | GRU输入加高斯噪声(σ=0.01)模拟推理时误差 |
| AdaLN-Zero训练太慢 | History conditioning直到后期才生效 | 适当增加lr或减少zero-init的scale |

### Novelty and Elegance Argument

**核心insight只有一个**: 扩散策略不利用历史(LDP发现) → 需要同时解决容量(compressed memory)和利用(explicit extra-window supervision) → **GRU+AdaLN-Zero+Remote-PTP是最简方案**。

**与最近工作的精确差异**:

| Work | Key Difference |
|------|---------------|
| CDP (CoRL 2025) | 固定窗口，无全历史，无历史利用监督 |
| LDP (CoRL 2025 Best Paper) | 观测端，全注意力O(T²)，PTP预测local past；我们：动作端，GRU压缩O(1)，Remote-PTP预测extra-window past |
| MTIL (RA-L 2025) | Mamba全历史但无扩散，无利用监督 |

**Paper narrative**:
> "Diffusion policies don't use history; giving them all-history access isn't enough — you must explicitly supervise extra-window history utilization. We show that a simple GRU memory + Remote-PTP achieves this with constant-cost inference."

## Claim-Driven Validation Sketch

### Claim 1: 全历史记忆提升long-horizon成功率 (Main Table)

- **Systems**: (1) CDP, (2) CDP + GRU-memory only, (3) CDP + Remote-PTP only, (4) ADP (GRU + Remote-PTP), (5) CDP-L100 (naive expansion)
- **Tasks**: 3 long-horizon (DexArt laptop, Adroit hammer, Adroit pen) + 1 short-horizon (MetaWorld reach as negative control)
- **Metric**: 成功率 ± std (3 seeds)
- **Success criterion**: ADP > CDP ≥ 5% on all 3 long-horizon tasks; ADP ≈ CDP on short-horizon (no regression)

### Claim 2: 模型真正利用历史语义 (History Intervention)

- **Experiment**: ADP在正常历史 vs truncated vs shuffled vs cross-episode replaced
- **Metric**: 成功率差异
- **Success criterion**: History corruption显著降低成功率 (≥10% drop)

### Claim 3: 推理成本可接受

- **Metric**: Wall-clock ms/step, GPU显存
- **Success criterion**: 推理延迟增加<20%, 显存增加<10%

## Experiment Handoff Inputs

- **Must-prove**: (1) 全历史提升成功率; (2) 模型真正利用历史语义; (3) 推理成本可接受
- **Must-run ablations**: GRU消融(#2), Remote-PTP消融(#3), History intervention
- **Appendix ablations**: GRU vs Mamba vs LSTM; AdaLN-Zero vs CrossAttn vs FiLM; adp2; 更多任务
- **Critical tasks**: DexArt laptop, Adroit hammer, Adroit pen, MetaWorld reach
- **Highest-risk**: (1) CDP固定窗口确实是瓶颈(需诊断实验验证); (2) Remote-PTP能否有效约束GRU编码质量

## Compute & Timeline

- Diagnostic (gating): 3 configs × 3 tasks × 1 seed ≈ 9 GPU-hours
- Main (5 systems × 4 tasks × 3 seeds): ≈ 60 runs × 3h ≈ 180 GPU-hours
- History intervention: 3 conditions × 3 tasks × 1 seed ≈ 27 GPU-hours (推理only，很快)
- Total: ≈ 220 GPU-hours (~9 days single A100)
- Timeline: 诊断2d → 开发5d → 主实验7d → 分析2d ≈ 16天
