# Round 3 Refinement

## Problem Anchor
[Re-anchored to CDP extension after drift correction]

- **Bottom-line problem**: CDP的CausalTransformer在固定horizon=40的**动作序列**窗口内运行（前32步历史动作buffer + 后8步待生成动作），观测（32帧）通过DP3Encoder编码后经cross-attention注入。窗口外的所有历史信息被完全丢弃，限制了long-horizon任务的成功率。
- **Must-solve bottleneck**: (1) 固定窗口丢失长程历史信息；(2) 扩散策略低利用历史——LDP证明即使给更长观测上下文，action predictability ratio仍远低于1。
- **Non-goals**: (a) 不做真实机器人实验；(b) 不追求百万级长序列。
- **Paper identity**: **CDP扩展论文**——在CDP的TEDi因果扩散框架上加入全历史记忆模块。如果TEDi某些细节（如chunk-wise denoising的具体实现）与长历史记忆不兼容，允许局部适配，但整体扩散生成架构保持不变。CDP是直接前身和核心baseline。
- **Constraints**: 8卡GPU（每卡48GB）；4仿真环境；基于CDP代码库改动；新增adp3。
- **Success condition**: 在至少3个long-horizon任务上，ADP成功率≥CDP+5%，推理延迟增加<30%。

## Anchor Check

- **Original bottleneck**: CDP固定窗口 + 扩散策略不利用历史
- **Drift correction**: Round 2被检测到从"CDP扩展"漂移到"通用长上下文扩散"。本轮修正：锁回CDP扩展定位，TEDi是默认保留的框架（允许必要局部适配），CDP是核心baseline和代码基础
- **"Any diffusion framework"已删除**: 不再声称通用性，聚焦于CDP/TEDi的具体扩展

## Simplicity Check

- **Dominant contribution**: 为CDP加入Mamba双历史记忆 + Query-Conditioned Remote-PTP窗外利用监督
- **Components removed**: "any diffusion framework"声明、adp2、self-verification
- **Round 3新修正**: Remote-PTP从随机位置改为query-conditioned（修复phase clock漏洞）；精确定义序列化契约

## Changes Made

### 1. 锁回CDP扩展定位 (CRITICAL — drift fix)
- **Reviewer said**: Problem Anchor从"CDP修复"漂移到"通用长上下文扩散"，paper身份模糊
- **Action**: 删除"any diffusion framework"，明确TEDi是默认保留框架（允许必要局部适配），CDP是核心baseline
- **Reasoning**: CDP是我们的代码库，最熟悉，调试最快；论文聚焦"如何让CDP用到窗外历史"比"通用方案"更锋利
- **Impact**: Paper identity清晰化

### 2. Remote-PTP改为Query-Conditioned (CRITICAL — mechanism fix)
- **Reviewer said**: 不告诉模型预测哪个位置→欠定；固定位置→退化为phase clock
- **Action**: 引入相对偏移查询q(Δ)，Δ∈{1,2,3,4}对应chunk级偏移(每chunk=8步，实际偏移=Δ×8步)，`head([z_hist, q(Δ)]) → remote action snippet at t0-Δ*8`
- **Reasoning**: 查询条件化让模型必须根据z_hist和偏移量精确回忆远端历史；多个Δ值防止退化为单一phase clock
- **Impact**: Remote-PTP机制严格化，可学习目标明确

### 3. 精确定义序列化契约 (CRITICAL — implementation fix)
- **Reviewer said**: prefix截止时刻、obs/action配对、online update顺序不够精确
- **Action**: 完整定义训练/推理一致的序列化规范
- **Impact**: 可直接实现

### 4. Obs-side supervision选择：收缩claim (IMPORTANT)
- **Reviewer said**: Remote-PTP只监督action，对obs历史利用只是indirect
- **Action**: 不加额外obs target（避免复杂化），收缩claim为"Mamba记忆编码obs+action联合历史，Remote-PTP通过预测远端动作间接验证记忆质量"
- **Reasoning**: 远端动作的预测隐式需要对应观测上下文（因为同一关节角度在不同场景下含义不同），不需要显式obs prediction
- **Impact**: Claim更诚实，无额外模块

### 5. 主表简化 (Simplification)
- **Reviewer said**: ADP-act-only和ADP-obs-only留一个在主表
- **Action**: 主表保留ADP-obs-only作为消融（因为obs是更重要的历史源），ADP-act-only降为appendix
- **Impact**: 主表更简洁

## Revised Proposal

# Research Proposal: History-Aware CDP via Mamba Memory and Extra-Window Supervision

## Problem Anchor
[Same as top of this document]

## Technical Gap

CDP的CausalTransformer在horizon=40的**动作序列**窗口内运行（前32步历史动作buffer，后8步待生成动作）。观测（32帧点云/图像）由DP3Encoder编码后通过cross-attention注入，与动作序列分离。**窗口外的所有历史信息被完全丢弃。**

两个独立问题使朴素扩窗不够：
1. **Capacity**: O(T²)注意力使300步全历史在当前CausalTransformer中不可承受
2. **Utilization**: LDP (CoRL 2025 Best Paper)证明扩散策略即使给更多上下文也不会自动利用

## Method Thesis

- **One-sentence thesis**: 在CDP的TEDi因果扩散框架上，通过Mamba-2将全历史观测embedding和动作压缩为持久记忆状态，经AdaLN-Zero注入CausalTransformer，并用Query-Conditioned Remote-PTP监督窗外历史利用，使CDP在长时程任务中真正利用超窗历史。

## Contribution Focus

- **Dominant contribution**: 为CDP加入constant-cost双历史记忆（Mamba-2编码obs+action） + query-conditioned extra-window supervision (QC-Remote-PTP)
- **Key insight**: "如何让CDP真正利用窗外历史"——需同时解决capacity（Mamba压缩）和utilization（QC-Remote-PTP监督）
- **Explicit non-contributions**: 不提出新SSM模型；不提出新扩散框架；不做test-time reranking；不声称通用性

## Proposed Method

### Complexity Budget

- **Frozen / reused**: DP3Encoder（Stage 2冻结），TEDi因果扩散框架（chunked causal mask, KV cache, DDPMTEDiScheduler），action buffer lifecycle
- **New trainable** (3个):
  1. `HistoryMamba`: 2层Mamba-2 (d_model=256) — 统一编码obs embedding + action的交织序列
  2. `AdaLN-Zero`: 每个TemporalTransformerBlock加gated (shift, scale, gate) from z_hist — zero-init
  3. `QC-Remote-PTP head`: Linear([z_hist; q(Δ)], m×action_dim) — query-conditioned远端动作预测
- **Not used**: Full attention on history, memory bank, self-verification, separate obs/action encoders, 可变噪声

### Serialization Contract (精确定义)

```
Notation:
  t0 = current chunk start (in episode-level timestep)
  chunk_size = n_action_steps = 8
  local_window = horizon = 40 (32 history + 8 generate)
  
  Episode timeline: o_0, a_0, o_1, a_1, ..., o_{t0-1}, a_{t0-1}, [o_{t0}...o_{t0+31}], [generate a_{t0+32}...a_{t0+39}]

Training sample definition:
  {
    // Existing CDP fields:
    obs:    o_{t0} ... o_{t0+31}           // 32 observation frames → DP3Encoder → cross-attn
    action: a_{t0} ... a_{t0+39}           // 40-step action trajectory (32 history + 8 future)
    
    // NEW fields for history:
    prefix_obs_embs:  e_0, e_1, ..., e_{t0-1}   // cached DP3Encoder outputs, (t0, obs_emb_dim)
    prefix_actions:   a_0, a_1, ..., a_{t0-1}    // historical actions, (t0, action_dim)
    prefix_len:       t0                          // actual prefix length (0 if episode start)
    
    // NEW fields for QC-Remote-PTP:
    remote_queries:   [Δ_1, Δ_2, Δ_3, Δ_4]     // chunk-level offsets, Δ_k ∈ {1,2,3,4}
    remote_targets:   [a_{t0-Δ_1*8 : t0-Δ_1*8+m}, ...]  // m=4 consecutive actions at each offset
    remote_valid:     [bool, bool, bool, bool]    // False if offset exceeds prefix
  }

Mamba input sequence (training):
  [(e_0, a_0), (e_1, a_1), ..., (e_{t0-1}, a_{t0-1})]
  → interleaved tokens: [proj(e_0)+type_obs, proj(a_0)+type_act, proj(e_1)+type_obs, ...]
  → total 2*t0 tokens
  → Mamba encode → take last token → z_hist ∈ R^256

Online inference contract:
  Step 0 (episode start):
    state = None
    
  Each decision step (every 8 timesteps):
    1. Get obs_emb = DP3Encoder(current_obs)   // for cross-attention (existing)
    2. z_hist, state = HistoryMamba.step(obs_emb, prev_action, state)  // O(1) update
       - prev_action = last executed action (zero if first step)
       - Appends (obs_emb, prev_action) as 2 new tokens to Mamba state
    3. Run CausalTransformer with:
       - action_buffer (existing 40-step, with existing noise injection)
       - cond = obs_features via cross-attention (existing)
       - history_emb = z_hist via AdaLN-Zero (NEW)
       - KV cache (existing)
    4. Execute first 8 actions
    5. Pop buffer, push new noise (existing TEDi lifecycle)
    6. prev_action = last executed action
```

### Core Mechanism 1: HistoryMamba

[Same architecture as Round 2 — 2-layer Mamba-2, stateless training, explicit-state inference]

Key parameters: d_model=256, d_state=64, d_conv=4, expand=2, n_layers=2.
Parameters: ~2M. Input: interleaved (obs_emb, action) tokens. Output: z_hist ∈ R^256.

### Core Mechanism 2: AdaLN-Zero History Conditioning

[Same as Round 2 — per-block gated modulation, zero-initialized]

Each TemporalTransformerBlock adds:
```python
self.history_adaLN = nn.Sequential(
    nn.SiLU(),
    nn.Linear(256, 3 * hidden_size)  # shift, scale, gate
)
nn.init.zeros_(self.history_adaLN[-1].weight)
nn.init.zeros_(self.history_adaLN[-1].bias)
```

### Core Mechanism 3: Query-Conditioned Remote-PTP (QC-Remote-PTP)

**关键改进**: 不再从随机位置预测，而是通过相对偏移查询q(Δ)精确指定预测位置。

```python
class QCRemotePTPHead(nn.Module):
    def __init__(self, d_model=256, max_delta=4, m=4, action_dim=7):
        super().__init__()
        # Learnable offset embeddings
        self.delta_emb = nn.Embedding(max_delta + 1, d_model)  # Δ ∈ {0,1,2,3,4}
        # Prediction head: takes [z_hist; q(Δ)]
        self.pred = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, m * action_dim)
        )
        self.m = m
        self.action_dim = action_dim
    
    def forward(self, z_hist, deltas):
        """
        z_hist: (B, d_model)
        deltas: (B, num_queries) — chunk-level offsets Δ
        Returns: predicted remote actions (B, num_queries, m, action_dim)
        """
        B, Q = deltas.shape
        q = self.delta_emb(deltas)  # (B, Q, d_model)
        z_expanded = z_hist.unsqueeze(1).expand(-1, Q, -1)  # (B, Q, d_model)
        input_feat = torch.cat([z_expanded, q], dim=-1)  # (B, Q, 2*d_model)
        pred = self.pred(input_feat)  # (B, Q, m*action_dim)
        return pred.view(B, Q, self.m, self.action_dim)
```

**训练时loss计算**:

```python
def compute_remote_ptp_loss(self, z_hist, batch):
    """
    For each query Δ_k, predict m=4 actions starting at position (t0 - Δ_k * chunk_size).
    Δ_k ∈ {1,2,3,4} → 查询1-4个chunk前的动作片段.
    """
    deltas = batch['remote_queries']        # (B, 4) — values in {1,2,3,4}
    targets = batch['remote_targets']       # (B, 4, m, Da) — ground truth
    valid = batch['remote_valid']           # (B, 4) — bool mask
    
    pred = self.qc_remote_ptp(z_hist, deltas)  # (B, 4, m, Da)
    
    # Only compute loss on valid queries (where prefix is long enough)
    loss = F.mse_loss(pred, targets, reduction='none')  # (B, 4, m, Da)
    loss = loss.mean(dim=(-1, -2))  # (B, 4) — per-query loss
    loss = (loss * valid.float()).sum() / valid.float().sum().clamp(min=1)
    
    return loss
```

**Why query-conditioned fixes the phase clock problem**:
- 模型必须根据**不同的Δ值**回忆不同位置的历史 — 不能靠单一phase信号
- Δ=1 (8步前) 和 Δ=4 (32步前) 需要不同的记忆检索 — 强制z_hist编码时间定位信息
- 如果Mamba没有真正记忆远端历史，不同Δ的预测都会失败
- 4个查询在一个batch中同时训练 → 高效

**Hyperparameters**:
- max_delta = 4 (查询1-4个chunk前，即8-32步前的动作)
- m = 4 (每次预测4个连续动作)
- num_queries = 4 (每个sample 4个查询)
- ptp_weight = 0.5

### Integration into CDP

| 文件 | 变更 | 说明 |
|------|------|------|
| `policy/adp3.py` | NEW | 基于cdp3.py，加HistoryMamba + AdaLN-Zero + QC-Remote-PTP |
| `model/diffusion/history_mamba.py` | NEW | HistoryMamba类 |
| `model/diffusion/causal_transformer.py` | MODIFIED | 每个block加history_adaLN |
| `config/adp3.yaml` | NEW | 基于cdp3.yaml |
| `train.py` | MODIFIED | _POLICY_MAP加adp3; multi-stage支持 |
| `common/sampler.py` | MODIFIED | 返回prefix + remote query数据 |
| `scripts/cache_obs_embeddings.py` | NEW | 缓存DP3Encoder输出 |

新增参数: ~3.5M (HistoryMamba ~2M + AdaLN-Zero ~1.5M + QC-Remote-PTP ~30K)

### Training Plan

```
Stage 1: Standard CDP training (existing checkpoint可复用)
  → Output: trained DP3Encoder + CausalTransformer

Transition: Cache obs embeddings (~150MB, ~2h)

Stage 2: History-aware decoder training
  - Freeze DP3Encoder
  - Warm-start CausalTransformer from Stage 1
  - New: HistoryMamba + AdaLN-Zero (zero-init) + QC-Remote-PTP
  - Loss = bc_loss + 0.5 * qc_remote_ptp_loss
  - 2000 epochs
  - lr: 5e-5 (pretrained), 1e-4 (new)
  - AdamW, cosine schedule, 500 warmup
```

### Failure Modes

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Mamba state → constant | z_hist variance → 0; ptp_loss flat | Add Mamba input dropout; check interleave |
| QC-Remote-PTP → phase clock | 所有Δ的loss相同; intervention test不敏感 | 增大max_delta; 验证不同Δ的loss差异 |
| Stage transition gap | Stage 2 bc_loss起点≫Stage 1终点 | 确认缓存embedding与Stage 1 checkpoint一致 |
| Exposure bias | Long-horizon后半段崩溃 | Mamba输入加噪声(σ=0.01) |

### Novelty Argument

**Paper narrative**:
> "CDP的固定动作窗口丢失长程历史，且扩散策略不会自动利用更多上下文（LDP发现）。我们为CDP加入Mamba双历史记忆和query-conditioned窗外监督，使其在长时程任务中真正利用全历史信息。"

**精确差异**:

| Work | 本方法的差异 |
|------|------------|
| CDP (CoRL 2025) | 固定窗口，无全历史记忆，无利用监督 |
| LDP (CoRL 2025 Best) | 长obs上下文+PTP(local past)，全注意力O(T²)，非TEDi，无action history |
| MTIL (RA-L 2025) | Mamba全历史，但纯L2 loss非扩散，无利用监督 |

**Key differentiators**:
1. vs LDP: 我们用Mamba O(n)替代全注意力O(T²)；编码obs+action双历史；QC-Remote-PTP预测窗外远端（而非LDP的local past）；在TEDi因果扩散框架内工作
2. vs MTIL: 我们保持扩散策略生成优势；加入显式利用监督(QC-Remote-PTP)；MTIL无此机制
3. vs CDP: 加入Mamba全历史记忆 + 窗外利用监督；其余保持不变

## Claim-Driven Validation

### Claim 1: 全历史记忆提升long-horizon成功率 (Main Table)

- **Systems**: (1) CDP, (2) ADP-obs-only (消融), (3) ADP-full (obs+action), (4) CDP-L100 (naive expansion)
- **Tasks**: 3 long-horizon (DexArt laptop, Adroit hammer, Adroit pen) + 1 short-horizon (MetaWorld reach)
- **Metric**: success rate ± std (3 seeds)
- **Success**: ADP-full > CDP ≥ 5% on long-horizon

### Claim 2: QC-Remote-PTP是利用全历史的关键

- **Systems**: ADP-full vs ADP-full-w/o-PTP
- **Expected**: 无PTP → 性能退化

### Claim 3: 模型真正利用历史语义

- **Experiment**: ADP-full在正常历史 vs truncated/shuffled/cross-episode-replaced
- **Success**: Corruption ≥ 10% drop

### Claim 4: 推理成本可接受

- **Metric**: ms/step, GPU memory
- **Success**: Latency increase < 30%

### Appendix ablations
- ADP-act-only vs ADP-obs-only vs ADP-full (分离obs/action贡献)
- Mamba vs GRU vs Transformer history encoder
- AdaLN-Zero vs CrossAttn injection
- QC-Remote-PTP的max_delta和m的sensitivity

## Compute

- **Hardware**: 8× GPU (48GB)
- Main: 4 systems × 4 tasks × 3 seeds = 48 runs; 8 GPUs并行 → ~48h
- Ablation (Claim 2): 1 system × 4 tasks × 3 seeds = 12 runs → ~12h
- Intervention (Claim 3): inference only → ~1h
- Appendix: ~80h wall-clock
- **Total wall-clock**: ~10 days (must-run) + ~5 days (appendix)
