# Round 2 Refinement

## Problem Anchor
[Verbatim — with user correction applied]

- **Bottom-line problem**: CDP的CausalTransformer在固定horizon=40的**动作序列**窗口内运行（前32步是历史动作buffer + 后8步是待生成动作），观测（32帧点云/图像）通过DP3Encoder编码后经**cross-attention**注入（与动作序列分离）。窗口外的所有历史信息被完全丢弃，无法利用更长期的历史来改善任务成功率，尤其在long-horizon灵巧操作任务中。
- **Must-solve bottleneck**: (1) 固定窗口丢失长程历史信息（观测+动作）；(2) 扩散策略存在"反向copycat问题"——LDP (CoRL 2025 Best Paper)证明：即使给予长观测历史上下文，扩散策略的action predictability ratio远低于1，模型低利用时间动作依赖关系。
- **Non-goals**: (a) 不做真实机器人实验（纯仿真验证）；(b) 不追求百万级长序列能力（任务最长~300步）。
- **NOTE**: 观测端历史建模**不再**是non-goal——LDP的3倍提升来自长观测上下文，排除它等于放弃最大收益来源。因果扩散架构**可以修改**——只要保持扩散策略框架即可，TEDi不是神圣不可变的约束。
- **Constraints**: 8卡GPU（每卡48GB显存）；4仿真环境（Adroit/DexArt/MetaWorld/RoboFactory）；基于现有CDP代码库改动；新增adp3（adp2作为后续工作）。
- **Success condition**: 在至少3个long-horizon任务上，ADP成功率显著超过CDP（≥5%绝对提升），推理延迟增加<30%。

## Anchor Check

- **Original bottleneck**: 固定窗口历史 + 扩散策略不利用历史
- **Why the revised method still addresses it**: 双历史（obs+action）同时解决capacity和utilization，Mamba线性复杂度保证可扩展性，Remote-PTP强制利用
- **Reviewer suggestions rejected as drift**: Self-verification仍然被删除；Memory tokens方案暂不采用（Mamba recurrent state已够用）
- **User correction applied**: LDP使用的是长观测历史（非动作历史），PTP预测过去动作但基于长观测上下文。需要同时考虑obs和action历史。

## Simplicity Check
  
- **Dominant contribution after revision**: Mamba-based dual-history (obs+action) memory + Remote-PTP显式窗外利用监督——首个在因果扩散策略中以线性成本同时利用全历史观测和动作的方案
- **Components removed or merged**:
  - 删除: self-verification, adp2(降为后续), 模块对比实验从主文降到appendix
- **Added (justified by user correction + literature)**:
  - Observation history encoder (Mamba压缩缓存的obs embeddings)
  - Multi-stage training (LDP-validated, necessary for obs history)
- **Why this is still the smallest adequate route**: 
  - Mamba是编码obs+action联合历史的最简架构（比Transformer更轻，比GRU对长序列更强）
  - Multi-stage训练是LDP验证的必要路径（not optional engineering choice）
  - 相比MTIL（全Mamba替代Transformer），我们只用Mamba做history encoder，保持CausalTransformer做扩散生成

## Changes Made

### 1. 从action-only到obs+action双历史 (MAJOR REVISION)
- **User said**: LDP用的是观测历史，不应排除obs端；需要综合考量
- **Action**: 引入obs history encoding：缓存DP3Encoder embedding → Mamba压缩 → 与action history融合
- **Reasoning**: LDP的3x提升来自长观测上下文；action history计算几乎免费；两者服务不同功能（obs=环境状态变化，action=运动模式）
- **Impact**: 方法信息量更大，但需要multi-stage训练

### 2. GRU → Mamba (ARCHITECTURE CHANGE)
- **User said**: 8卡48G条件下GRU可能过于保守；需考量Transformer或其他
- **Action**: 使用Mamba-2作为统一历史编码器，处理obs embedding + action的交织序列
- **Reasoning**: (1) MTIL已验证Mamba在full history机器人操作中有效; (2) O(n)训练+O(1)-like推理; (3) 比GRU对长序列信息保持更强; (4) 比Transformer推理更高效
- **Impact**: 更强的history encoding能力，推理效率不输GRU

### 3. LDP描述修正 (FACTUAL CORRECTION)
- **User said**: LDP是长观测历史（非动作历史），PTP预测过去动作但基于长obs上下文
- **Action**: 全文修正LDP描述，调整技术路线论证
- **Reasoning**: 准确引用是论文基础
- **Impact**: 技术叙事更准确

### 4. 约束更新: 8卡48G
- **User said**: 有8块GPU每个48G
- **Action**: 更新compute budget和实验规模
- **Reasoning**: 更多计算资源使得更powerful架构和更大实验矩阵可行
- **Impact**: 可以做更多seed/task的实验

## Revised Proposal

# Research Proposal: History-Aware Causal Diffusion Policy via Mamba Memory and Extra-Window Supervision

## Problem Anchor
[Same as above — see top of this document]

## Technical Gap

### How Current Methods Fail

CDP的CausalTransformer在horizon=40的**动作序列**窗口内运行：前32步（n_obs_steps=32）是**历史动作buffer**（带噪声注入的因果条件），后8步（n_action_steps=8）是待生成动作。观测（32帧点云/图像）由DP3Encoder编码后通过**cross-attention**注入，与动作序列分离。窗口外的所有历史观测和动作被完全丢弃。

**关键证据**:
1. CDP论文Limitation明确承认固定窗口限制long-horizon
2. LDP (CoRL 2025 Best Paper): 扩散策略在长观测历史下的action predictability ratio远低于1——模型低利用时间动作依赖，即使观测历史更长
3. BPP (ICLR 2026): 朴素长观测历史导致spurious correlations，需要smart selection
4. MTIL (RA-L 2025): Full history encoding (obs+action) via Mamba显著优于short history

### Why Naive Approaches Are Insufficient

| 方案 | 问题 |
|------|------|
| 扩大attention窗口 | O(T²)，300步→225倍计算 |
| 只加长观测历史 | LDP的发现：不加PTP则利用率极低 |
| 只加长动作历史 | 丢失环境状态变化的关键信息 |
| 全换成Mamba (MTIL) | 放弃扩散策略的生成多样性优势(MTIL用纯L2 loss) |

### Smallest Adequate Intervention

三个问题需要三个对应干预：
- **Obs capacity**: 缓存DP3Encoder embedding → Mamba压缩 → O(n)编码全观测历史
- **Action capacity**: 同一Mamba编码全动作历史（低维，几乎免费）
- **Utilization**: Remote-PTP只监督窗外信息 → 强制Mamba记忆有语义的远端信息

## Method Thesis

- **One-sentence thesis**: 通过Mamba将全部历史观测embedding和动作压缩为持久记忆状态，经AdaLN-Zero注入CausalTransformer，并用Remote Past-Token Prediction监督窗外历史利用，使因果扩散策略在长时程任务中真正利用超窗历史。

## Contribution Focus

- **Dominant contribution (unified)**: Mamba dual-history (obs+action) memory + extra-window supervision——首个让因果扩散策略以线性成本利用全部历史观测和动作的方法
- **Key insight**: 不是"如何编码全历史"（已有方案），而是"如何让因果扩散策略真正利用全历史"——需要同时解决capacity（Mamba压缩）和utilization（Remote-PTP监督）
- **Explicit non-contributions**: (a) 不提出新SSM模型（使用标准Mamba-2）；(b) 不做test-time reranking
- **Architecture flexibility**: 因果扩散架构可根据需要修改（只要保持扩散策略框架），TEDi不是硬约束

## Proposed Method

### Complexity Budget

- **Frozen / reused**: DP3Encoder（冻结后缓存embedding）, DDPMTEDiScheduler, chunked causal mask, KV cache, action buffer lifecycle
- **New trainable components** (3):
  1. `HistoryMamba`: Mamba-2 encoder for interleaved (obs_emb, action) tokens
  2. `AdaLN-Zero modulation layers`: per-block gated (shift, scale, gate) from history state
  3. `remote_pred_head`: Linear layer for Remote-PTP (predicts extra-window actions from history state only)
- **Intentionally not used**: Full-attention on history (O(T²)), Memory bank / retrieval, Self-verification, Multi-scale hierarchical compression, Separate obs/action encoders (unified Mamba更简洁)

### System Overview

```
Training (Multi-stage, LDP-validated):
  
  Stage 1: Short-context CDP training (existing)
    - Train DP3Encoder + CausalTransformer normally
    - 1000 epochs, standard CDP config
  
  Stage 2: Freeze DP3Encoder → Cache embeddings → Train history-aware decoder
    - For each training sample:
      1. Load cached obs embeddings for full episode: e_0, e_1, ..., e_T
      2. Interleave with actions: [(e_0, a_0), (e_1, a_1), ..., (e_{t-1}, a_{t-1})]
      3. HistoryMamba.encode_prefix() → z_hist
      4. CausalTransformer(local_window, obs_features, diff_steps, z_hist via AdaLN-Zero)
      5. Loss = bc_loss + λ * remote_ptp_loss
    - 2000 epochs, with history-specific lr

Inference (Online):
  At each chunk step t:
  1. obs_emb = DP3Encoder(o_t)                          # Existing
  2. z_hist, state = HistoryMamba.step(obs_emb, a_{t-1}, state)  # O(1) per step
  3. CausalTransformer denoise with KV cache + z_hist AdaLN-Zero  # Existing + modulation
  4. Execute n_action_steps=8, push to buffer, feed back to Mamba
```

### Core Mechanism 1: HistoryMamba (Unified Obs+Action History Encoder)

```python
class HistoryMamba(nn.Module):
    """
    Mamba-2 based encoder for interleaved observation embedding + action history.
    Stateless training interface + explicit state for inference.
    """
    def __init__(self, obs_emb_dim=64, action_dim=7, d_model=256, n_layers=2):
        super().__init__()
        # Project obs embeddings and actions to same dimension
        self.obs_proj = nn.Linear(obs_emb_dim, d_model)
        self.act_proj = nn.Linear(action_dim, d_model)
        
        # Learnable type embeddings (obs vs action)
        self.type_emb = nn.Embedding(2, d_model)  # 0=obs, 1=action
        
        # Mamba-2 backbone
        from mamba_ssm import Mamba2
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
    
    def _interleave(self, obs_embs, actions, lengths):
        """
        Interleave obs embeddings and actions into a single sequence.
        obs_embs: (B, T_max, obs_emb_dim)
        actions: (B, T_max, action_dim)
        Returns: (B, 2*T_max, d_model) interleaved tokens
        """
        B, T = obs_embs.shape[:2]
        obs_tokens = self.obs_proj(obs_embs) + self.type_emb(torch.zeros(B,T, dtype=torch.long, device=obs_embs.device))
        act_tokens = self.act_proj(actions) + self.type_emb(torch.ones(B,T, dtype=torch.long, device=obs_embs.device))
        # Interleave: o0, a0, o1, a1, ...
        tokens = torch.stack([obs_tokens, act_tokens], dim=2).reshape(B, 2*T, -1)
        return tokens
    
    def encode_prefix(self, obs_embs, actions, lengths):
        """
        Training: encode variable-length prefix (no internal state).
        Returns: z_hist (B, d_model)
        """
        tokens = self._interleave(obs_embs, actions, lengths)
        x = tokens
        for layer in self.mamba_layers:
            x = x + layer(x)  # residual Mamba
        x = self.norm(x)
        # Take the last valid token's representation
        # Use lengths to index: last token at position 2*length-1
        last_idx = (2 * lengths - 1).clamp(min=0)
        z_hist = x[torch.arange(x.shape[0]), last_idx]
        return self.output_proj(z_hist)  # (B, d_model)
    
    def step(self, obs_emb, action, state):
        """
        Inference: O(1) incremental update with explicit state.
        obs_emb: (B, obs_emb_dim)
        action: (B, action_dim)  — previous executed action
        state: list of per-layer Mamba states, or None
        Returns: z_hist (B, d_model), new_state
        """
        obs_token = self.obs_proj(obs_emb.unsqueeze(1)) + self.type_emb(torch.zeros(1, dtype=torch.long, device=obs_emb.device))
        act_token = self.act_proj(action.unsqueeze(1)) + self.type_emb(torch.ones(1, dtype=torch.long, device=obs_emb.device))
        tokens = torch.cat([obs_token, act_token], dim=1)  # (B, 2, d_model)
        
        new_state = []
        x = tokens
        for i, layer in enumerate(self.mamba_layers):
            s = state[i] if state else None
            x_out, s_new = layer.step(x, s)  # Mamba step function
            x = x + x_out
            new_state.append(s_new)
        x = self.norm(x)
        z_hist = self.output_proj(x[:, -1])  # last token
        return z_hist, new_state
```

**Key design decisions**:
- 统一编码器：obs embedding和action交织为单一序列，避免分离encoder的复杂度
- 2层Mamba-2：300步 × 2(obs+act) = 600 tokens，2层足够
- d_model=256, d_state=64：与CausalTransformer的n_emb=256对齐
- 训练时无状态（每sample独立），推理时O(1)增量更新
- 参数量：~2M（2层Mamba-2 with expand=2）

### Core Mechanism 2: AdaLN-Zero History Conditioning

[Same as Round 1 refinement — no change needed]

每个TemporalTransformerBlock新增gated (shift, scale, gate)，从z_hist生成，zero-initialized。训练开始时不影响已有CDP行为。

### Core Mechanism 3: Remote Past-Token Prediction (Remote-PTP)

[Same design as Round 1 refinement, with adjustment for dual-history]

```python
# Remote-PTP: 从z_hist预测窗外远端历史动作
# z_hist同时编码了obs+action历史，但预测目标只是远端动作
# → 强制Mamba记忆远端obs-action联合信息中与动作相关的部分

remote_pred = self.remote_pred_head(z_hist)  # (B, m * action_dim)
remote_target = batch['remote_anchor_actions']  # (B, m, Da)
remote_ptp_loss = F.mse_loss(remote_pred, remote_target)
```

**Why predict actions (not observations) as remote target**:
- Actions是低维的，prediction head更简单
- LDP已验证past action prediction有效
- Obs embedding prediction需要更复杂的decoder
- 预测远端动作已经隐式要求记忆对应的观测上下文（因为同一动作在不同观测条件下含义不同）

### Integration into CDP Codebase

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `policy/adp3.py` | NEW | 基于cdp3.py，加入HistoryMamba + AdaLN-Zero + Remote-PTP |
| `model/diffusion/history_mamba.py` | NEW | HistoryMamba类定义 |
| `model/diffusion/causal_transformer.py` | MODIFIED | TemporalTransformerBlock加入history_adaLN (AdaLN-Zero) |
| `config/adp3.yaml` | NEW | 基于cdp3.yaml，加入history相关超参 + multi-stage config |
| `train.py` | MODIFIED | _POLICY_MAP新增adp3; 支持multi-stage training |
| `common/sampler.py` | MODIFIED | SequenceSampler返回prefix数据 |
| `scripts/train_policy.sh` | MODIFIED | 支持adp3 |
| `scripts/cache_obs_embeddings.py` | NEW | Stage 1→2过渡：缓存DP3Encoder输出 |

**新增参数**: HistoryMamba ~2M + AdaLN-Zero 8层 ~1.5M + remote_pred_head ~10K ≈ 3.5M。相对CDP原有~8M，增加~44%。

### Training Plan

**Two-stage training** (LDP-validated):

```
Stage 1: Short-context CDP (existing training)
  - Standard CDP training with normal config
  - Train DP3Encoder + CausalTransformer
  - 1000 epochs (or use pre-trained CDP checkpoint)
  
  Output: trained DP3Encoder weights

Transition: Cache observation embeddings
  - Run DP3Encoder on all training data
  - Save embeddings to disk (one file per episode)
  - For 10 tasks × ~200 episodes × ~300 steps × 64-dim embedding
  - Storage: ~10 tasks × 200 × 300 × 64 × 4 bytes ≈ 150 MB (negligible)

Stage 2: Long-context history-aware decoder
  - Freeze DP3Encoder (use cached embeddings)
  - Initialize CausalTransformer from Stage 1 weights
  - Initialize HistoryMamba + AdaLN-Zero + remote_pred_head randomly
  - AdaLN-Zero: zero-init → training starts identical to Stage 1 checkpoint
  - Loss = bc_loss + 0.5 * remote_ptp_loss
  - 2000 epochs
  - lr: 5e-5 for pretrained CausalTransformer, 1e-4 for new modules
  
Optimizer: AdamW (betas=[0.9, 0.95])
  - CausalTransformer: lr=5e-5, weight_decay=1e-3
  - HistoryMamba + AdaLN-Zero: lr=1e-4, weight_decay=1e-3
  - remote_pred_head: lr=1e-4, weight_decay=0
Schedule: Cosine with 500 warmup steps
EMA: same as CDP
Batch: 64 (fits in 8×48G)
```

**为什么需要multi-stage而非end-to-end**:
- LDP的核心发现：PTP的收益在policy head，不在visual encoder
- 冻结DP3Encoder后缓存embedding → 训练速度提升10x（LDP数据）
- 避免长历史的反向传播穿过visual encoder（显存不可承受）
- Stage 1可复用已有CDP checkpoint，不需要从头训练

### Failure Modes and Diagnostics

| Failure Mode | Detection | Mitigation |
|-------------|-----------|------------|
| Mamba state退化为常数 | z_hist cosine similarity跨步数变化; remote_ptp_loss不收敛 | 增加Mamba层数/d_state; 检查interleave逻辑 |
| Obs embedding缓存与Stage 1不一致 | Stage 2 bc_loss起点远高于Stage 1终点 | 确保缓存用的是Stage 1最终checkpoint |
| Multi-stage transition gap | Stage 2初期性能下降 | AdaLN-Zero保证初始等价; 用较低lr for pretrained parts |
| Remote-PTP与BC loss冲突 | BC loss因ptp_weight上升 | 降低ptp_weight到0.1-0.3 |
| Exposure bias (推理时) | Long-horizon任务后半段急剧下降 | Mamba输入加噪声模拟推理误差 |
| Mamba step()接口不稳定 | 推理时结果与training不一致 | 用reference implementation验证step vs full forward一致性 |

### Novelty and Elegance Argument

**与最近工作的精确差异**:

| Work | Key Difference |
|------|---------------|
| CDP (CoRL 2025) | 固定窗口，无全历史，无利用监督 |
| LDP (CoRL 2025 Best Paper) | 长obs上下文+PTP，但全注意力O(T²)，无因果扩散框架，无action history |
| MTIL (RA-L 2025) | Mamba全历史(obs+act)，但非扩散策略(纯L2 loss)，无扩散生成过程 |
| BPP (ICLR 2026) | VLM选keyframe，不用扩散策略 |
| Hiveformer (CoRL 2022) | 全历史全注意力O(T²)，非扩散，keypose级(~10步) |

**论文叙事**:
> "扩散策略不利用历史（LDP发现）。我们证明：在因果扩散框架中，Mamba双历史记忆（obs+action）+ 显式窗外监督（Remote-PTP）是让模型真正利用全历史的最简方案。"

**Key differentiator vs MTIL**: MTIL用Mamba替代了整个policy架构（纯L2 loss）；我们保持扩散策略的生成多样性优势，只用Mamba做history encoder。这是"给扩散策略加记忆"而非"用Mamba替代扩散策略"。

**Key differentiator vs LDP**: LDP用全注意力+PTP，O(T²)不可扩展且只做obs端；我们用Mamba O(n)+Remote-PTP，同时编码obs+action，且保持扩散生成框架。注意：扩散架构细节（如是否保留TEDi的chunk-wise denoising）可根据实验结果灵活调整。

## Claim-Driven Validation Sketch

### Claim 1: 双历史记忆提升long-horizon成功率 (Main Table)

- **Systems**: (1) CDP, (2) ADP-act-only (Mamba action history only), (3) ADP-obs-only (Mamba obs history only), (4) ADP-full (Mamba obs+action), (5) CDP-L100 (naive window expansion)
- **Tasks**: 3 long-horizon (DexArt laptop, Adroit hammer, Adroit pen) + 1 short-horizon (MetaWorld reach as negative control)
- **Metric**: 成功率 ± std (3 seeds)
- **Success criterion**: ADP-full > CDP ≥ 5% on long-horizon; ADP-full ≥ ADP-act-only ≥ ADP-obs-only (obs贡献 > action贡献)

### Claim 2: Remote-PTP是利用全历史的关键 (Utilization Defense)

- **Systems**: ADP-full vs ADP-full-w/o-PTP (no remote supervision)
- **Expected**: 无PTP的全历史模型性能退化（LDP的核心发现在我们框架中复现）

### Claim 3: 模型真正利用历史语义 (History Intervention)

- **Experiment**: ADP-full在正常历史 vs truncated vs shuffled vs cross-episode replaced
- **Success criterion**: History corruption ≥ 10% drop

### Claim 4: 推理成本可接受

- **Metric**: Wall-clock ms/step, GPU显存
- **Success criterion**: 推理延迟增加<30%, 显存可控

## Experiment Handoff Inputs

- **Must-prove**: (1) 双历史提升成功率; (2) Remote-PTP是关键; (3) 历史语义被利用; (4) 推理成本可接受
- **Must-run ablations**: obs-only vs act-only vs both; with/without Remote-PTP; history intervention
- **Appendix ablations**: Mamba vs GRU vs Transformer; AdaLN-Zero vs CrossAttn; adp2; 更多任务; self-verification
- **Critical tasks**: DexArt laptop, Adroit hammer, Adroit pen, MetaWorld reach
- **Highest-risk**: (1) Multi-stage training的过渡是否平滑; (2) Mamba-2的step()接口在推理中是否稳定; (3) Remote-PTP是否有效约束Mamba编码质量

## Compute & Timeline

- **Hardware**: 8× GPU (48GB each)
- Stage 1: 使用现有CDP checkpoints (已有)
- Obs embedding caching: ~2h per task (8 tasks并行 ≈ 2h total)
- Stage 2 training: 每个config ~8h on single GPU; 8 GPUs并行可跑8个configs同时
- Main experiment: 5 systems × 4 tasks × 3 seeds = 60 runs; 8 GPUs并行 → ~60h wall-clock
- History intervention: inference only, ~1h
- Total: ~70h wall-clock (~560 GPU-hours)
- Timeline: Stage 1准备(1天) → 开发(5天) → 主实验(4天) → 分析(2天) ≈ 12天
