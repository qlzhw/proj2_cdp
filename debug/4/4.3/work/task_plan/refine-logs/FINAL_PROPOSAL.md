# Research Proposal: History-Aware CDP via Extra-Window Mamba Memory and QC-Remote-PTP

## Problem Anchor

- **Bottom-line problem**: CDP的CausalTransformer在固定horizon=40的**动作序列**窗口内运行（前32步历史动作buffer + 后8步待生成动作），观测（32帧）通过DP3Encoder编码后经cross-attention注入。窗口外的所有历史信息被完全丢弃。
- **Must-solve bottleneck**: (1) 固定窗口丢失长程历史；(2) 扩散策略低利用历史(LDP发现)。
- **Paper identity**: CDP扩展论文。TEDi框架默认保留（允许必要局部适配）。
- **Constraints**: 8卡GPU(48GB)；4仿真环境；新增adp3。
- **Success condition**: ≥3个long-horizon任务上ADP > CDP ≥5%，推理延迟增加<30%。

## Method Thesis

CDP的local buffer负责短期因果动作连续性，HistoryMamba负责extra-window memory，QC-Remote-PTP逼模型真正使用这段memory。

## Contribution Focus

- **Dominant**: Extra-window Mamba dual-history memory (obs+action) + query-conditioned utilization supervision for CDP
- **Non-contributions**: 不提出新SSM；不提出新扩散框架；不做test-time reranking；不声称通用性

## Unified Time Indexing

```
Decision moment: τ = 32 + 8c (only at replanning boundaries, c=0,1,2,...)
Chunk size: C = 8
Local buffer size: L = 32

CDP Local Window:
  local_buffer = [a_{τ-32}, ..., a_{τ-1}]     (32 historical actions)
  future_chunk = [a_τ, ..., a_{τ+7}]            (8 actions to generate)
  obs_window   = [o_{τ-32}, ..., o_{τ-1}]       (32 obs frames → cross-attn)

Extra-Window (HistoryMamba's domain):
  extra_window = episode[0 : τ-33]  (everything BEFORE local buffer)
  K = floor((τ-32) / 8) extra-window chunks
  When τ ≤ 32: extra_window empty → z_hist = 0 → AdaLN-Zero → no effect

Three clean boundaries:
  [extra-window prefix | local 32-step buffer | 8-step generation]
       HistoryMamba         CDP original          CDP original
```

## Proposed Method

### Complexity Budget

- **Frozen/reused**: DP3Encoder (Stage 2冻结), TEDi框架 (scheduler, causal mask, KV cache, buffer lifecycle)
- **New trainable** (3):
  1. HistoryMamba: 2层Mamba-2 (d=256), split-pool chunk tokens, ~2M params
  2. AdaLN-Zero: per-block gated (shift, scale, gate), zero-init, ~1.5M params
  3. QC-Remote-PTP head: offset_emb + 2-layer MLP, ~30K params
- **Total new**: ~3.5M (vs CDP ~8M, +44%)

### Core Mechanism 1: HistoryMamba (Extra-Window Only, Split-Pool)

编码extra-window prefix的chunk-level token序列。每个8-step chunk用split-pool保留最小时序结构：

```python
def build_chunk_token(obs_embs_8, actions_8):
    obs_first  = obs_embs_8[:4].mean(0)   # first-half obs
    obs_last   = obs_embs_8[4:].mean(0)   # last-half obs
    act_first  = actions_8[:4].mean(0)     # first-half action
    act_last   = actions_8[4:].mean(0)     # last-half action
    return linear_proj(cat([obs_first, obs_last, act_first, act_last])) + pos_emb(k)
```

- 训练: K个chunk tokens → Mamba forward → z_hist (stateless)
- 推理: 每执行完一个8-step chunk，该chunk从local buffer弹出进入extra-window → append 1 token → O(1) update
- Train/inference完全同构

### Core Mechanism 2: AdaLN-Zero

每个TemporalTransformerBlock: `SiLU → Linear(256, 3H)`, zero-init → 训练初期z_hist=0时等价于原始CDP。

### Core Mechanism 3: QC-Remote-PTP (Strictly Extra-Window, Granularity-Aligned)

Query-conditioned预测extra-window中的远端chunk summary:

```python
# Query: offset Δ relative to extra-window end
# query_chunk_idx = K-1-Δ, ALL positions < τ-32 (guaranteed extra-window)
# Target: split-pool summary of queried chunk (first-half mean, last-half mean)
pred = head([z_hist, offset_emb(Δ)])  # → (2 * action_dim)
target = cat([actions[:4].mean(), actions[4:].mean()])  # same granularity as token
```

- 4 queries per sample, evenly spread across extra-window (recent → distant)
- 监督目标与chunk token粒度完全对齐
- 只通过z_hist预测，无shortcut

### Training: Two-Stage (LDP-validated)

```
Stage 1: CDP checkpoint (existing or train 1000 epochs)
Transition: Freeze DP3Encoder → cache obs embeddings (~150MB)
Stage 2: 2000 epochs, freeze DP3Encoder
  Loss = bc_loss + 0.5 * qc_remote_ptp_loss
  lr: 5e-5 (pretrained CausalTransformer), 1e-4 (new modules)
  AdamW, cosine, 500 warmup, EMA same as CDP
```

### Codebase Integration

| File | Change | Description |
|------|--------|-------------|
| policy/adp3.py | NEW | CDP + HistoryMamba + AdaLN-Zero + QC-Remote-PTP |
| model/diffusion/history_mamba.py | NEW | HistoryMamba class |
| model/diffusion/causal_transformer.py | MOD | Add history_adaLN per block |
| config/adp3.yaml | NEW | Based on cdp3.yaml |
| train.py | MOD | _POLICY_MAP + multi-stage |
| common/sampler.py | MOD | Return extra-window prefix + remote queries |
| scripts/cache_obs_embeddings.py | NEW | Cache DP3Encoder outputs |

### Failure Modes

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Mamba state→constant | z_hist variance→0 | Increase layers/d_state |
| QC-Remote-PTP→trivial | All Δ same loss | Widen query spread |
| Stage transition gap | Stage 2 bc_loss spike | Verify cached embeddings |
| Exposure bias | Late-episode collapse | Noise on Mamba input |
| Early episode (no extra-window) | z_hist=0 | AdaLN-Zero handles gracefully |

### Novelty

| Work | Difference |
|------|-----------|
| CDP | Fixed window, no extra-window memory, no utilization supervision |
| LDP | O(T²) full-attn obs-only, local PTP; we: O(n) Mamba obs+action, QC-Remote-PTP strictly extra-window |
| MTIL | Pure L2 no diffusion; we: keep diffusion + add utilization supervision |

## Claim-Driven Validation

### Claim 1: Extra-window memory improves long-horizon success (Main Table)
- Systems: CDP, ADP-obs-only, ADP-full, CDP-L100
- Tasks: 3 long-horizon + 1 short-horizon, 3 seeds
- Success: ADP-full > CDP ≥5% on long-horizon

### Claim 2: QC-Remote-PTP essential for utilization
- ADP-full vs ADP-w/o-PTP → degradation expected

### Claim 3: Model truly uses history semantics
- History intervention: truncate/shuffle/replace → ≥10% drop

### Claim 4: Acceptable inference cost
- Latency <30% increase, memory acceptable

### Appendix
- act-only vs obs-only vs full; Mamba vs GRU; AdaLN vs CrossAttn

## Compute & Timeline
- 8× GPU 48GB, ~560 GPU-hours, ~10 days must-run + ~5 days appendix
