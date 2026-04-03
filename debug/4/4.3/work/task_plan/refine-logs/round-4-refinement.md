# Round 4 Refinement (Final Tightening)

## Problem Anchor
[Unchanged from Round 3 — CDP extension, drift fixed]

## Anchor Check
- Problem Anchor: **preserved** (confirmed by GPT-5.4 Round 3)
- Paper identity: **locked** — History-Aware CDP extension
- No drift

## Simplicity Check
- Dominant contribution unchanged: Mamba extra-window memory + QC-Remote-PTP for CDP
- Round 4 changes are **purely precision fixes**, no new modules or claims

## Changes Made

### 1. 统一时间索引为τ-based定义 (CRITICAL — final precision)
- **Reviewer said**: t0记号与CDP原生窗口语义不统一；local history和extra-window的边界模糊
- **Action**: 全部重写为τ-based精确数学定义
- **Impact**: 无歧义的实现规范

### 2. HistoryMamba收缩为只编码extra-window prefix (Simplification)
- **Reviewer said**: 如果Mamba编码full prefix，QC-Remote-PTP的"only accessible through z_hist"不严格成立
- **Action**: HistoryMamba只编码τ-33之前的历史（即local 32-step buffer之外的部分）
- **Reasoning**: Local 32-step history已由CDP原生buffer处理；Mamba只负责extra-window → paper叙事最干净：**"CDP buffer负责短期因果连续性，HistoryMamba负责extra-window memory"**
- **Impact**: Shortcut完全堵死；职责分离清晰

### 3. QC-Remote-PTP query位置严格约束在extra-window内 (CRITICAL)
- **Reviewer said**: Δ=1..4 (8-32步前)可能落在local buffer内，不是真正的extra-window
- **Action**: Query位置从extra-window prefix中采样，保证 < τ-32；采样策略覆盖近端和远端
- **Impact**: "only accessible through z_hist"严格成立

### 4. 训练/推理memory update同构化 (CRITICAL)
- **Reviewer said**: 训练per-step token vs 推理per-chunk append不一致
- **Action**: 统一为chunk-level tokenization — 训练和推理都按8-step chunk追加(obs_embs, actions)
- **Impact**: 训练/推理完全同构

## Revised Proposal (Final Version)

# History-Aware CDP via Extra-Window Mamba Memory and QC-Remote-PTP

## Problem Anchor
CDP's CausalTransformer operates on a fixed 40-step action sequence: 32-step local history buffer + 8-step generation, with 32-frame observations via cross-attention. All extra-window history is discarded.
Paper identity: CDP extension. TEDi framework preserved.
Success: ≥5% on ≥3 long-horizon tasks, latency <30% increase.

## Method Thesis
Add a Mamba-2 extra-window memory encoder (encoding historical obs embeddings + actions BEFORE the local buffer) to CDP, inject via AdaLN-Zero, and supervise with QC-Remote-PTP to ensure the model actually utilizes extra-window history.

**One sentence**: CDP's local buffer handles short-term causal action continuity; HistoryMamba handles extra-window memory; QC-Remote-PTP forces the model to truly use that memory.

## Contribution Focus
Dominant: Extra-window Mamba memory + QC-Remote-PTP supervision for CDP
Not claimed: new SSM, new diffusion framework, generality beyond CDP, test-time reranking

## Unified Time Indexing (τ-based)

```
MATHEMATICAL DEFINITION
========================

Decision moment: τ (episode-level timestep at which a new 8-action chunk is planned)
Chunk size: C = n_action_steps = 8
Local buffer size: L = n_obs_steps = 32 (in action steps)
Horizon: H = 40 (L + C)

CDP Local Window (action sequence in CausalTransformer):
  local_buffer  = [a_{τ-L}, a_{τ-L+1}, ..., a_{τ-1}]     # 32 historical actions
  future_chunk  = [a_τ, a_{τ+1}, ..., a_{τ+C-1}]           # 8 actions to generate
  full_window   = [a_{τ-L}, ..., a_{τ+C-1}]                 # 40-step trajectory

Observation cross-attention:
  obs_window = [o_{τ-L}, o_{τ-L+1}, ..., o_{τ-1}]          # 32 observation frames
  → DP3Encoder → obs_features (B, 32, obs_emb_dim) → cross-attn

EXTRA-WINDOW HISTORY (everything before the local buffer):
  extra_window = episode[0 : τ-L-1]
  extra_len = max(0, τ - L)

  When τ ≤ L (early episode): extra_window is empty, z_hist = zero vector
  When τ > L: extra_window contains the first τ-L steps of the episode

HistoryMamba INPUT (extra-window only, chunk-level tokens):
  For each past chunk k = 0, 1, ..., K-1 where K = floor((τ-L) / C):
    obs_embs_chunk_k  = pool(e_{k*C}, e_{k*C+1}, ..., e_{k*C+C-1})  # mean-pool 8 obs embs
    actions_chunk_k   = pool(a_{k*C}, a_{k*C+1}, ..., a_{k*C+C-1})   # mean-pool 8 actions
    token_k = proj(obs_embs_chunk_k) + proj(actions_chunk_k) + pos_emb(k)
  
  → Mamba input: [token_0, token_1, ..., token_{K-1}]  (K tokens, one per chunk)
  → For 300-step episode with τ=300: K = floor((300-32)/8) = 33 tokens — very compact
  → z_hist = Mamba output at last token position ∈ R^256

QC-Remote-PTP QUERY POSITIONS (strictly extra-window):
  Valid queries must satisfy: query_pos < τ - L  (i.e., before local buffer)
  Offset Δ is defined in chunk units relative to extra-window END:
    query_chunk_idx = K - 1 - Δ  where Δ ∈ {0, 1, ..., min(K-1, max_Δ-1)}
    query_action_start = query_chunk_idx * C
    target = [a_{query_action_start}, ..., a_{query_action_start + m - 1}]
  
  Default: sample num_queries=4 offsets, covering both recent extra-window and distant past
  Example (τ=200, L=32, C=8):
    Extra-window: steps 0..167, K=21 chunks
    Δ=0: chunk 20 → actions at step 160..163  (most recent extra-window)
    Δ=5: chunk 15 → actions at step 120..123  (mid-range)
    Δ=10: chunk 10 → actions at step 80..83   (distant)
    Δ=19: chunk 1 → actions at step 8..11     (near episode start)
  
  GUARANTEE: ALL query positions are strictly before τ-L (step 168 in this example)
             → NOT accessible from CDP's local 32-step buffer
             → ONLY accessible through z_hist
```

## Core Mechanisms

### 1. HistoryMamba (Extra-Window Only)

```python
class HistoryMamba(nn.Module):
    """
    Encodes ONLY extra-window history (before local buffer).
    Chunk-level tokenization for train/inference consistency.
    """
    def __init__(self, obs_emb_dim=64, action_dim=7, d_model=256, n_layers=2):
        super().__init__()
        self.obs_proj = nn.Linear(obs_emb_dim, d_model)
        self.act_proj = nn.Linear(action_dim, d_model)
        self.pos_emb = nn.Embedding(512, d_model)  # max 512 chunks
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model
    
    def _build_chunk_tokens(self, obs_emb_chunks, action_chunks, num_chunks):
        """
        obs_emb_chunks: (B, K_max, obs_emb_dim) — mean-pooled per chunk
        action_chunks: (B, K_max, action_dim) — mean-pooled per chunk
        num_chunks: (B,) — actual number of extra-window chunks
        """
        B, K_max = obs_emb_chunks.shape[:2]
        pos_ids = torch.arange(K_max, device=obs_emb_chunks.device).unsqueeze(0).expand(B,-1)
        tokens = self.obs_proj(obs_emb_chunks) + self.act_proj(action_chunks) + self.pos_emb(pos_ids)
        return tokens  # (B, K_max, d_model)
    
    def encode_prefix(self, obs_emb_chunks, action_chunks, num_chunks):
        """Training: stateless, encode extra-window prefix."""
        tokens = self._build_chunk_tokens(obs_emb_chunks, action_chunks, num_chunks)
        x = tokens
        for layer in self.mamba_layers:
            x = x + layer(x)
        x = self.norm(x)
        # Take last valid chunk's representation
        last_idx = (num_chunks - 1).clamp(min=0)
        z_hist = x[torch.arange(x.shape[0]), last_idx]
        return self.output_proj(z_hist)  # (B, d_model)
    
    def step(self, new_chunk_obs_emb, new_chunk_action, state):
        """
        Inference: append ONE new chunk token, O(1) update.
        Called every C=8 steps when a new chunk exits the local buffer.
        """
        B = new_chunk_obs_emb.shape[0]
        chunk_idx = state['chunk_count'] if state else 0
        pos_id = torch.tensor([chunk_idx], device=new_chunk_obs_emb.device).expand(B)
        token = (self.obs_proj(new_chunk_obs_emb.unsqueeze(1)) 
                + self.act_proj(new_chunk_action.unsqueeze(1))
                + self.pos_emb(pos_id).unsqueeze(1))  # (B, 1, d_model)
        
        new_state = {'chunk_count': chunk_idx + 1, 'mamba_states': []}
        x = token
        for i, layer in enumerate(self.mamba_layers):
            s = state['mamba_states'][i] if state else None
            x_out, s_new = layer.step(x, s)
            x = x + x_out
            new_state['mamba_states'].append(s_new)
        x = self.norm(x)
        z_hist = self.output_proj(x[:, -1])
        return z_hist, new_state
```

**Train/Inference同构**:
- 训练: chunk-level tokens → Mamba forward → z_hist
- 推理: 每执行完一个8-step chunk, 该chunk从local buffer弹出后进入extra-window → append一个新chunk token到Mamba → z_hist更新
- **两者都是chunk-level tokenization, 完全同构**

### 2. AdaLN-Zero
[Unchanged from Round 3]

### 3. QC-Remote-PTP (Strictly Extra-Window)

```python
class QCRemotePTPHead(nn.Module):
    def __init__(self, d_model=256, num_queries=4, m=4, action_dim=7):
        super().__init__()
        self.offset_emb = nn.Embedding(64, d_model)  # max 64 chunk offsets
        self.pred = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, m * action_dim)
        )
        self.m = m
        self.action_dim = action_dim
    
    def forward(self, z_hist, offsets):
        """
        z_hist: (B, d_model)
        offsets: (B, Q) — chunk-level offsets from extra-window END
                 ALL positions guaranteed to be in extra-window (< τ-L)
        """
        B, Q = offsets.shape
        q = self.offset_emb(offsets)                    # (B, Q, d_model)
        z_exp = z_hist.unsqueeze(1).expand(-1, Q, -1)   # (B, Q, d_model)
        pred = self.pred(torch.cat([z_exp, q], dim=-1))  # (B, Q, m*Da)
        return pred.view(B, Q, self.m, self.action_dim)
```

**Query采样策略** (data loading时):

```python
def sample_extra_window_queries(num_chunks_K, num_queries=4):
    """
    Sample query offsets covering both recent and distant extra-window.
    Δ=0 is most recent extra-window chunk, Δ=K-1 is earliest.
    """
    if num_chunks_K < 1:
        return [], []  # no extra-window, skip QC-Remote-PTP
    
    if num_chunks_K <= num_queries:
        offsets = list(range(num_chunks_K))  # use all available
    else:
        # Spread evenly: e.g., K=20, Q=4 → Δ=[0, 6, 13, 19]
        offsets = [int(round(i * (num_chunks_K-1) / (num_queries-1))) 
                   for i in range(num_queries)]
    return offsets
```

### Training Plan
[Same as Round 3 — Two-stage, LDP-validated]

### Training Data Construction

```python
# In SequenceSampler, for each sample at decision moment τ:

extra_window_end = max(0, τ - L)  # L=32

# Chunk-level pooling of extra-window history:
num_chunks_K = extra_window_end // C  # C=8
obs_emb_chunks = []   # (K, obs_emb_dim)
action_chunks = []     # (K, action_dim)
for k in range(num_chunks_K):
    start = k * C
    end = start + C
    obs_emb_chunks.append(cached_obs_embs[start:end].mean(dim=0))
    action_chunks.append(episode_actions[start:end].mean(dim=0))

# QC-Remote-PTP targets:
query_offsets = sample_extra_window_queries(num_chunks_K, num_queries=4)
remote_targets = []
for delta in query_offsets:
    chunk_idx = num_chunks_K - 1 - delta
    action_start = chunk_idx * C
    target = episode_actions[action_start : action_start + m]  # m=4 consecutive actions
    remote_targets.append(target)

sample = {
    **existing_cdp_sample,
    'extra_obs_emb_chunks': pad(obs_emb_chunks),    # (K_max, obs_emb_dim)
    'extra_action_chunks': pad(action_chunks),        # (K_max, action_dim)
    'extra_num_chunks': num_chunks_K,
    'remote_query_offsets': pad(query_offsets),        # (Q,)
    'remote_targets': pad(remote_targets),             # (Q, m, Da)
    'remote_valid': [True] * len(query_offsets) + [False] * (4 - len(query_offsets)),
}
```

### Failure Modes

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Mamba state → constant | z_hist variance across τ → 0 | Increase Mamba layers/d_state |
| QC-Remote-PTP → trivial | All Δ offsets give same loss | Widen query spread; verify different Δ produce different preds |
| Stage transition gap | Stage 2 bc_loss >> Stage 1 end | Verify cached embeddings match checkpoint |
| Exposure bias | Late-episode performance collapse | Add noise to Mamba input during training |
| Early-episode (no extra-window) | z_hist=0 hurts performance | Zero-init AdaLN-Zero handles this (no effect when z_hist=0) |

### Novelty

**Paper narrative**: CDP的local buffer负责短期因果动作连续性，HistoryMamba负责extra-window memory，QC-Remote-PTP逼模型真正使用这段memory。

vs CDP: No extra-window memory, no utilization supervision
vs LDP: O(T²) full-attn obs-only, local PTP; we: O(n) Mamba obs+action, QC-Remote-PTP strictly extra-window
vs MTIL: Pure L2 no diffusion no supervision; we: keep diffusion + add utilization supervision

## Validation
[Same structure as Round 3]
Main table: CDP, ADP-obs-only, ADP-full, CDP-L100 × (3 long + 1 short tasks) × 3 seeds
PTP necessity: ADP-full vs ADP-w/o-PTP
History intervention: truncate/shuffle/replace
Inference cost: ms/step, GPU memory

## Compute
8× 48GB, ~10 days must-run
