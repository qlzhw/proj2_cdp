# AGENTS.md — model/diffusion/ (Backbone Networks & Schedulers)

## Files
| File | Role | Used by |
|------|------|---------|
| `causal_transformer.py` | CausalTransformer backbone | CDP2, CDP3 |
| `conditional_unet1d.py` | ConditionalUnet1D backbone | DP2, DP3 |
| `schedulers.py` | DDPMTEDiScheduler (per-token diffusion) | CDP2, CDP3 |
| `mask_generator.py` | MaskGenerator (chunked causal masks) | CDP2, CDP3 |
| `positional_embedding.py` | SinusoidalPosEmb | both UNet and Transformer |
| `conv1d_components.py` | Conv1dBlock, Downsample1d, Upsample1d | UNet |
| `cached_attention.py` | CachedMultiheadAttention | Transformer (KV cache) |
| `ema_model.py` | EMAModel (exponential moving average) | train.py |

## CausalTransformer (`causal_transformer.py`)
Core innovation of CDP. Architecture:
- Input: `(B, T, action_dim)` noised actions + obs conditioning + timestep embedding
- Chunked tokens: actions split into `n_action_token` chunks of `chunk_size = horizon // n_action_token`
- Each chunk flattened → linear projection → transformer token
- Observation tokens prepended (attend-to-all, no causal restriction)
- Standard transformer encoder layers with `CachedMultiheadAttention`
- Output: predicted noise, same shape as input actions

Key methods:
- `forward(sample, timestep, cond, attention_mask)` — full forward (training)
- `forward_kvcache(sample, timestep, cond, kv_cache)` — incremental forward (inference), only processes new tokens

Mask structure (built by `MaskGenerator`):
- Observation tokens: visible to all (no masking)
- Action chunk i: can attend to obs + chunks 0..i (causal within action tokens)
- This is CHUNK-level causal, not token-level — entire chunks are masked/unmasked together

## ConditionalUnet1D (`conditional_unet1d.py`)
Standard 1D U-Net for temporal diffusion (from Diffusion Policy):
- Input: `(B, action_dim, horizon)` noised trajectory
- `global_cond`: flattened observation features, injected via FiLM conditioning
- Encoder-decoder with skip connections, residual blocks, group norm
- Output: predicted noise, same shape

## DDPMTEDiScheduler (`schedulers.py`)
Extended DDPM scheduler supporting per-token independent timesteps (TEDi-style):
- Standard DDPM `add_noise()` / `step()` but timestep can be `(B, n_tokens)` shaped
- `set_timesteps()` configures per-token denoising schedule
- Used only by CDP variants; DP variants use standard `DDPMScheduler` from diffusers

## MaskGenerator (`mask_generator.py`)
Builds attention masks for CausalTransformer:
- `get_causal_mask(n_obs_tokens, n_action_tokens, chunk_size)` → `(seq_len, seq_len)` boolean mask
- Obs tokens row: all True (attend everywhere)
- Action chunk i row: True for obs + chunks ≤ i, False for chunks > i
- Supports variable number of obs tokens and action chunks
