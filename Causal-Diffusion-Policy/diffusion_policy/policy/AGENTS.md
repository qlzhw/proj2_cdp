# AGENTS.md — policy/ (Policy Implementations)

## Architecture
All 4 policies inherit `BasePolicy` (→ `ModuleAttrMixin` for device/dtype propagation) and export class named `DP`.

Two families:
- **DP variants** (dp2, dp3): standard DDPM diffusion with `ConditionalUnet1D` backbone
- **CDP variants** (cdp2, cdp3): causal diffusion with `CausalTransformer` backbone + action buffer

## Shared Interface
Every policy implements:
- `compute_loss(batch)` → scalar loss (training)
- `predict_action(obs_dict)` → `{action, action_pred}` dict (inference)
- `set_normalizer(normalizer)` → registers `LinearNormalizer` for obs/action scaling
- `get_optimizer(...)` → separate param groups for visual encoder (lower lr) vs rest

## DP2/DP3 (Standard Diffusion)
- `compute_loss()`: encode obs → sample noise/timestep → predict noise via UNet → MSE loss
- `predict_action()`: encode obs → DDPM reverse loop (`scheduler.step()`) for `num_inference_steps` → unnormalize
- Condition: `global_cond` = flattened encoded observations (all `n_obs_steps` frames)
- Full `horizon`-length trajectory predicted and denoised in one pass

## CDP2/CDP3 (Causal Diffusion — this project's contribution)
Key additions over DP:
1. **Action buffer** (`n_action_token`): maintains history of previously executed action chunks
2. **Chunked causal masking**: `horizon` split into `n_action_token` chunks; each chunk attends only to itself + prior chunks
3. **Per-token diffusion**: each action token can have independent noise level; `DDPMTEDiScheduler` manages per-token timesteps
4. **KV cache inference**: `conditional_sample_kvcache()` — only new tokens go through transformer, prior KV reused

Training flow (`compute_loss`):
1. Encode obs → obs conditioning
2. Concatenate action buffer (clean, from prior steps) with noised current actions
3. Build causal mask via `MaskGenerator` (chunk-level causal + observation-to-all)
4. Transformer forward → predict noise for current tokens only
5. MSE loss on noise prediction (only on non-buffer tokens)

Inference flow (`predict_action`):
1. Encode obs, prepare action buffer from history
2. Reverse diffusion with KV cache: each step only processes current noised tokens
3. After denoising: execute `n_action_steps`, shift buffer, cache KV state

## Critical Parameters
| Param | Scope | Meaning |
|-------|-------|---------|
| `horizon` | all | total action prediction length |
| `n_obs_steps` | all | observation context window |
| `n_action_steps` | all | actions actually executed per step |
| `num_inference_steps` | all | DDPM reverse steps |
| `n_action_token` | CDP only | action buffer size (prior action chunks) |
| `n_diffusion_steps_per_token` | CDP only | per-token denoising budget |

## File-Specific Notes
- `base_policy.py`: abstract base, no logic — just interface + `ModuleAttrMixin`
- `dp2.py` / `dp3.py`: nearly identical structure, differ only in observation encoder
- `cdp2.py` / `cdp3.py`: nearly identical structure, differ only in observation encoder
- `__init__.py`: empty
