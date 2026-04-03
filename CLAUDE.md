# AGENTS.md — CDP (Causal Diffusion Policy)

## Guidelines

### General Preferences
- **Language**: Always respond in Chinese, but provide key technical terms in English parentheses (e.g., 6D姿态估计 (6D Pose Estimation), 扩散模型 (Diffusion Model)).
- **Opening**: Start every response with "Yes, Sir~".
- **Style**:
  - You are my most candid advisor. Actively challenge my assumptions, question my reasoning, and speak up directly when something is off. For every conclusion I draw, scrutinize it for logical flaws, loopholes, self-consolation, rationalizations, wishful thinking, and risks I'm underestimating. No formalities, no sugarcoating, no going along to get along, and definitely no ambiguous fluff. Your advice must be fact-based, with clear reasoning, evidence, strategy, and concrete actionable steps. Prioritize my growth over my immediate comfort. Read between the lines—understand what I'm not saying rather than just my literal words. If you have a more sound judgment, stand by it. Be brutally honest with me. Hold nothing back.
  - When working on tasks, be concise, professional and academic, and avoid fluff.
  - When explaining issues and answering my questions, use plain and easy-to-understand language.
- **Implementation Decisions**: For any critical or ambiguous implementation details, always ask for clarification instead of making assumptions.
- **Environment**: Use the **cdp** Miniconda environment; ensure all operations and command executions are performed within this environment. Use `conda run --no-capture-output -n cdp <command>` to run commands.
- **Proxy**: 下载国外资源时，优先用国内源下载，如果国内源还是太慢，则换为先设置代理，再执行下载命令  
  ```bash
  unset http_proxy https_proxy no_proxy
  export http_proxy=http://192.168.10.134:7897
  export https_proxy=http://192.168.10.134:7897
  export no_proxy=localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8  # 内网不走代理
  ```

## Project Identity
CDP: transformer-based diffusion model for robot visuomotor policy learning, conditioning on historical action sequences (TEDi-style causal conditioning). Built on top of [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy).

Paper: "CDP: Towards Robust Autoregressive Visuomotor Policy Learning via Causal Diffusion" (arXiv:2506.14769)

## Repository Layout
```
.                                   # outer wrapper repo
├── Causal-Diffusion-Policy/        # inner project (installable package)
│   ├── train.py                    # Hydra entry point — TrainDP3Workspace
│   ├── setup.py                    # pip install -e .
│   └── diffusion_policy/           # core library (see child AGENTS.md)
├── scripts/                        # CLI wrappers (train_policy.sh, gen_demonstration_*.sh)
├── third_party/                    # 7 vendored deps (gym-0.21.0, mujoco-py, Metaworld, dexart, mj_envs, mjrl, pytorch3d_simplified)
├── visualizer/                     # separate pip-installable visualization package
├── debug/                          # debug notebooks/scripts
└── notes/                          # research notes
```
**Nested repo**: the actual code lives in `Causal-Diffusion-Policy/`, not root. `train.py` and `diffusion_policy/` are inside the inner directory.

## 4 Policy Variants
| Policy | File | Backbone | Observation | Scheduler |
|--------|------|----------|-------------|-----------|
| DP2 | `policy/dp2.py` | ConditionalUnet1D | Multi-image (RGB) | DDPMScheduler |
| DP3 | `policy/dp3.py` | ConditionalUnet1D | Point cloud | DDPMScheduler |
| CDP2 | `policy/cdp2.py` | CausalTransformer | Multi-image (RGB) | DDPMTEDiScheduler |
| CDP3 | `policy/cdp3.py` | CausalTransformer | Point cloud | DDPMTEDiScheduler |

All export class `DP`. Dynamically imported via `_POLICY_MAP` dict in `train.py`. CDP variants add: action buffer, chunked causal masking, KV cache inference.

## Config System
- **Framework**: Hydra 1.2.0 + OmegaConf
- **Algorithm configs**: `config/{dp2,dp3,cdp2,cdp3}.yaml` — set `policy._target_` and hyperparams
- **Task configs**: `config/task/*.yaml` — 61 tasks across 4 environments (Adroit/DexArt/MetaWorld/RealDex)
- **Composition**: algorithm config `defaults:` imports task config; Hydra `instantiate()` builds objects — no explicit registry
- **Override pattern**: `python train.py --config-name=cdp3 task=adroit_hammer ...`

## Data Pipeline
`zarr`-backed `ReplayBuffer` → `SequenceSampler` (episode-boundary-aware, pads at edges) → Dataset → `LinearNormalizer` (fit on data stats) → Policy

Key files: `common/replay_buffer.py`, `common/sampler.py`, `model/common/normalizer.py`

## Training Flow (`train.py`)
1. Hydra resolves config → `TrainDP3Workspace.__init__()` instantiates dataset, policy, optimizer, EMA
2. `run()`: training loop with `num_epochs` × `num_train_steps` gradient steps
3. Each step: sample batch → `policy.compute_loss()` → backward → optimizer step → EMA update
4. Periodic: validation rollout via `env_runner.run()` → log to wandb → checkpoint

## Environments & Runners
| Environment | Wrapper | Runner | Third-party |
|-------------|---------|--------|-------------|
| Adroit (MuJoCo/mjrl) | `env/adroit/adroit.py` | `adroit_runner.py` | `mj_envs`, `mjrl`, `gym-0.21.0` |
| DexArt | `env/dexart/dexart_wrapper.py` | `dexart_runner.py` | `dexart-release` |
| MetaWorld | `env/metaworld/metaworld_wrapper.py` | `metaworld_runner.py` | `Metaworld` |
| RealDex | N/A | `realdex_runner.py` | N/A |

Runners wrap envs with `MultiStepWrapper` + `SimpleVideoRecordingWrapper`. Evaluation uses `LargestKRecorder` for top-K success rate averaging.

## Shared Utilities (high-reuse across modules)
- `common/pytorch_util.py`: `dict_apply()`, `replace_submodules()`, `optimizer_to()`
- `model/common/normalizer.py`: `LinearNormalizer`, `SingleFieldLinearNormalizer`
- `model/common/module_attr_mixin.py`: device/dtype propagation mixin
- `common/normalize_util.py`: `get_image_range_normalizer()`
- `common/logger_util.py`: `LargestKRecorder` for evaluation metrics

## Commands
```bash
# Training (always use scripts wrapper)
bash scripts/train_policy.sh <algorithm> <task> <exp_name> <seed> <gpu_id>
# Example
bash scripts/train_policy.sh cdp3 adroit_hammer exp01 0 0

# Demo generation
bash scripts/gen_demonstration_adroit.sh hammer

# Direct train.py (advanced)
conda run --no-capture-output -n cdp python train.py --config-name=cdp3 task=adroit_hammer ...
```

## Key Conventions
- All policies export class named `DP` (not the algorithm name) — `_POLICY_MAP` in `train.py` maps string keys to modules
- `n_obs_steps`, `n_action_steps`, `horizon` define the temporal window: observe `n_obs_steps` frames, predict `horizon` actions, execute `n_action_steps`
- CDP adds `n_action_token` (action buffer length) and `n_diffusion_steps_per_token` (per-token denoising budget)
- Point cloud observations: 1024 points, processed by DP3Encoder (PointNet-based)
- Image observations: processed by `MultiImageObsEncoder` (ResNet18 backbone + spatial softmax + optional crop randomization)
- Checkpoints saved to `data/outputs/` with wandb logging
- No test suite exists; validation is rollout-based (env_runner success rate)

## Installation
See `INSTALL.md`. Key: `conda create -n cdp python=3.8`, then pip install torch + project deps + `pip install -e .` in inner dir + install all 7 third_party packages.
