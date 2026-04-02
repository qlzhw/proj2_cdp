# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Guidelines

### General Preferences
- **Language**: Always respond in Chinese, but provide key technical terms in English parentheses (e.g., 6D姿态估计 (6D Pose Estimation), 扩散模型 (Diffusion Model)).
- **Opening**: Start every response with "Yes, Sir~".
- **Style**:
  - When working on tasks, be concise, professional and academic, and avoid fluff.
  - When explaining issues and answering my questions, use plain and easy-to-understand language.
- **Implementation Decisions**: For any critical or ambiguous implementation details, always ask for clarification instead of making assumptions.
- **Environment**: Use the **dp-adroit** Miniconda environment; ensure all operations and command executions are performed within this environment. Use `conda run --no-capture-output -n dp-adroit <command>` to run commands.
- **Proxy**: 下载国外资源时，优先用国内源下载，如果国内源还是太慢，则换为先设置代理，再执行下载命令  
  ```bash
  unset http_proxy https_proxy no_proxy
  export http_proxy=http://192.168.10.134:7897
  export https_proxy=http://192.168.10.134:7897
  export no_proxy=localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8  # 内网不走代理
  ```


## Architecture Overview

Fork of [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) + custom **EnergyNet / Generate & Rank** pipeline. O(N+M) task-method separation:
- **Task side**: `Dataset` + `EnvRunner` + `config/task/<task_name>.yaml`
- **Method side**: `Policy` + `Workspace` + `config/<workspace_name>.yaml`

### Key Abstractions
| Component | Base Class | Role |
|-----------|-----------|------|
| **Workspace** | `workspace/base_workspace.py` | Training lifecycle: init, train loop, checkpointing, eval |
| **Policy** | `policy/base_image_policy.py` / `base_lowdim_policy.py` | Inference (`predict_action`) + loss (`compute_loss`) + normalization |
| **Dataset** | `torch.utils.data.Dataset` | Returns `{obs, action}` dicts; provides `get_normalizer()` |
| **EnvRunner** | — | Runs policy in env, returns logs/metrics compatible with `wandb.log` |

### Observation/Action Interface
- **Observation Horizon** (`To` / `n_obs_steps`), **Action Horizon** (`Ta` / `n_action_steps`), **Prediction Horizon** (`T` / `horizon`)

### Multi-Stage Pipeline
1. **Stage-1 (ScoreNet)**: Standard diffusion training (`agent_type='score'`)
2. **Stage-2 (EnergyNet)**: DSM score matching (`agent_type='energy'`), freezes `obs_encoder`
3. **Stage-3 (Eval)**: Online Generate & Rank via `eval.py --use_energy_ranking`


---

# DEEP KNOWLEDGE BASE

**Generated:** 2026-03-24 | **Commit:** `3eb6938` | **Branch:** `3.23-adaptive-energy`

## STRUCTURE MAP
```
.
├── diffusion_policy/          # Core library → see diffusion_policy/AGENTS.md
│   ├── policy/                # 13 files — Policy classes (dual-mode score/energy/ranker)
│   ├── model/                 # 5 subdirs — UNet, Transformer, EnergyNet, BET, vision encoders
│   │   ├── energy/            # TrajectoryEnergyNet (ranker network)
│   │   ├── diffusion/         # ConditionalUnet1D, TransformerForDiffusion, schedulers
│   │   ├── common/            # Normalizer, LR scheduler, tensor utils, DictOfTensorMixin
│   │   ├── vision/            # ResNet obs encoder, multi-image obs encoder
│   │   └── bet/               # BET baseline (MinGPT-based)
│   ├── workspace/             # 12 files — Training orchestrators per method
│   ├── dataset/               # 11 files — Dataset adapters + RankerCacheDataset
│   ├── config/                # Hydra configs: 22 workspace YAMLs + task/ subdir (20 tasks)
│   ├── env_runner/            # 9 files — Vectorized env evaluation
│   ├── env/                   # 4 envs: pusht, kitchen, block_pushing, robomimic
│   ├── common/                # 16 files — replay_buffer, sampler, normalizer, pytorch_util
│   ├── real_world/            # 11 files — UR5 + RealSense + SpaceMouse real robot
│   ├── shared_memory/         # Lock-free ring buffers for real-time multi-camera
│   ├── add_src/               # DBSCAN eps sweep utility
│   └── scripts/               # Data conversion, real robot scripts
├── debug/                     # Experiment pipelines by version → see debug/AGENTS.md
├── notes/                     # Dev plans (3.8-3.10), analysis, usage guides
├── tests/                     # 13 pytest files
├── train.py                   # @hydra.main → Workspace._target_ → workspace.run()
├── eval.py                    # Click CLI → load ckpt → optional G&R → env_runner.run()
└── eval_real_robot.py         # Real robot eval with UR5
```

## DISPATCH FLOW

### train.py
```
@hydra.main(config_path="diffusion_policy/config")
  → OmegaConf.register_new_resolver("eval", eval)
  → cls = hydra.utils.get_class(cfg._target_)
  → workspace = cls(cfg)
  → workspace.run()
```

### eval.py
```
Click CLI args: -c checkpoint, -o output_dir, -d device
  → payload = torch.load(checkpoint, pickle_module=dill)
  → cfg = payload["cfg"]
  → workspace = cls(cfg); workspace.load_payload(payload)
  → policy = workspace.model (or ema_model)
  → if --use_energy_ranking: patch policy.use_energy_ranking/num_candidates/etc
  → if --ranker_ckpt_path: policy.init_ranker_net(**arch), agent_type="ranker"
  → env_runner = hydra.utils.instantiate(cfg.task.env_runner)
  → runner_log = env_runner.run(policy)
```

## CONFIG ROUTING (Hydra)

```
train.py --config-name=train_diffusion_unet_image_workspace task=pusht_image
         │                                                    │
         ▼                                                    ▼
config/train_diffusion_unet_image_workspace.yaml    config/task/pusht_image.yaml
  _target_: ...TrainDiffusionUnetImageWorkspace        name: pusht_image
  policy:                                              shape_meta: {obs: ..., action: ...}
    _target_: ...DiffusionUnetImagePolicy              dataset:
    shape_meta: ${task.shape_meta}                       _target_: ...PushTImageDataset
    agent_type: score                                  env_runner:
    obs_encoder: ...                                     _target_: ...PushTImageRunner
```

No explicit registry — `_target_` + `hydra.utils.instantiate()` is the discovery mechanism.

## ANTI-PATTERNS & GOTCHAS

| Rule | Detail | Location |
|------|--------|----------|
| **NEVER unfreeze obs_encoder in energy/ranker** | `freeze_encoder_for_stage2=True` → `.eval()` + `requires_grad_(False)`. Optimizer must exclude frozen params. | `policy/diffusion_unet_image_policy.py` |
| **energy mode requires `obs_as_global_cond=True`** | Code raises NotImplementedError otherwise | `policy/diffusion_unet_image_policy.py` |
| **t_eval must be `torch.long`** | Float dtype breaks noise scheduler embeddings | `workspace/train_diffusion_unet_image_workspace.py` |
| **DictOfTensorMixin is frozen** | `params_dict.requires_grad_(False)` after load. Not trainable. | `model/common/dict_of_tensor_mixin.py` |
| **AsyncVectorEnv fork + OpenGL** | Environments with OpenGL init segfault in forked children. Use `dummy_env_fn`. | `gym_util/async_vector_env.py` |
| **Action window alignment** | `start = n_obs_steps - 1; end = start + n_action_steps`. Must be consistent between cache scoring and eval. | Multiple files |
| **Normalizer bugs** | Print `scale`/`bias` vectors to debug. Normalization mismatch is the #1 silent failure mode. | `model/common/normalizer.py` |

## KEY FILES (by reference centrality)

| File | Lines | Role |
|------|-------|------|
| `policy/diffusion_unet_image_policy.py` | 824 | **Heart of the project.** Dual-mode policy: compute_loss (score/energy/ranker), predict_action (with/without G&R), forward_energy, _energy_rank_and_aggregate |
| `workspace/train_diffusion_unet_image_workspace.py` | 479 | Training orchestrator. Handles agent_type switching, encoder freezing, ranking validation, rollout skipping |
| `model/energy/trajectory_energy_net.py` | ~200 | Standalone ranker network. Conv1d trajectory encoder → scalar energy. Supports prefix_mask |
| `common/replay_buffer.py` | 590 | zarr-backed episode storage. `data/` + `meta/episode_ends`. Supports numpy/zarr backends |
| `common/sampler.py` | — | SequenceSampler: episode boundary padding for To/Ta. Critical for correct training |
| `model/common/normalizer.py` | — | LinearNormalizer: saved in checkpoint, applied on GPU. Keys must match shape_meta |
| `dataset/ranker_cache_dataset.py` | — | Loads merged.pt (candidates/global_cond/scores) for ranker training |
| `eval.py` | — | Click entry with G&R parameter patching. Writes eval_log.json + run_meta.json |
