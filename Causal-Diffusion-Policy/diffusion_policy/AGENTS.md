# AGENTS.md — diffusion_policy (Core Library)

## Module Map
```
diffusion_policy/
├── policy/          # 4 policy variants (see child AGENTS.md)
├── model/
│   ├── diffusion/   # backbone networks + schedulers (see child AGENTS.md)
│   ├── vision/      # observation encoders (PointNet, MultiImage, crop_randomizer)
│   └── common/      # normalizer, lr_scheduler, module_attr_mixin
├── dataset/         # per-environment datasets (adroit, dexart, metaworld, realdex)
├── env/             # environment wrappers (adroit, dexart, metaworld)
├── env_runner/      # rollout executors (one per environment)
├── gym_util/        # MultiStepWrapper, VideoRecordingWrapper, async vector env
├── common/          # replay_buffer, sampler, pytorch_util, checkpoint_util
└── config/          # Hydra YAML configs (algorithm + task)
```

## Cross-Module Dependencies
```
train.py → dataset/* + policy/* + env_runner/* + common/* + model/ema_model
dataset/* → ReplayBuffer + SequenceSampler + LinearNormalizer
policy/*  → model/diffusion/* + model/vision/* + model/common/* + common/pytorch_util
env_runner/* → env/* + gym_util/* + common/logger_util
```

## Dataset Pattern (all 4 are structurally identical)
Each `{env}_dataset.py`:
1. `__init__()`: loads zarr-backed `ReplayBuffer`, builds `SequenceSampler` with episode-boundary padding
2. `get_normalizer()`: fits `LinearNormalizer` on dataset statistics, applies `get_image_range_normalizer()` for images
3. `__getitem__()`: returns `{obs: {agent_pos, point_cloud/image}, action}` dict via sampler

Key params: `horizon`, `n_obs_steps`, `pad_before`, `pad_after`, `val_ratio`

## Environment Wrapper Pattern
Each wrapper: raw env → gym.Env interface with `obs_dict` containing `{agent_pos, point_cloud}` or `{agent_pos, image}`.
- Adroit: uses `mj_envs` + `mjrl` demos, point cloud from MuJoCo geom sampling
- MetaWorld: uses `metaworld` benchmark, point cloud from MuJoCo site sampling
- DexArt: uses `dexart` SAPIEN-based sim, point cloud from depth cameras

## Runner Pattern
All runners share: `MultiStepWrapper` (action repeat) → `SimpleVideoRecordingWrapper` → rollout loop → `LargestKRecorder` (top-K success rate). Return `log_data` dict with `mean_score`, `test/mean_score`, video paths.

## Config Resolution
- Algorithm YAML (`config/{dp2,dp3,cdp2,cdp3}.yaml`) sets `policy._target_`, `shape_meta`, temporal params
- Task YAML (`config/task/*.yaml`) sets `dataset._target_`, `env_runner._target_`, data paths
- Algorithm defaults import task via `defaults: [task: ...]`
- `hydra.utils.instantiate()` builds all objects — no manual registry

## Observation Encoders (`model/vision/`)
- `DP3Encoder` (`pointnet_extractor.py`): PointNet on 1024 points → 512-dim feature
- `MultiImageObsEncoder` (`multi_image_obs_encoder.py`): ResNet18 + spatial softmax + optional `CropRandomizer`
- Encoder choice is implicit in policy: DP3/CDP3 use DP3Encoder, DP2/CDP2 use MultiImageObsEncoder

## Normalizer System (`model/common/normalizer.py`)
`LinearNormalizer` wraps per-key `SingleFieldLinearNormalizer`. Two modes:
- `normalize()` / `unnormalize()`: scale data to [-1, 1] range using fitted min/max
- Fitted once via `fit()` on training data statistics, then frozen
- Actions always normalized; observations depend on type (agent_pos: normalize, images: range normalizer [0,1]→[-1,1])
