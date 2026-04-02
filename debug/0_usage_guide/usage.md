### Training

1. To generate the demonstrations, run the appropriate `gen_demonstration_xxxxx.sh` script—check each script for specifics. For example:
   - **Adroit**: `conda run --no-capture-output -n cdp bash scripts/gen_demonstration_adroit.sh [hammer|door|pen]`
   - **Dexart**: `conda run --no-capture-output -n cdp bash scripts/gen_demonstration_dexart.sh [laptop|faucet|bucket|toilet]`
   - **Metaworld**: `conda run --no-capture-output -n cdp bash scripts/gen_demonstration_metaworld.sh [basketball|...]`
   ```bash
   conda run --no-capture-output -n cdp bash scripts/gen_demonstration_adroit.sh hammer
   ```
   This command collects demonstrations for the Adroit `pen` task and automatically stores them in `Causal-Diffusion-Policy/data/` folder.
2. To train and evaluate a policy, run the following command:

   命令格式为：
   `conda run --no-capture-output -n cdp bash scripts/train_policy.sh [算法名称] [任务名称] [附加信息到输出目录名中] [随机种子] [GPU编号]`
   ```bash
   conda run --no-capture-output -n cdp bash scripts/train_policy.sh dp2 adroit_pen 0402 0 3
   conda run --no-capture-output -n cdp bash scripts/train_policy.sh cdp2 adroit_pen 0402 0 4
   conda run --no-capture-output -n cdp bash scripts/train_policy.sh dp3 adroit_pen 0402 0 5
   conda run --no-capture-output -n cdp bash scripts/train_policy.sh cdp3 adroit_pen 0402 0 6
   ```
   These commands train a DP2, CDP2, DP3, CDP3 policy on the Adroit `pen` task, respectively.

