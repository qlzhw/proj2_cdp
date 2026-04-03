### Training

1. To generate the demonstrations, run the appropriate `gen_demonstration_xxxxx.sh` script—check each script for specifics. For example:
   - **Adroit**: `conda run --no-capture-output -n cdp bash scripts/gen_demonstration_adroit.sh [hammer|door|pen]`
   - **Dexart**: `conda run --no-capture-output -n cdp bash scripts/gen_demonstration_dexart.sh [laptop|faucet|bucket|toilet]`
   - **Metaworld**: `conda run --no-capture-output -n cdp bash scripts/gen_demonstration_metaworld.sh [basketball|...]`
   ```bash
   conda run --no-capture-output -n cdp bash scripts/gen_demonstration_adroit.sh pen
   ```
   This command collects demonstrations for the Adroit `pen` task and automatically stores them in `Causal-Diffusion-Policy/data/` folder.
2. To train and evaluate a policy, run the following command:

   命令格式为：
   `conda run --no-capture-output -n cdp bash scripts/train_policy.sh [算法名称] [任务名称] [附加信息到输出目录名中] [随机种子] [GPU编号]`
   screen -S 4.2pd2-pen-0
   ```bash
   conda run --no-capture-output -n cdp bash scripts/train_policy.sh dp2 adroit_pen 0402 2 2
   conda run --no-capture-output -n cdp bash scripts/train_policy.sh cdp2 adroit_pen 0402 1 6
   conda run --no-capture-output -n cdp bash scripts/train_policy.sh dp3 adroit_pen 0403 0 1
   conda run --no-capture-output -n cdp bash scripts/train_policy.sh cdp3 adroit_pen 0403 0 4
   ```
   These commands train a DP2, CDP2, DP3, CDP3 policy on the Adroit `pen` task, respectively.

3. Resume:
   ```bash
   conda run --no-capture-output -n cdp bash scripts/train_policy.sh [算法名称] [任务名称] [附加信息到输出目录名中] [随机种子] [GPU编号]
   conda run --no-capture-output -n cdp bash scripts/train_policy.sh cdp3 adroit_pen 0402 0 7
   ```
   Resume后会从 `data/outputs/<exp_name>_seed<seed>/checkpoints/latest.ckpt` 继续训练。
   如果已经训练到第 `N` 轮，则会继续训练到目标 `training.num_epochs`（默认 `3000`），而不是恢复后再额外训练一整轮 `3000` epochs。


测频率就保持50，我保存了log，训练完之后我手动计算；DexArt也暂时不改，我已经记录下来了
dexart目前待确认dp3
论文表写点云 512x3，但 DexArt 这边是：
生成脚本 gen_demonstration_dexart.sh 用 --num_points 1024
config/task/dexart_laptop.yaml 也是 [1024, 3]