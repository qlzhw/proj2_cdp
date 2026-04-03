目前仓库的cdp是在dp3的基础上改进而来，而cdp是我的baseline，目前cdp方法是引入固定窗口大小的历史动作，并且还有一些tricks，我准备改进为adp，a代表all history，也就是要引入所有历史动作，而非现在cdp固定步数的历史动作，目的是，通过足够的历史信息，提升模型完成任务的成功率。
我找了4篇相关论文，你需要详细学习论文以及对应的github仓库并理解其核心思想和实现方法，提取其中可以借鉴的部分：
https://www.alphaxiv.org/abs/2209.04899
https://www.alphaxiv.org/abs/2505.09561v2
https://www.alphaxiv.org/abs/2506.14769
https://www.alphaxiv.org/abs/2603.14498

除此之外，你需要自由探索新的论文和github仓库，寻找可以借鉴的部分。

引入所有历史动作只是我的想法，如果你自己深入调研完有更好的思路或者想法【特别是分析Learning Long-Context Diffusion Policies via Past-Token Prediction能否与cdp或者全部历史动作结合】，可以与我讨论，我们的终极目标是提升模型完成任务的成功率。

注意：目前可以通过dp2,cdp2,dp3,cdp3在训练的时候区分policy,例如
scripts/train_policy.sh dp2 adroit_pen 0402 0 3 和
scripts/train_policy.sh cdp2 adroit_pen 0402 0 4
希望维持现在的4个算法不变，新改动的算法命名为adp2,adp3

针对这个改动，我有一些思考：
①我的任务大多在300步以内，执行到后半段的时候，前面的动作对当前决策的价值极低，反而会引入大量无关噪声，导致模型产生虚假相关性并严重过拟合。这个可以通过对历史动作进行加权来解决，越近的动作权重越大，越远的动作权重越小。也可以定期把太久远的动作融合为一个整体信息，降低维度，避免前面的动作信息占比太大（这个点R3DP中有类似做法）。
②关于历史动作，训练的时候可以参考cdp为历史动作加入小噪声、Instruction-driven history-aware policies for robotic manipulations直接对历史信息做小部分mask等等，来提高模型能力。
③目前cdp基于Transformer，其注意力机制的计算复杂度是序列长度的平方。如果Episod太长，不利于推理延迟与显存，特别是推理延迟，会彻底摧毁机器人的实时闭环控制能力。我们不能做没有使用价值的模型，需要综合考虑和比较其他方式，例如flash attention、Cross-Attention、GRU等。（关于这一点，cdp引入了kv cache机制，不过好像不能直接应用到adp，不过是有利的参考）
④关于所有历史动作，如果放到大语言模型领域，引入所有历史聊天记录早已成熟，甚至现在有一百万上下文窗口的模型，所以在技术上应该是可以实现的，其中Instruction-driven history-aware policies for robotic manipulations就是引入了所有历史观测和动作。
⑤我目前想的是引入所有历史动作，但是ldp中引入的是历史观测，这与我的方法不同，并且ldp引入历史观测，同时预测过去动作和未来动作，再通过多组候选，将预测过去动作与实际过去动作计算mse，选择最小mse的那组候选执行，这个方法和巧妙，考虑能不能和我们的adp结合。截止目前，有历史信息只包括定长历史动作，有只包括定长历史观测，也有包括所有历史观测和动作，你要合理分析。
⑥一些其他的小trick，你要从我给你的论文、github repo以及自有探索发现以及利用！