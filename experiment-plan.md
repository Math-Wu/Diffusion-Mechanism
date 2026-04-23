# Experiment Plan: Sampler × Architecture in Diffusion Models

## 0. 目标与核心问题

### 0.1 研究目标
研究 **diffusion sampler 的离散化误差** 如何与 **不同 backbone 学到的 reverse field / denoiser 的数值性质与归纳偏置** 耦合，并回答：

1. 不同 architecture 是否存在稳定的 **solver preference**？
2. 这种 preference 是否能用 learned field 的 **smoothness / stiffness / temporal redundancy / phase structure / frequency difficulty** 来解释？
3. 最终 sample quality 的下降，是否主要由 **solver 误差是否落在 architecture 的脆弱时段与脆弱频段** 决定，而不是由 solver 总误差决定？

### 0.2 一句话题眼
这不是“哪个 sampler 更好”的工程 benchmark，而是：

> **不同 backbone 学到的 denoising field 的数值特性不同，从而导致相同 solver 的离散化误差在不同 architecture 上呈现不同的时域与频域落点，最终造成 architecture-specific solver compatibility。**

### 0.3 主假设

- **H1: Solver ranking depends on architecture.**
  在 few-step 区间（尤其 4–20 NFE）中，best solver family 会随 backbone 改变。
- **H2: Higher-order / multistep gains require smoother and more redundant temporal dynamics.**
  若某 backbone 在相邻 timestep 上输出更平滑、更冗余，则 UniPC / DPM-Solver++ 类方法更有优势。
- **H3: Quality degradation is governed more by error overlap than total error.**
  真正决定 sample quality 的，不是 solver 总误差，而是 solver 误差是否刚好落在 architecture 本来就难的 `t × frequency` 区域。
- **H4: Optimal solver may be phase-dependent, and the phase boundary may shift across architectures.**
  早期、晚期的最优 solver 可能不同，且切换点依赖 backbone。

---

## 1. 实验哲学：如何把问题做“干净”

### 1.1 主体实验采用“两条线”

#### A. Controlled-Matched Track（正文主线）
用于回答科学问题。

原则：
- 同一数据集
- 同一训练目标与噪声日程
- 同一 conditioning 方式
- 同一训练协议
- 相近参数量
- 相近单步 FLOPs / MACs
- 相同采样口径与评测协议

只允许 **architecture macro-structure**（U-Net / U-ViT / DiT）不同。

#### B. Canonical Track（附录/补充）
用于证明结论不依赖于你手工匹配出来的某一组模型。

原则：
- 使用各 family 较常见或较 canonical 的设定
- 不强求精确参数匹配
- 重点看“规律是否保留”而不是绝对数值

**结论**：
- **正文以 matched models 为主**，因为你要研究的是 architecture 本身，不是“谁堆了更多容量”。
- **附录再放 canonical models**，做外部有效性验证。

### 1.2 参数量是否要相近？
**要，但不能只看参数量。必须同时控制单步计算量。**

建议标准：
- 参数量差距控制在 **±10%–15%** 内
- 单步 FLOPs / MACs / wall-clock 差距控制在 **±15%–20%** 内
- 对 transformer 家族，**patch size 和 token count 尽量固定一致**（至少 U-ViT 和 DiT 必须一样）

原因：
- U-ViT 论文直接把 U-ViT 与“similar-size U-Net”比较，强调 long skip 的作用而非单纯容量差异。
- DiT 论文指出其 scaling 更适合用 **GFLOPs** 来衡量，而不是只看参数量。

所以最合理的做法是：

> **正文匹配 parameters + per-step compute；附录提供 canonical setting。**

### 1.3 不要做“大规模 architecture search”
不要在主实验里对每个 family 都做大规模架构搜索，否则你研究的就不是 architecture family，而是“谁被搜得更好”。

**允许的有限调节**：
- 只调 **width / base channels / hidden size** 来对齐预算
- 尽量固定以下内容：
  - U-Net 的 stage 数
  - U-Net 每层 resblock 数
  - U-ViT / DiT 的层数（depth）
  - U-ViT / DiT 的 patch size
  - attention head dimension
  - MLP ratio

**推荐策略**：
- 先定一个 budget
- 只在一维网格上调 width，使参数量和 FLOPs 落入预算窗口
- 选“最接近预算”的模型，而不是“最佳 FID 的模型”

---

## 2. 直接回答你的几个关键问题

### 2.1 如何选择 backbone 架构设定？

#### 主线必须用 3 类 backbone

1. **CNN U-Net**
   - 代表卷积、多尺度、显式下采样/上采样、skip connection
2. **U-ViT**
   - 代表 transformer backbone + long skip
   - 是 U-Net 与 DiT 之间最关键的桥梁架构
3. **DiT**
   - 代表 isotropic transformer，无 U-shape，多层同形

这样才能拆解：
- 差异来自 **token mixing / attention**？
- 还是来自 **U-shape / long skip**？
- 还是来自 **局部性 / 频谱偏置 / temporal redundancy**？

#### 主线先做“预算匹配版 v1”

原则：
- **先用 CIFAR pilot 确定候选网格，再冻结 1 组 main config**
- **保持 token count 对齐，放松“U-ViT 与 DiT 必须同 depth”**
- **优先匹配 per-step compute，其次匹配参数量**

##### CIFAR-10（32×32, pixel-space）受控设定
目标预算：**~35M–50M 参数，单步 compute 尽量接近**

- **U-Net**
  - 4 个 resolution stages
  - `num_res_blocks = 2`
  - attention 放在低分辨率层（例如 16×16, 8×8）
  - `base_channels` 从 `{128, 160, 192}` 中选，使预算最接近目标
- **U-ViT**
  - patch size = 2
  - token count 固定为 `16 x 16 = 256`
  - depth 从 `{10, 12, 14}` 中选
  - hidden size 从 `{384, 512, 640}` 中选
  - heads = `hidden_size / 64`（向下取整到整数）
  - MLP ratio = 4
  - 保留对称 long skips
- **DiT**
  - patch size = 2（与 U-ViT 保持一致）
  - token count 同样固定为 `256`
  - depth 从 `{10, 12, 14}` 中选
  - hidden size 从 `{320, 384, 512, 640}` 中选
  - heads、MLP ratio 与 U-ViT 保持同规则
  - **不要求与 U-ViT 完全同 depth，只要求落在同一 compute window**

##### ImageNet-64（64×64, pixel-space）受控设定
目标预算（v1）：**~70M–100M 参数，单步 compute 尽量接近**

- **U-Net**
  - 4 个 resolution stages
  - `num_res_blocks = 2`
  - attention 放在 32×32 / 16×16 / 8×8（根据预算做轻微删减）
  - `base_channels` 从 `{160, 192, 224}` 中选
- **U-ViT**
  - patch size = 4
  - token count 固定为 `16 x 16 = 256`
  - depth 从 `{12, 16}` 中选
  - hidden size 从 `{512, 640, 768}` 中选
  - heads = `hidden_size / 64`
  - MLP ratio = 4
  - 保留对称 long skips
- **DiT**
  - patch size = 4（与 U-ViT 保持一致）
  - token count 同样固定为 `256`
  - depth 从 `{12, 14, 16}` 中选
  - hidden size 从 `{512, 640, 768}` 中选
  - **同样不强求与 U-ViT 同 depth；若 depth 不同，只要 GFLOPs / wall-clock 落入窗口即可**

##### 可选扩展：ImageNet-256 latent（附录/外部验证）
- 所有 backbone 共用同一个 VAE
- latent size 通常是 32×32
- U-ViT / DiT 统一 patch size = 2
- 不作为正文主线，因为 VAE 会引入额外变量

#### 一个很重要的控制原则

> **不要在主实验里同时让 architecture depth、patch size、token 数、训练 recipe 都自由变化。**

否则你最后看到的不是 architecture 差异，而是“一揽子系统差异”。

补充说明：
- U-Net / U-ViT / DiT 这三个 family **不建议改**，这是最干净的主线组合
- 真正需要 pilot 的，不是“是否换 family”，而是 **每个 family 在统一预算下的具体宽度/深度配置**

---

### 2.2 训练时是控制相同步数，还是 loss 到某个阈值？

#### 不建议：按 train loss/val loss 设统一停止线
原因：
- diffusion loss 与最终 sample quality 并非严格单调对齐
- 不同 backbone 的 loss 标度和收敛速度可能不同
- 同样的 loss 数值并不代表“学到了同样程度的 generative competence”

#### 也不建议：看到某个模型先达到最低 FID 就立刻停
原因：
- 容易让不同 architecture 停在不同训练成熟度
- 可能把“谁更早收敛”混进“谁与 solver 更兼容”

#### 推荐：**固定最大训练预算 + 统一 checkpoint 选择规则**

主线做法：
1. 给所有模型相同的 **最大训练 budget**（更推荐用“images seen”而不是纯 step 数）
2. 定期存 checkpoint
3. 用同一个 **高精度 reference sampler** 做小规模验证生成
4. 用统一规则选 checkpoint

**推荐 checkpoint 选择规则**：
- 主规则：选择该架构在统一 reference sampler 下 **validation/proxy FID 最优** 的 checkpoint
- 辅助规则：训练必须跑到预设 budget，避免某架构过早停止

#### 最终建议
- **主实验训练到相同最大 budget，不提前停。**
- **最终使用“最佳 validation generative checkpoint”做采样比较。**
- 另外做一个 **matched-step checkpoint ablation** 放附录，证明结论对 checkpoint 选择不敏感。

### 2.3 训练收敛的指标是什么？
建议同时跟踪 3 类指标：

#### A. Optimization diagnostics（仅用于监控，不作为最终 stopping rule）
- train epsilon-MSE
- validation epsilon-MSE
- gradient norm
- EMA 与 non-EMA 的 loss gap

#### B. Generative diagnostics（用于 checkpoint selection）
- **proxy FID**（推荐 10k samples）
- 可选：precision / recall
- 可选：IS（仅 CIFAR，辅助用，不作为核心指标）

#### C. “真的收敛了吗”的判断
- validation loss 基本平台化
- proxy FID 在连续 3–4 次评测中无显著提升
- doubling budget 后最终 full FID 几乎不变

#### 实操建议
- **训练中每隔固定 budget** 做一次 10k-sample proxy FID
- 最终候选 checkpoint 再做 50k-sample full FID
- 对小数据集（CIFAR）多种子；对大数据集（ImageNet64）可少一些种子但加强 paired evaluation

---

### 2.3补充：seed 策略（必须单独写清）

这里至少要区分 3 类 seed：

1. **split seed**
   - 控制 train / validation 的划分
   - 一旦划定，所有 backbone、所有训练重复都共用同一个 split
2. **training seed**
   - 控制参数初始化、数据顺序、训练时采样的噪声与 timestep 等随机性
   - `U-Net x 3 seeds` 的含义，就是同一个 U-Net 配置独立训练 3 次，而不是 3 个不同 U-Net
3. **sampling seed / noise bank seed**
   - 控制生成时的初始噪声
   - solver 对比时应固定同一组 noise bank，做 paired comparison

#### 推荐 seed 策略（v1）

- **Pilot 阶段**：每个 backbone 先只跑 `1 training seed`
- **CIFAR 主结果**：先跑 `2 training seeds`
- 若不同 solver 的差距很接近，或 architecture ranking 不稳定，再补到 `3 training seeds`
- **ImageNet-64**：先跑 `1 training seed` 做现象确认；只有 CIFAR 主结论成立后，再补第 `2 training seed`

#### 为什么不能一开始每个模型只训练 1 个？

因为你要比较的是：

> **architecture-specific solver preference 是否稳定存在**

若只看单个 training seed，你有可能看到的是：
- 某次初始化特别幸运 / 特别差
- 某次训练恰好更平滑或更不平滑
- 某个 solver 对这个单次训练轨迹刚好更友好

因此：
- **探索阶段可以 1 seed**
- **主结论阶段不建议只靠 1 seed 定论**
- 最终表格最好报告 mean / std，或至少把各 seed 的点都画出来

---

### 2.4 diffusion 训练里还有很多超参数，怎么控制变量？

核心思想：

> **把“训练-to-competence”视作准备工作，只允许最小化的优化调参；把“architecture × sampler”作为主变量。**

#### 必须固定的训练变量
- 数据预处理与 normalization
- train/val split
- conditioning 方式
- forward diffusion family
- noise schedule
- model output parameterization（例如 epsilon-pred）
- loss weighting
- optimizer family
- EMA
- batch size（全局 batch）
- LR schedule 形式
- timestep sampling strategy
- augmentation（最好关闭）
- 如果是 latent：同一个 VAE

#### 可以允许的很小范围 tuning
仅限于：
- **base learning rate**
- **warmup 长度**
- **width/base_channels**（为匹配预算）

**不允许**在不同 architecture 上分别改：
- noise schedule
- prediction target
- loss weighting
- optimizer family
- augmentation
- CFG / guidance trick
- 额外 regularization trick

### 2.5 solver 应该复用开源代码，还是从头实现？

#### 推荐：**核心 solver 复用官方/主流实现，但必须套一层你自己的统一 wrapper。**

原因：
- 从头实现容易把论文里的细节写错，尤其是 parameterization 转换、time grid、multistep warm start
- 你要研究的是 solver × architecture，不是复现 solver 论文
- 但完全黑箱调用也不够，因为你需要插桩记录中间量与局部误差

#### 最合理的工程方案
- **官方实现 / 主流实现**：DPM-Solver++, UniPC, DPM-Solver-v3, DEIS
- **自己实现**：Euler / DDIM-style first-order / Heun（这几个简单、可验证、便于做 reference）
- **统一 wrapper API**：
  - 统一输入输出为 `model_fn(x_t, t) -> eps` 或统一转换成 ODE drift
  - 统一时间参数化
  - 统一 step 记录
  - 统一随机种子与 noise init
  - 统一 NFE 计数
  - 统一关闭 solver-specific trick（正文）

#### 在正式实验前必须做的 sanity checks
1. 在一个固定 checkpoint 上复现公开 solver 的大致 ranking
2. 验证你的 wrapper 不改变论文默认结果的数量级
3. 检查 NFE 计数是否正确
4. 检查 predictor-corrector 框架下是否有“隐性额外 denoiser 调用”

---

### 2.6 NFE 到底是什么？

**NFE = Number of Function Evaluations**，在 diffusion 采样语境下，通常指 **每生成一个样本时，对 denoiser / score network 的前向调用次数**。

#### 你在文中必须这样定义
- **NFE 统计的是 denoiser 网络前向次数，不是 macro-step 数。**
- 如果一个方法每个 macro-step 调用模型 1 次，则 `NFE = step count`
- 如果某个方法 1 个 macro-step 内调用 2 次模型，则 `NFE = 2 × step count`
- 如果某个 corrector 不增加模型调用，则不额外增加 NFE

#### 你的主实验里建议这样做
- **正文不用 CFG**，避免“一步两次前向”的口径混乱
- 采用 unconditional 或 class-conditional（直接 label embedding）模型
- 因此正文里大多数 solver 的 NFE 将基本等于 denoiser 调用次数

#### 如果后面你想扩展到 CFG
需要同时报告：
- raw denoiser calls
- nominal sampling steps
- effective conditional NFE

---

## 3. 统一训练协议（建议默认版）

> 下面给出一套默认训练协议。核心不是“这是最强 recipe”，而是“这是一套在三类 backbone 上都能统一执行、且利于 solver 研究的 clean recipe”。

### 3.1 主训练设定（默认）
- **diffusion family**: VP diffusion
- **training schedule**: cosine noise schedule
- **prediction target**: epsilon prediction
- **loss**: MSE on epsilon
- **timestep sampling**: training 时从连续时间区间均匀采样（或等价离散化采样，但三类模型必须一致）
- **optimizer**: AdamW
- **betas**: `(0.9, 0.999)`
- **weight decay**: `0.01`
- **gradient clipping**: `1.0`
- **EMA**: `0.9999`
- **LR schedule**: linear warmup + cosine decay
- **augmentation**: none
- **mixed precision**: bf16 或 fp16（全模型一致）

### 3.2 仅允许的 pilot tuning
在 10% 预算的小 pilot 上，仅允许下面的小网格：

- `lr ∈ {1e-4, 2e-4, 3e-4}`
- `warmup_steps ∈ {5k, 10k}`
- `base_channels` 或 `hidden_size` 用于匹配预算

**一旦 pilot 定下来，后续主实验冻结。**

### 3.3 训练预算建议
#### CIFAR-10
- global batch: `256`
- max budget: `200M images seen`
- checkpoint interval: `10M images`
- proxy FID eval: 每 `20M images`
- training seeds:
  - pilot：`1`
  - main：先 `2`
  - 若 solver ranking 接近或不稳定，再补到 `3`

#### ImageNet-64
- global batch: `256` 或 `512`（三类模型一致）
- max budget（v1）: `250M–300M images seen`
- checkpoint interval: `20M images`
- proxy FID eval: 每 `40M images`
- training seeds:
  - 现象确认：`1`
  - 最终确认：`2`

> 若 CIFAR 上连 Result A 都不明显，不建议直接把 ImageNet-64 扩到完整矩阵。

### 3.4 checkpoint 选择
- 使用 **EMA 权重**
- 从所有已保存 checkpoint 中，选择 **reference sampler 下 proxy/full FID 最优** 的 checkpoint
- 附录再提供：
  - same-step checkpoint 对比
  - same-budget 最后 checkpoint 对比

---

## 4. 数据集与实验层级

### 4.1 主数据集

#### Dataset A: CIFAR-10 32×32, unconditional, pixel-space
作用：
- 低成本诊断台
- 多 solver sweep
- 多 seed
- 适合做局部误差、频谱图、Jacobian 近似、reference trajectory

#### Dataset B: ImageNet-64, class-conditional, pixel-space
作用：
- 主语义 benchmark
- 验证 architecture-specific solver preference 是否迁移到更复杂语义数据
- 无需引入 VAE confound

### 4.1A Validation 划分策略

#### CIFAR-10
- CIFAR 官方没有单独 validation split，因此**必须从官方 training set 中固定切出 validation**
- 推荐：使用固定 `split seed` 做 **分层抽样**，切成：
  - `45k train`
  - `5k validation`
- 每个类别保留相同数量到 validation，避免类别分布偏移
- 所有 backbone、所有 training seeds、所有 solver 实验都共用这一个 split
- **官方 test set 不参与 checkpoint selection，只用于最终报告或附录 sanity check**

#### ImageNet-64
- 若使用标准 ImageNet 划分，建议固定一套 held-out validation protocol 用于 checkpoint selection
- 原则与 CIFAR 相同：**validation 只用于模型选择，不用于最终 claim**

### 4.2 外部验证（可选）

#### Dataset C: LSUN-Bedroom 256 或 FFHQ 256
作用：
- 验证结论不是 CIFAR/ImageNet 特有
- 检查不同空间频谱结构的数据是否改变 coupling pattern

#### Dataset D: ImageNet-256 latent（附录）
作用：
- 靠近 DiT / U-ViT 常见设置
- 验证正文规律在 latent setup 中仍成立

---

## 5. Sampler 选择与采样协议

### 5.1 正文 core solvers（5 个）

1. **Euler / DDIM-style first-order baseline**
2. **Heun (2nd-order baseline)**
3. **DEIS**
4. **DPM-Solver++**
5. **UniPC**

### 5.2 附录 / 扩展 solvers

6. **DPM-Solver-v3**
7. **AMED-Plugin**（如果你愿意做少量训练辅助）
8. **S4S**（作为 learned solver upper-bound，非 training-free）

### 5.3 正文统一采样口径

#### 统一 time grid
正文主比较中，所有 solver 先共用 **同一类 time grid**，不要一开始就给每个 solver 用作者最优化的 grid。

**推荐主 grid**：
- 在 `logSNR` 或 `lambda` 空间中均匀取点
- 所有 architecture 与 solver 共享这一 grid family

原因：
- 这样主效应更接近 `solver × architecture`
- 避免主结论被 `solver × timestep schedule` 混淆

#### 附录再做两类扩展
- **Solver-recommended grid**
- **Optimized timesteps**

### 5.4 NFE sweep
正文统一：

`NFE ∈ {4, 6, 8, 10, 15, 20, 35, 50}`

说明：
- `4–10`: few-step 关键区间
- `15–20`: practical 区间
- `35–50`: ranking 是否收敛

### 5.5 reference sampler

需要一个高精度数值 reference，而不是假设某个 solver 是“真值”。

#### 推荐做法
- **CIFAR-10**：Heun 1024-step（或 2048-step）
- **ImageNet-64**：Heun 256-step 或 512-step
- 在一个固定子集上把步数翻倍，检查：
  - sample-level L2 / LPIPS 变化是否很小
  - FID 是否几乎不变

若 doubling 后差别已可忽略，则把较低成本版本当作 numerical reference。

---

## 6. 主评测指标

### 6.1 质量指标

#### (1) Raw FID
保留，但不能作为唯一核心指标。

#### (2) Relative degradation（正文主指标）
定义：

`ΔFID(a, s, N) = FID(a, s, N) - FID(a, ref)`

其中：
- `a` = architecture
- `s` = solver
- `N` = NFE
- `ref` = 对应 architecture 的高精度 reference sampler

**这是正文最重要的质量指标。**

理由：
- 去掉不同 architecture 自身天花板不同的影响
- 更直接回答“某 architecture 对 solver 有多敏感”

#### (3) Precision / Recall
用于拆分 fidelity 与 coverage。

#### (4) Instability / Failure rate
记录：
- NaN / Inf
- 颜色爆炸
- 大面积饱和
- 明显崩坏样本比例

### 6.2 代价指标
- wall-clock per sample
- images/sec
- per-sample MACs / FLOPs
- peak memory（可选）

### 6.3 统计协议
- CIFAR：主结果先做 `2 training seeds`，必要时补到 `3`
- ImageNet64：先 `1 training seed` 做确认，最终关键结论补到 `2`
- FID 最终使用 50k samples
- 训练中 proxy FID 使用 10k samples
- paired noise bank：不同 solver 对比时使用同一组初始噪声
- sampling seed 与 training seed 分开记录，避免混淆

---

## 7. 从“工程 benchmark”走向“机制解释”的实验

### 7.1 机制实验总思路
你需要把 story 写成：

`architecture -> field signature -> solver local error placement -> overlap with architecture difficulty -> sample quality`

---

## 8. 机制量一：局部离散化误差（local truncation error）

### 8.1 定义
固定 architecture `a`、solver `s`、step size `h`、时间点 `t`。

从相同 `x_t` 出发：
- 用 solver 走一步得到 `x_{t-h}^{(a,s)}`
- 用 fine reference 走到相同终点得到 `x_{t-h}^{(a,ref)}`

定义：

`L_{a,s}(t; h) = E || x_{t-h}^{(a,s)} - x_{t-h}^{(a,ref)} ||_2^2`

### 8.2 采样 `x_t` 的两种方式

#### 方式 A（推荐主方式）
从真实 `x_0` 前向加噪得到 `x_t`
- 好处：更接近 denoising task difficulty
- 适合结合 architecture difficulty map

#### 方式 B（补充）
从 reference trajectory 中抽取 `x_t`
- 好处：更接近真实生成路径
- 适合检查 compounding effect

### 8.3 怎么做
- 选固定 held-out 图片子集（如 2k images）
- 选 `t` 的 16 个 bin（均匀按 logSNR 分桶）
- 每个 bin 内随机抽样若干 `x_t`
- 每个 solver 在固定 `h` 上做 one-step comparison

### 8.4 产出图
- `t vs local error` 曲线（不同 architecture × solver）
- `t × solver` heatmap（每个 architecture 一张）

---

## 9. 机制量二：经验阶数（empirical order）

### 9.1 定义
对固定 `a,s,t`，在多个 step size `h` 上测 local error 或 terminal error，拟合：

`log(error) = c + p_eff * log(h)`

其中 `p_eff` 为经验阶数。

### 9.2 目标
观察：
- 某 solver 的理论高阶是否在不同 architecture 上都能实现？
- 是否某 architecture 上高阶行为塌陷，变得更像低阶？

### 9.3 实操
- 选 `h ∈ {h0, h0/2, h0/4}`
- 每个 architecture、solver、时间区间都拟合 `p_eff`

### 9.4 产出图
- `log h` vs `log error` 图
- `p_eff` 的 bar plot / heatmap

---

## 10. 机制量三：field smoothness / stiffness proxy

### 10.1 时间平滑性
固定 `x_t`，测：

`S_time(a,t) = E || f_a(x_t, t+δ) - f_a(x_t, t) || / δ`

以及二阶 finite difference。

### 10.2 状态敏感性 / stiffness proxy
近似测 Jacobian 的谱范数或 top singular value：

`J_a(x_t, t) = ∂f_a / ∂x`

估计：
- power iteration + JVP/VJP
- 仅对少量样本和少量时间点做

### 10.3 目标
验证：
- smoother / less stiff 的 field 是否更适合 high-order / multistep solver
- 更 stiff 的 field 是否更依赖保守 solver 或更细步长

### 10.4 产出图
- `t vs temporal smoothness`
- `t vs Jacobian spectral norm`
- 与 `solver gain` 的散点相关图

---

## 11. 机制量四：temporal redundancy

### 11.1 输出级 redundancy
对相邻时间点的 denoiser 输出测 cosine similarity：

`R_out(a,t) = cos( eps_a(x_t,t), eps_a(x_t,t-δ) )`

### 11.2 表征级 redundancy（可选）
对选定 block / layer 的 hidden state：
- cosine similarity
- CKA / PWCCA（若你愿意）

### 11.3 目标
验证：
- 若某 architecture 的相邻 step 更冗余，则历史信息/插值信息更有用
- 这种冗余是否能预测 UniPC / multistep DPM-Solver++ 的收益

### 11.4 产出图
- `t vs redundancy`
- `redundancy vs solver gain` 散点图

---

## 12. 机制量五：architecture difficulty map（最关键）

### 12.1 定义
在 held-out 真实图像上取 `x_0`，前向加噪到 `x_t`，让模型输出统一转换为 `x0_hat`。

做 2D FFT，并按 radial frequency band 分桶。定义：

`D_a(t, r) = E || P_r( x0_hat^a(x_t,t) - x_0 ) ||_2^2`

其中 `P_r` 是第 `r` 个频带投影。

### 12.2 解释
这个图回答：

> **某 architecture 在什么时间段、什么频段最难。**

### 12.3 实操
- `t` 分 16 个 logSNR bins
- frequency bands 分 8 个 radial bins
- 对每个 architecture 画一张 `t × freq` heatmap

### 12.4 产出图
- U-Net / U-ViT / DiT 的 difficulty map 并排展示

---

## 13. 机制量六：solver error map（与 difficulty map 配对）

### 13.1 定义
对固定 architecture `a` 与 solver `s`，测 solver 一步误差的频域分布：

`E_{a,s}(t, r) = E || P_r( x_{t-h}^{(a,s)} - x_{t-h}^{(a,ref)} ) ||_2^2`

### 13.2 解释
这个图回答：

> **某 solver 在某 architecture 上的误差，主要掉在什么时间段、什么频段。**

### 13.3 产出图
- 固定 architecture，比不同 solver
- 或固定 solver，比不同 architecture

---

## 14. 核心机制量：overlap score

### 14.1 定义
先把 `D_a(t,r)` 与 `E_{a,s}(t,r)` 归一化，再定义：

`O(a,s) = Σ_{t,r} D_norm(a,t,r) * E_norm(a,s,t,r)`

### 14.2 核心假设

> **最终质量下降更取决于 overlap score，而不是 solver 总误差。**

### 14.3 必做对照
比较两个相关性：

1. `total solver error -> ΔFID`
2. `overlap score -> ΔFID`

如果后者显著更强，这是整篇 paper 最重要的机制证据之一。

### 14.4 产出图
- `total error vs ΔFID`
- `overlap score vs ΔFID`
- 同一张图中对比拟合优度

### 14.5 机制量优先级（按性价比分层）

#### P0：必须先做
- local truncation error
- difficulty map
- solver error map
- overlap score

理由：
- 这 4 个量最直接服务于 H3
- 它们能最快回答“误差落点是否比总误差更重要”

#### P1：建议补做
- temporal smoothness
- output-level redundancy

理由：
- 这两类量成本较低
- 对 H2 的解释较直观，适合做辅助证据

#### P2：有信号后再做
- empirical order
- stiffness proxy（Jacobian spectral norm）
- hidden-state redundancy / CKA / PWCCA

理由：
- 这几项要么计算更重，要么数值噪声更大，要么解释门槛更高
- 不建议在主结论尚未出现前优先投入

补充说明：
- “机制量列得多”本身不是问题
- 但这些量的**优先级必须不一样**
- 尤其 `local error + reference rollout` 与 `Jacobian` 并不是完全“免费”的分析
- 最稳妥的策略是：先用 P0 打主线，再决定是否扩到 P1 / P2

---

## 15. 因果干预实验（避免只停留在相关性）

### 15.1 干预 A：Hybrid solver by phase
目的：测试 architecture 是否具有不同的最佳 phasewise solver。

做法：
- 固定总 NFE
- early 使用 solver A，late 使用 solver B
- 扫描切换点
- 比较不同 architecture 的最优切换位置

优先级说明：
- 这是**最先做**的干预
- 它不需要重训 backbone，成本最低
- 若连这个都看不到 phase signal，就不应急着做更重的结构重训

### 15.2 干预 B：Skip-ablated U-ViT
目的：测试 long skip 是否影响 solver compatibility。

做法：
- 对 U-ViT 逐级减弱或移除 long skip
- 其他训练协议不变
- 观察 solver ranking 是否向 DiT 靠拢

优先级说明：
- 这是**第二个做**的结构干预
- 相比改 DiT attention，skip ablation 往往更容易实现与解释

### 15.3 干预 C：Local-window DiT
目的：测试 attention locality 是否影响 solver compatibility。

做法：
- 在 DiT 中注入 local attention window
- 其他训练协议不变
- 观察：
  - solver ranking 是否系统移动
  - smoothness / redundancy / difficulty map 是否改变

优先级说明：
- 这是**最后做**的结构干预
- 实现与训练风险通常都比 skip ablation 更高

---

## 16. 控制变量总表

| 类别 | 变量 | 正文主线 | 备注 |
|---|---|---:|---|
| Data | dataset / split | 固定 | 三类 backbone 完全一致 |
| Data | validation split seed | 固定 | CIFAR 从官方 train 中切 `45k/5k` |
| Data | preprocessing / normalization | 固定 | 不做花式增强 |
| Conditioning | unconditional or class-cond | 固定 | 正文不用 CFG |
| Training | diffusion family | 固定 | VP |
| Training | noise schedule | 固定 | cosine |
| Training | prediction target | 固定 | epsilon |
| Training | loss weighting | 固定 | 同一定义 |
| Training | optimizer family | 固定 | AdamW |
| Training | LR schedule form | 固定 | warmup + cosine |
| Training | EMA | 固定 | 同一 decay |
| Training | batch size | 固定 | global batch 相同 |
| Training | max budget | 固定 | 用 images seen 表达 |
| Training | width/base_channels | 可变 | 仅用于匹配预算 |
| Training | base LR | 小范围 pilot tuning | 仅在小网格里选 |
| Training | training seed | 重复 | CIFAR 先 2、必要时 3；ImageNet 先 1、最终 2 |
| Architecture | macro-structure | 主变量 | U-Net / U-ViT / DiT |
| Inference | time grid family | 固定（正文） | 同一 shared grid |
| Inference | solver-specific trick | 关闭（正文） | 阈值化、动态裁剪等先关掉 |
| Inference | sampling seed / noise bank | 固定 | 做 paired 对比 |
| Metrics | FID stats / sample count | 固定 | 同 clean-fid protocol |
| Latent | VAE | 固定（若用 latent） | 正文尽量先不用 latent |

---

## 17. 完整实验矩阵

### Phase 0: 预算匹配 pilot
目标：确定三类 backbone 的 matched setting。

#### 输入
- CIFAR-10
- 10% 训练预算
- width/base_channels 网格

#### 输出
- 每个 family 选出 1 个 main config
- 参数量与 per-step compute 落入同一预算窗口
- 训练稳定性正常

#### 保存内容
- 参数量
- 单步 MACs / FLOPs
- 训练 loss 曲线
- proxy FID 曲线

---

### Phase 1: CIFAR 主模型训练（先轻后重）

#### CIFAR-10
- U-Net × 1 seed（pilot）
- U-ViT × 1 seed（pilot）
- DiT × 1 seed（pilot）
- 选定 main config 后，各 backbone 先补到 `2 training seeds`
- 若 architecture-specific ranking 接近或不稳定，再补第 `3 seed`

#### 保存内容
- train / val loss
- proxy FID
- checkpoint
- EMA checkpoint

---

### Phase 2: CIFAR 核心 solver benchmark

#### 矩阵
- architectures: 3
- solvers: 5
- datasets: 1（先只做 CIFAR）
- NFE: 8 个点
- seeds: CIFAR 先 2，必要时 3

#### 输出
- raw FID
- ΔFID
- precision / recall
- runtime
- instability rate

---

### Phase 3: CIFAR 数值机制分析（先 P0）

#### 要测的量
- local truncation error
- difficulty map
- solver error map
- overlap score

#### 若主线出现明显信号，再补
- temporal smoothness
- output-level redundancy
- empirical order
- stiffness proxy

#### 采样子集建议
- held-out images: 2k
- noise seeds for trajectories: 512
- selected timesteps: 16 bins

---

### Phase 4: 低成本干预与结构干预（仍在 CIFAR）
- Hybrid solver switching
- Skip-ablated U-ViT
- Local-window DiT

推进条件：
- 只有在 CIFAR 上已经至少看到 Result A，并且 overlap / mechanism analysis 有初步解释力时，才进入结构重训干预

---

### Phase 5: ImageNet-64 确认实验
- 先 1 seed 跑通主矩阵中的最关键子集
- 若 CIFAR 上的现象在 ImageNet-64 上复现，再补第 2 seed
- 若 CIFAR 上结果不稳，不建议直接扩到 ImageNet-64 完整矩阵

---

### Phase 6: 外部验证
- LSUN-Bedroom / FFHQ
- DPM-Solver-v3
- solver-recommended grids
- optimized timesteps

---

## 18. 图表清单（建议正文图）

### Figure 1. Quality–NFE curves
- 每个 architecture 一个 panel
- x: NFE
- y: ΔFID（主）/ FID（辅）
- 线：不同 solver

### Figure 2. Best solver heatmap
- 行：architecture
- 列：NFE
- 颜色：最优 solver family

### Figure 3. Pareto frontier
- x: wall-clock / sample 或 per-sample FLOPs
- y: ΔFID
- 点：solver@NFE

### Figure 4. Empirical order plot
- x: step size
- y: local error（log-log）
- 比较不同 architecture 上的 `p_eff`

### Figure 5. Architecture difficulty maps
- `t × frequency` heatmap
- U-Net / U-ViT / DiT 并排

### Figure 6. Solver error maps
- 固定 architecture 比 solver
- 或固定 solver 比 architecture

### Figure 7. Overlap explains quality
- 左：total error vs ΔFID
- 右：overlap score vs ΔFID
- 比较相关性与拟合优度

### Figure 8. Hybrid solver switching
- x: switch point
- y: ΔFID
- 不同 architecture 的最优切换点对比

### Figure 9. Intervention moves solver preference
- Local-window DiT / Skip-ablated U-ViT 的质量曲线或 solver ranking 变化图

---

## 19. 结果判定标准（你要追的关键结果）

### 必须打中的 Result A
**不同 architecture 的 best solver family 在 4–20 NFE 内不同。**

### 必须打中的 Result B
**overlap score 对 ΔFID 的解释力明显强于 total solver error。**

### 最强加分项 Result C
**结构干预（locality / skip）会系统地移动 solver preference。**

只要 A + B 成立，这个题就已经成立；若再有 C，paper 的说服力会强很多。

---

## 20. 训练/采样代码组织建议

### 20.1 不要把 solver 逻辑散落在各项目录中
建议统一接口：

- `train.py`
- `sample.py`
- `solvers/`
  - `euler.py`
  - `heun.py`
  - `deis_wrapper.py`
  - `dpmpp_wrapper.py`
  - `unipc_wrapper.py`
  - `dpmv3_wrapper.py`
- `analysis/`
  - `eval_fid.py`
  - `eval_local_error.py`
  - `eval_empirical_order.py`
  - `eval_redundancy.py`
  - `build_difficulty_map.py`
  - `build_error_map.py`
  - `compute_overlap.py`
  - `plot_all.py`

### 20.2 每次采样都要记录
- architecture 名称
- checkpoint id
- solver 名称
- solver 版本 / commit hash
- NFE
- time grid
- random seed / noise bank id
- whether thresholding/clipping is on
- runtime

### 20.3 机制实验需要额外保存
- selected trajectories
- selected hidden states（若做表征冗余）
- per-step `eps_hat` / `x0_hat`
- FFT-band error

---

## 21. 常见坑与规避方式

### 坑 1：把“训练没训好”误判为 architecture 差异
规避：
- 同一最大训练 budget
- 同一 checkpoint 选择规则
- 补 same-step ablation

### 坑 2：把“time grid 优化”误判为 solver 优势
规避：
- 正文用 shared grid
- optimized timesteps 放附录

### 坑 3：把“CFG doubling denoiser calls”混进 NFE
规避：
- 正文不用 CFG

### 坑 4：只报 raw FID，忽略 architecture 天花板不同
规避：
- 主指标使用 ΔFID

### 坑 5：只做大表，没有机制解释
规避：
- 一开始就规划 difficulty map、error map、overlap score

### 坑 6：把 solver 实现细节差异当成科学结论
规避：
- 统一 wrapper
- version lock
- sanity checks

### 坑 7：拿 test set 当 validation 用
规避：
- CIFAR 固定从官方 train 切出 `45k/5k`
- test 不参与 checkpoint 选择
- 所有 backbone 与 training seeds 共用同一 split

---

## 22. 推荐执行顺序（最实用版）

### 第 1 周：搭骨架
- 统一训练脚手架
- 统一 solver API
- 实现 Euler / Heun
- 接入官方 DPM-Solver++ / UniPC / DEIS

### 第 2 周：预算匹配 pilot
- CIFAR 上跑小预算
- 选定 3 个 main backbone config

### 第 3–4 周：CIFAR 主 benchmark
- 先跑 2 seeds
- 完成 core 5 solvers × NFE sweep
- 先确认 solver ranking 是否随 architecture 变化

### 第 5 周：CIFAR 机制分析
- local error
- difficulty map / error map / overlap
- 若有空间，再补 smoothness / redundancy

### 第 6 周：低成本干预
- hybrid solver switching

### 第 7 周：结构干预
- skip-ablated U-ViT
- 若信号足够强，再做 local-window DiT

### 第 8–9 周：ImageNet64 确认
- 先 1 seed 跑关键子集
- 现象成立后补第 2 seed

### 第 10 周：外部验证与补图
- DPM-Solver-v3
- optimized timesteps
- 额外数据集（可选）

---

## 23. 最终的写作主线（论文 narrative）

你可以按下面这个逻辑写论文：

1. **现象**：solver ranking changes across architectures
2. **数值解释**：different architectures induce different field smoothness / stiffness / redundancy / phase structure
3. **机制解释**：solver errors are not equally harmful; what matters is where those errors land in `t × frequency`
4. **因果支持**：changing locality / skip changes solver preference
5. **实用结论**：given a backbone family, one can choose better solver families and phase allocations

---

## 24. 文献对齐（用于写作，不是必须全部正文展开）

你的实验方案与下列工作天然对齐：

- **Karras et al., 2022, Elucidating the Design Space of Diffusion-Based Generative Models**
  - 强调把训练、采样、preconditioning 等设计因素拆开分析
- **Bao et al., 2022/2023, U-ViT**
  - 提出 long skip 对 diffusion 很关键，适合做桥梁架构
- **Peebles & Xie, 2022/2023, DiT**
  - 强调 DiT 的 scaling 更适合用 GFLOPs 观察
- **Lu et al., 2022, DPM-Solver / DPM-Solver++**
  - 代表 diffusion-specific ODE solver family
- **Zhao et al., 2023, UniPC**
  - 代表 predictor-corrector / arbitrary-order family
- **Zheng et al., 2023, DPM-Solver-v3**
  - 把 solver 与 pretrained model statistics 联系起来
- **Xue et al., 2024, Optimized Time Steps**
  - 表明小 NFE 下 timestep grid 本身就是关键变量
- **An et al., 2024, Inductive Biases of DiT**
  - 表明 DiT 的 generalization 与 attention locality 强相关
- **Unraveling the Temporal Dynamics of the U-Net in Diffusion Models, 2023**
  - 表明不同 timestep / skip components 的作用不同，支持 phasewise 分析

---

## 25. 最后的简化版决策清单

如果你现在马上开工，先按下面执行：

### 主线最小可发表版本
- **Backbones**: U-Net / U-ViT / DiT
- **Datasets**: CIFAR-10 + ImageNet-64
- **Solvers**: Euler, Heun, DEIS, DPM-Solver++, UniPC
- **Main metric**: ΔFID
- **Mechanism metrics**: local error, empirical order, redundancy, difficulty map, error map, overlap score
- **Interventions**: local-window DiT, skip-ablated U-ViT

### 先不做的
- 不先碰 text-to-image
- 不先碰 CFG
- 不先混入 VAE（正文）
- 不先做 solver-specific tuned timesteps
- 不先把 architecture 搜到极致

这会让你的主问题最干净，也最容易形成有说服力的 mechanism story。
