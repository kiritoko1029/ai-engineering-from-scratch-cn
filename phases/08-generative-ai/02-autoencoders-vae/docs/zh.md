# 自编码器与变分自编码器（VAE）

> 普通的自动编码器仅负责压缩数据后再进行重构，它只会记忆数据而无法生成新内容。若加入一个技巧——强制让代码呈现高斯分布的形式——就能得到一个采样器。正是这个“将 `z = μ + σ·ε` 重新参数化”的技巧，使得2026年所有使用的潜在扩散模型与流匹配图像模型都在输入端采用了VAE结构。

**类型：** 构建
**语言：** Python
**先修课程：** 第3阶段 · 02课（反向传播）、第3阶段 · 07课（卷积神经网络）、第8阶段 · 01课（分类学）
**耗时：** 约75分钟

## 问题所在

将784像素的MNIST数字压缩为16位编码，然后再进行重构。普通的自编码器在重构时的均方误差表现优异，但其编码空间结构十分粗糙。如果在该编码空间中随机选取一个点并解码，得到的结果将是噪声，因为它缺乏有效的采样机制，本质上只是披着压缩模型外衣的结构。

实际上我们需要的条件是：(a) 编码空间是一个干净、平滑且可采样的分布——例如各向同性的高斯分布 `N(0, I)`；(b) 对任意采样结果解码后都能得到合理的数字图像；(c) 编码器与解码器依然具备良好的压缩能力。三个目标，一种架构，一个损失函数。

Kingma在2013年提出的VAE通过训练编码器使其输出分布 `q(z|x) = N(μ(x), σ(x)²)`，并利用KL惩罚项将该分布向先验分布 `N(0, I)` 拉近，之后再从 `q(z|x)` 中采样得到 `z` 进行解码，从而解决了这一问题。在推理阶段则可直接忽略编码器，直接从 `N(0, I)` 中采样并解码。正是KL惩罚项迫使编码空间具备结构化特征。

到了2026年，VAE已很少以独立形式出现——由于扩散模型在原始图像质量方面表现更优，VAE已被其取代——但它们依然是所有潜在空间扩散模型（如SD 1/2/XL/3、Flux、AudioCraft）所采用的编码器。掌握VAE就相当于掌握了你所使用的各类图像处理流程中那个不可见的底层结构。

## 概念概述

![自编码器与VAE：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`, `x̂ = decoder(z)`, 损失函数 = `||x - x̂||²`。代码空间为非结构化状态。

**VAE编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。二者共同定义了概率分布 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 直接从 `q(z|x)` 中采样是不可微分的。可将采样表达式改写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。如此一来，`z` 就成为 `(μ, σ)` 的确定性函数加上非参数噪声——梯度可以通过 `μ` 和 `σ` 传递。

**损失函数。** 证据下界（ELBO），包含两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重构过程将 `x̂` 向 `x` 推近，而 KL 散度则将 `q(z|x)` 向先验分布推动。二者之间存在权衡关系：当 β 值较小时（<1），样本更清晰，但代码空间偏离高斯分布的程度更大；当 β 值较大时（>1），代码空间更为规整，但样本的清晰度会降低。β-VAE（Higgins 2017）使这一参数变得广为人知，并推动了解耦技术的研究。

**采样过程。** 在推理阶段：首先从 `z ~ N(0, I)` 中抽取随机值，然后通过解码器进行前向传播。仅需一次前向传递，无需像扩散模型那样的迭代采样步骤。

```figure
vae-latent-grid
```

## 构建它

`code/main.py` 实现了一个无需使用 numpy 或 torch 的小型 VAE。输入数据为从 8 维空间中的双组分高斯混合模型中生成的 8 维合成数据。编码器与解码器均为单隐藏层的 MLP 结构。该实现包含了 tanh 激活函数、前向传播过程、损失计算方式，以及手动编写的反向传播逻辑。此代码并非用于生产环境，仅用于教学演示。

### 步骤 1：编码器前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用 `log σ²` 而非 `σ`，以此使网络输出不受约束（对 σ 进行 softplus 处理会带来问题——当 σ 接近 0 时梯度会消失）。

### 步骤 2：重新参数化并解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 步骤 3：ELBO 指标

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

由于两个分布均为高斯分布，因此可以求出精确的封闭形式KL散度。无需进行数值积分。即便到了2026年，仍有人会使用蒙特卡洛方法来估算KL散度——这种做法毫无必要地使代码运行速度降低3倍。

### 步骤 4：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。共五行。

## 常见陷阱

- **后验坍缩问题**。KL散度项会极度强烈地推动 `q(z|x) → N(0, I)`，导致 `z` 完全无法携带关于 `x` 的任何信息。解决方法：采用 β-退火法（初始值设为 β=0，逐步提升至 1）、使用自由比特，或对非活跃维度跳过 KL 计算。
- **样本模糊问题**。高斯解码器所对应的似然函数意味着采用 MSE 重建方式，而这种方式对于 L2 范数（即均值）而言是贝叶斯最优的——一组合理数字的均值会呈现为模糊的数字。解决方法：使用离散解码器（如 VQ-VAE、NVAE），或仅将 VAE 用作编码器，并在潜在表示上叠加扩散模型（这正是 Stable Diffusion 的实现方式）。
- **β 值过大且提升过早**。参见后验坍缩问题。建议初始 β 值设为约 0.01，然后逐步提升。
- **潜在维度过小**。MNIST 数据集适用 16 维，ImageNet 256² 数据集适用 256 维，ImageNet 1024² 数据集适用 2048 维。Stable Diffusion 的 VAE 能将 512×512×3 分量的数据压缩为 64×64×4 分量（空间维度缩小了 32 倍，通道数也减少了 32 倍）。

## 使用它

2026年版的VAE技术栈：

| 场景 | 推荐方案 |
|-----------|----------|
| 用于扩散模型的图像潜在编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或Flux VAE |
| 音频潜在编码器 | Encodec（Meta开发）、SoundStream或DAC（Descript提供） |
| 视频潜在表示 | Sora的时空块结构、Latte VAE、WAN VAE |
| 解耦式表示学习 | β-VAE、FactorVAE、TCVAE |
| 用于Transformer建模的离散潜在表示 | VQ-VAE、RVQ（ResidualVQ） |
| 用于生成的连续潜在表示 | 基础VAE，随后在该潜在空间中对流模型/扩散模型进行条件控制 |

潜在扩散模型是一种在编码器与解码器之间嵌入了扩散模型的VAE。VAE负责粗略压缩数据，而扩散模型则承担核心处理任务。视频领域采用相同架构（VAE + 视频扩散DiT），音频领域则采用（Encodec + MusicGen Transformer）的组合方式。

## 发布它

将文件保存为 `outputs/skill-vae-trainer.md`。

该模型输入包括数据集概况、目标潜在维度以及下游应用场景（重建、采样或作为潜在扩散模型的输入），输出内容包括架构选择（普通型/β型/VQ型/RVQ型）、β值调度策略、潜在维度大小、解码器似然类型（高斯分布或分类分布），以及评估指标方案（重建均方误差、各维度的KL散度，以及 `q(z|x)` 与 `N(0, I)` 之间的弗雷歇距离）。

## 练习题

1. **简单。** 将 `code/main.py` 中的 `β` 值改为 `0.01`、`0.1`、`1.0` 或 `5.0`，并记录最终的重建 MSE 值与 KL 散度值。对于您生成的合成数据而言，哪个 β 值属于帕累托最优？
2. **中等难度。** 用伯努利似然函数（交叉熵损失）替换高斯解码器似然函数，并在相同合成数据的二值化版本上对比样本质量。
3. **困难。** 将 `code/main.py` 扩展为微型 VQ-VAE 模型：用大小为 K=32 的代码本中的最近邻查找来替代连续的 `z` 值。比较重建 MSE 值，并说明实际使用了多少个代码本条目（以此验证是否存在代码本坍缩现象）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 自编码器 | 编码-解码网络 | `x → z → x̂`，通过最小均方误差进行学习。不具备生成能力。 |
| VAE | 带采样器的自编码器 | 编码器输出一个概率分布，KL惩罚项用于塑造代码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时该值达到最小。 |
| 重参数化 | `z = μ + σ·ε` | 将随机节点表示为确定性部分与纯噪声之和，从而支持通过采样进行反向传播。 |
| 先验分布 | `p(z)` | 潜在变量的目标概率分布，通常为 `N(0, I)`。 |
| 后验坍缩 | “KL项占上风” | 编码器忽略输入 `x`，直接输出先验分布；解码器则必须生成虚假内容。 |
| β-VAE | 可调的KL权重 | `loss = recon + β·KL`。β值越大，特征分离度越高但图像越模糊。 |
| VQ-VAE | 离散潜在变量 | 用最近的代码本向量替代连续的 `z`，从而支持Transformer模型结构。 |

## 生产注意事项：在扩散模型服务器中，VAE 是计算负载最高的路径。

在 Stable Diffusion / Flux / SD3 的处理流程中，每个请求都会调用 VAE 两次——一次用于编码（在执行 img2img 或修复操作时），另一次用于解码。当分辨率达到 1024² 时，解码步骤往往会是整个流程中激活值与内存占用最高的环节，因为它需要将 `128×128×16` 大小的潜在特征上采样为 `1024×1024×3` 大小。这会带来两个实际问题：

- **对解码过程进行切片或平铺处理。** `diffusers` 库提供了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()` 方法。平铺方式虽然会产生轻微的接缝伪影，但能将内存占用从 `O(H·W)` 降低到 `O(tile²)`，对于分辨率在 1024² 及以上的场景在消费级 GPU 上尤为关键。
- **使用 bf16 格式进行解码，并在最终尺寸调整时采用 fp32 数值格式。** SD 1.x 版本的 VAE 是以 fp32 格式发布的，在分辨率达到 1024² 以上并转换为 fp16 格式时会无声地产生 NaN 值。SDXL 则提供了 `madebyollin/sdxl-vae-fp16-fix` 修复版本——建议始终使用该 fp16 修复版，或直接采用 bf16 格式。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 相关论文。  
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解耦型 β-VAE。  
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。  
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 当前最先进的图像 VAE。  
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器。  
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 的行业标准。
