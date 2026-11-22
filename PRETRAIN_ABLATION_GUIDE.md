# GSFix3D 预训练创新点与消融开关

对比基线：`GSFix3D-main`。创新实现：`GSFix3D-mainori`。

## 可消融的算法差异

| 编号 | 差异 | 创新版行为 | 关闭后的行为 | `config/train_gsfixer.yaml` 开关 |
|---|---|---|---|---|
| A1（代码修正） | 双输入实际启用 | mesh、GS 和带噪目标 latent 拼成 12 通道 | 仅 GS 和带噪目标 latent，8 通道 | `input_conditioning.dual_input_enabled` |
| A2 | 随机复合退化 | 随机模糊、噪声、JPEG、降采样、椒盐噪声和颜色扰动 | 原版固定 13x13 模糊和标准差 0.2 的高斯噪声 | `corruption_aug.enabled`；每个子退化也有独立 `enabled` |
| A3 | GS 黑洞增强 | 在 GS/mesh 输入独立生成大尺度、暗色、羽化的不规则孔洞 | 仅保留原版 PIPE 语义掩码增强 | `hole_aug.enabled` |
| A4 | 像素空间辅助监督 | `latent MSE + 0.05 * decoded-x0 L1` | 原版纯 latent MSE | `pixel_loss.enabled` |
| A5（合并原 A5/A6） | 扩展层 LoRA 预训练 | 冻结 UNet 主体，在 attention、projection、FFN、ResNet conv 和 time projection 上训练 LoRA；重建的 `conv_in` 仍训练 | 原版全量 UNet 训练 | `lora.enabled` |
| A7 | AdamW | AdamW，weight decay 0.01 | 原版 Adam，无 weight decay | `optimizer.adamw_enabled` |
| A8 | 梯度裁剪 | global norm 上限 1.0 | 不裁剪 | `gradient_clipping.enabled` |

核心方法创新消融为 A2、A3、A4、A5。A7、A8 属于训练策略消融。A1 是双输入配置读取修正，双输入本身来自原 GSFix3D，不作为新方法创新点。

## 不应包装成算法创新的差异

- `speedup.bf16`、`speedup.torch_compile`、SDPA/xFormers 选择属于速度、显存和硬件兼容优化。
- checkpoint 的 LoRA/`conv_in` 保存恢复、验证时切换 `eval()`、NaN/Inf 检查属于正确性或工程改进。
- 保存/验证周期和 `num_workers` 只影响运行效率。
- `speedup.timestep_max: 1000` 与当前 DDPM 的 1000 个时间步相同，当前设置不改变采样分布。
- `model_gsfixer.yaml` 的 checkpoint 名称变化只改变加载路径，不是训练算法。

## 必须控制的混杂变量

创新版还把 effective batch size 从 32 改为 64、学习率从 `3e-5` 改为 `1e-4`，并修改了 warmup、最终学习率比例和训练调度长度。做单因素消融时应固定这些数值，不要随着某个开关一起恢复，否则无法判断收益来自算法还是训练预算。

原版对子数据集配置了 `dual_input: true`，但训练入口从混合数据集的错误层级读取该字段，运行时实际传入 `false`。因此 A1 是重要实验因素和代码修正，但“mesh+GS 双输入”本身已存在于原 GSFix3D 设计中，不宜单独声称为新的方法贡献。

## 建议实验矩阵

以全部开关开启为 Full。核心方法每次只关闭一项：`-A2`、`-A3`、`-A4`、`-A5`；训练策略另做 `-A7`、`-A8`。A1 可作为实现核验或输入模态对照，但不要计入你的新增创新点。`-A5` 会从扩展层 LoRA 变为全量 UNet，参数量、显存和适宜学习率差异很大，建议同时报告训练参数量、显存、训练时长和质量指标，而不是只报告 PSNR/SSIM/LPIPS。

随机增强的消融至少使用相同随机种子；条件允许时使用 3 个种子报告均值和标准差。所有实验应从同一 Stable Diffusion v2 权重重新开始，不能跨开关恢复 checkpoint。
