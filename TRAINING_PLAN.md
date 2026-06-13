# PRNet 改进训练计划

> Baseline: PRNet (SMT-t backbone + CAP + GRD)
> 当前复现指标 (COD10K CAM subset): MAE 0.0218, maxFm 0.8845, Sm 0.9189
> 对比 SOTA: ZoomNeXt MAE 0.030, maxFm 0.871, Sm — | FSEL MAE 0.028, maxFm 0.879, Sm —

---

## Phase 0: 环境与数据准备

- [x] COD_datasets 软链接已删除（无效）
- [ ] 确认训练数据 `TrainDataset/Imgs/` + `TrainDataset/GT/` 路径
- [ ] 确认 TestDataset (CAMO, COD10K, NC4K) 可用
- [x] PRNet conda 环境可用

---

## Phase 1: 损失函数改进（零架构改动）

> 目标: 不修改模型结构，仅通过更好的损失函数提升指标
> 预期: MAE ↓ 0.5-1%，边界更清晰

| 组件 | 来源 | 文件 |
|------|------|------|
| Edge-weighted BCE | CGCOD `loss_f.py::seg_loss` | `COD/CGCOD/loss_f.py` |
| SSIM Loss | CGCOD `loss_f.py::SSIM` | `COD/CGCOD/loss_f.py` |

**修改:**
- `PRNet_train_cod.py`: 替换 `CE + iou_loss` 为 `seg_loss`（edge-weighted BCE + IoU）+ `l1_ssim_loss`
- ✅ 代码已实现 (2026-06-13)
- ✅ 数据已准备 (TrainDataset: 7000 imgs)
- ✅ NumPy 兼容性问题已修复
- ⏳ 待训练验证

---

## Phase 2: 边界分支（DGNet TextureEncoder + GIT）

> 目标: 解决伪装检测的边界模糊核心问题
> 预期: Sm ↑ 1-2%, boundary F-measure ↑

| 组件 | 来源 | 文件 |
|------|------|------|
| Texture Encoder | DGNet `DGNet.py::TextureEncoder` | `COD/DGNet/lib_pytorch/lib/DGNet.py` |
| Gradient-Induced Transition | DGNet `DGNet.py::GradientInducedTransition` | `COD/DGNet/lib_pytorch/lib/DGNet.py` |
| Neighbor Connection Decoder (可选) | DGNet `DGNet.py::NeighborConnectionDecoder` | `COD/DGNet/lib_pytorch/lib/DGNet.py` |

**修改:**
- `models/PRNet.py`: 
  1. 添加 TextureEncoder 分支（从 RGB 输入提取梯度特征）
  2. 在 CAP/GRD 之前插入 GIT 融合梯度信息
  3. 可选: 将 GRD 的部分相加替换为 NCD 的相乘连接

---

## Phase 3: 频率域注意力（FSEL Attention_F）

> 目标: 利用频域信息分离前景/背景纹理
> 预期: maxFm ↑ 1-2%, 在纹理复杂场景提升明显

| 组件 | 来源 | 文件 |
|------|------|------|
| Frequency Attention | FSEL `FSEL_modules.py::Attention_F` | `COD/FSEL/FSEL_ECCV_2024/lib/FSEL_modules.py` |
| Multi-Scale Spatial Attention | FSEL `FSEL_modules.py::Attention_S` | `COD/FSEL/FSEL_ECCV_2024/lib/FSEL_modules.py` |
| Frequency-gated FFN | FSEL `FSEL_modules.py::FeedForward` | `COD/FSEL/FSEL_ECCV_2024/lib/FSEL_modules.py` |

**修改:**
- `models/PRNet.py`: 将 `CoordAtt` 替换为 `Attention_F` + `Attention_S` 组合
- 在 MLPBlock 中替换 FFN 为 frequency-gated 版本

---

## Phase 4: 特征融合改进（ZoomNeXt MHSIU + RGPU）

> 目标: 更精细的多尺度特征融合
> 预期: 多尺度目标检测能力提升

| 组件 | 来源 | 文件 |
|------|------|------|
| MHSIU | ZoomNeXt `layers.py::MHSIU` | `COD/ZoomNeXt/methods/zoomnext/layers.py` |
| RGPU | ZoomNeXt `layers.py::RGPU` | `COD/ZoomNeXt/methods/zoomnext/layers.py` |

**修改:**
- `models/PRNet.py`: GRD 中的逐元素相加替换为 MHSIU 自适应加权融合
- 替换 MLPBlock 的上采样为 RGPU

---

## Phase 5: 全量组合

> 整合所有改进模块，训练完整 300 epoch

---

## 评估协议

| 数据集 | 指标 | 备注 |
|--------|------|------|
| CAMO (250张) | MAE, maxFm, wFm, Sm, Em | 全量评估 |
| COD10K CAM (2026张) | MAE, maxFm, wFm, Sm, Em | 跳过 NC |
| NC4K (619张) | MAE, maxFm, wFm, Sm, Em | 全量评估 |

---

## 训练配置

| 参数 | 值 |
|------|-----|
| backbone | SMT-t |
| 输入尺寸 | 384x384 |
| batch size | 8 |
| 学习率 | 5e-5 |
| epoch | 300 (Phase 1-4 可先跑 50) |
| optimizer | Adam |
| GPU | 单卡 |

---

## 实验记录

### Phase 1: 损失函数改进

| 实验 | 配置 | CAMO MAE | COD10K MAE | NC4K MAE | 备注 |
|------|------|---------|-----------|---------|------|
| Baseline | BCE + IoU | 0.0496 | 0.0218 | 0.0357 | 当前 |
| P1-v1 | + edge-weighted BCE | - | - | - | 待跑 |
| P1-v2 | + SSIM loss | - | - | - | 待跑 |
| P1-v3 | 全量组合 | - | - | - | 待跑 |

### Phase 2: 边界分支

| 实验 | 配置 | CAMO MAE | COD10K MAE | NC4K MAE | 备注 |
|------|------|---------|-----------|---------|------|
| P2-v1 | + DGNet TextureEncoder + GIT | - | - | - | 待跑 |

### Phase 3: 频率注意力

| 实验 | 配置 | CAMO MAE | COD10K MAE | NC4K MAE | 备注 |
|------|------|---------|-----------|---------|------|
| P3-v1 | + FSEL Attention_F | - | - | - | 待跑 |

### Phase 5: 全量组合

| 实验 | 配置 | CAMO MAE | COD10K MAE | NC4K MAE | 备注 |
|------|------|---------|-----------|---------|------|
| Full | 全部改进 | - | - | - | 待跑 |

---

## 参考代码位置

```bash
# PRNet baseline
COD/PRNet/models/PRNet.py
COD/PRNet/PRNet_train_cod.py

# 可移植模块
COD/DGNet/lib_pytorch/lib/DGNet.py          # TextureEncoder, GIT, NCD
COD/FSEL/FSEL_ECCV_2024/lib/FSEL_modules.py  # Attention_F, Attention_S, FFN
COD/CGCOD/loss_f.py                           # edge-weighted BCE, SSIM loss
COD/ZoomNeXt/methods/zoomnext/layers.py       # MHSIU, RGPU
COD/GLCONet/GLCONet_24_TNNLS/lib/GLCONet_module.py  # Multi-scale attention, GL_FI
```
