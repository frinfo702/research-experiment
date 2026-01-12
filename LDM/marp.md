---
marp: true
theme: academic
paginate: true
math: mathjax
---

# Latent Diffusion Model (LDM) 実験報告

## DDIM サンプラーによる画像生成実験

---

# 目次

1. 実験概要
2. 実験環境・設定
3. 実験結果
   - CelebA-HQ (本実験)
   - CIFAR-10 (比較実験)
4. 生成サンプル
5. 考察と知見
6. 今後の課題

---

# 実験概要

## 目的
- Latent Diffusion Model (LDM) の実験再現
- DDIM sampler による画像生成
- 異なる計算環境・データセットでの学習挙動の比較

## データセット
- **CelebA-HQ 256x256**: 高解像度顔画像 (本実験)
- **CIFAR-10 32x32**: 低解像度一般画像 (比較実験)

---

# 実験環境

| 環境 | スペック | データセット |
|------|----------|--------------|
| M4 Mac | Apple M4, 24GB RAM | CelebA-HQ 256x256 |
| Google Colab | T4 GPU | CelebA-HQ 128x128 |
| M4 Mac | Apple M4, 24GB RAM | CIFAR-10 64x64 |

## モデル構成
- **アーキテクチャ**: U-Net + Attention
- **タイムステップ**: T = 1000
- **ノイズスケジュール**: Cosine schedule
- **サンプラー**: DDIM
- **VAE**: stabilityai/sd-vae-ft-mse

---

# 実験設定: CelebA-HQ

## ハイパーパラメータ

| パラメータ | M4 Mac | Colab |
|-----------|--------|-------|
| Image Size | 256x256 | 128x128 |
| Downsample Factor | f=8 | f=4 |
| Latent Size | 32x32 | 32x32 |
| Batch Size | 64 | 64 |
| Learning Rate | 2e-4 | 2e-4 |

## 学習率スケジュール
- Linear warmup (5000 steps) → 固定 (lr = 2e-4)

---

# 実験結果: CelebA-HQ (M4 Mac)

## Loss 推移

| Step | Loss | Learning Rate |
|------|------|---------------|
| 1,000 | 0.577 | 4.0e-5 |
| 5,000 | 0.304 | 2.0e-4 |
| 10,000 | 0.280 | 2.0e-4 |
| 15,000 | 0.276 | 2.0e-4 |

## 結果
- Loss は **約 0.27-0.30 付近で収束**
- 人の顔が認識できる状態には至らなかった
- M4 Mac での学習速度: 約 2 it/s

---

# 実験結果: CelebA-HQ (Colab)

## 状況
- Google Colab (T4 GPU) で学習開始
- Loss は順調に低下し、**顔の輪郭が若干見え始めた**
- **セッション停止により中断**

## 課題
- Colab の実行時間制限
- チェックポイントからの再開は可能だが継続性に難あり
- 十分な学習ステップ数に到達できず

---

# 生成サンプル: CelebA-HQ (M4 Mac)

## @ 15,000 steps (Loss: 0.276)

![w:550](outputs/ldm_celeba_hq_256_T1000_cosine/samples/samples_00015000.png)

**評価**: ノイズが多く、顔の輪郭がぼんやりと見える程度

---

# 実験結果: CIFAR-10 (比較実験)

## Loss 推移

| Step | Loss |
|------|------|
| 10,000 | 0.353 |
| 100,000 | 0.353 |
| 200,000 | 0.348 |
| 440,000 | 0.347 |

## 設定
- Image Size: 64x64 → Latent: 8x8 (f=8)
- 44万ステップまで学習

---

# 生成サンプル: CIFAR-10

## @ 440,000 steps (Loss: 0.347)

![w:400](outputs/ldm_cifar10_T1000_cosine/samples/samples_00440000.png)

**評価**: 形状は認識できるが、**溶けたような画像**になっている

---

# CIFAR-10 の問題点

## Latent Diffusion が逆効果に

1. **元画像が低解像度 (32x32)**
   - VAE で圧縮 → 8x8 の潜在空間
   - 情報量が極端に少ない

2. **潜在空間での学習の限界**
   - 高解像度画像向けに設計された VAE
   - 低解像度では圧縮が過剰

3. **結果**
   - 44万ステップ学習しても画質改善せず
   - **溶けたようなぼやけた画像**しか生成できない

---

# Loss カーブの比較

## CelebA-HQ (M4 Mac)
```
Step 1000:  Loss = 0.577  (急激に低下)
Step 5000:  Loss = 0.304  (収束開始)
Step 15000: Loss = 0.276  (ほぼ収束)
```

## CIFAR-10
```
Step 10000:  Loss = 0.353
Step 100000: Loss = 0.353  ← ほぼ横ばい
Step 440000: Loss = 0.347  (改善わずか)
```

**注目**: CIFAR は Loss が下がっても画質は改善しない

---

# 考察: 主要な知見

## CelebA-HQ (本実験)
- M4 Mac では **計算リソース不足** で十分な学習ができない
- Colab では進捗があったが **セッション制限で中断**
- 継続的な長時間学習が必要

## CIFAR-10 (比較実験)
- **低解像度データに Latent Diffusion は不向き**
- VAE の圧縮が過剰になり、情報が失われる
- Pixel-space Diffusion の方が適切

---

# 知見のまとめ

## Latent Diffusion の適用条件

| 条件 | 推奨 |
|------|------|
| 高解像度画像 (128px+) | Latent Diffusion |
| 低解像度画像 (32px) | Pixel-space Diffusion |

## 理由
- VAE は高解像度画像の圧縮に最適化
- 低解像度では潜在空間が小さすぎて情報が失われる
- CIFAR-10 は Pixel-space で学習すべき

---

# 今後の課題

## CelebA-HQ 実験の継続
- [ ] Colab Pro+ 等で長時間学習
- [ ] チェックポイントからの再開
- [ ] EMA (Exponential Moving Average) の活用

## 評価の改善
- [ ] FID スコアによる定量評価
- [ ] 生成画像の多様性評価

## CIFAR-10 の再実験
- [ ] Pixel-space Diffusion での学習
- [ ] Latent との比較検証

---

# まとめ

## 実験の成果
1. LDM (DDIM) の実装と学習パイプラインの構築
2. Wandb によるメトリクス・サンプル画像の記録
3. 異なる環境・データセットでの学習挙動の比較

## 主要な知見
- **CelebA-HQ**: 学習は進むが十分なステップ数が必要
- **CIFAR-10**: **低解像度データには Latent Diffusion は逆効果**
  - Loss は下がるが画像は溶けたまま改善しない

---

# 参考文献

1. Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models"
2. Song, J., et al. (2020). "Denoising Diffusion Implicit Models"
3. Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models"

---

# 付録: 実験ログ

## Wandb Run IDs
- CelebA-HQ (M4 Mac): `run-20260111_170040-0nti4sdc`
- CIFAR-10 (M4 Mac): `run-20260111_000550-mr2wzfih`

## 出力ディレクトリ
- `outputs/ldm_celeba_hq_256_T1000_cosine/`
- `outputs/ldm_cifar10_T1000_cosine/`

