---
marp: true
theme: academic
paginate: true
math: mathjax
---

# Latent Diffusion Model (LDM) 実験

2026-01-12
後藤 健一郎

---

# 目次

1. はじめに
2. 手法
3. 実験設定
4. 実験結果
5. 考察
6. 結論

---

# はじめに

## LDM と DDIM を選定した理由
- LDM: 画像を潜在空間で学習することで計算量とメモリを削減しつつ高解像度生成を狙える
- DDIM: 生成ステップを短縮しやすく、決定論的サンプリングで比較がしやすい

## 実験の目的
- LDM の再現実験 (DDIM 生成を含む)
- 計算環境・データセット差による学習挙動の比較

---

# 手法

## LDM のアーキテクチャ概要
- **VAE で画像を潜在空間に圧縮** → 潜在空間で拡散モデルを学習 → VAE で復元
- U-Net + Attention により潜在表現の局所・大域情報を統合
- ノイズスケジュール: cosine (T=1000)

## 潜在空間のサイズ例
- 128px 画像 + f=4 → 32x32 latent
- 256px 画像 + f=8 → 32x32 latent

---

# 手法

## DDIM サンプリングの利点
- 少ないステップ数でも画質が保ちやすい
- 決定論的にサンプルが得られ、比較実験に向く
- 生成時間と品質のトレードオフを制御しやすい

---

# 実験設定

| 項目 | Colab | Mac |
|------|-------|-----|
| GPU/CPU | T4 GPU | M4 chip |
| データセット | CelebA-HQ | CelebA-HQ |
| 解像度 | 128px | 256px |
| 圧縮率 | f=8 | f=8 |
| Latent | 16x16 | 32x32 |
| VAE | diffusers AutoencoderKL | stabilityai/sd-vae-ft-mse |
| Batch | 64 | 8 |
| LR | 2e-4 | 2e-4 |
| Timesteps | 1000 | 1000 |
| Schedule | cosine | cosine |
| Sampler | DDIM | DDIM |

---

# 実験結果

## CelebA-HQ
- 顔の輪郭や髪型の構造は復元される傾向
- 128px + f=4 では表情の局所構造が比較的残りやすい
- 256px + f=8 は計算負荷が高く、学習が遅い

---

# 実験結果

## CIFAR-10
- 図x のように、形状は識別できるが**溶けたような画像**になる
- Loss は下がるが、視覚的な品質改善に結びつかない

---

# 考察

## CIFAR-10 が溶けたようになった理由
- 32x32 画像を VAE で圧縮すると **潜在空間が極端に小さくなる**
  - 例: f=8 → 4x4 latent となり情報量が大幅に欠落
- 高解像度用 VAE の圧縮設計が低解像度に不適合
- 潜在空間での拡散学習では、欠落情報を復元できない

---

# 考察

## CelebA-HQ が相対的に成功する理由
- 128px 以上では f=4, 8 でも latent が 32x32 程度確保できるため情報保持が可能
- 低解像度 (32px) では Pixel-space Diffusion の方が適切
- f=4 の設定は顔の細部や建物の柱などの局所構造を保ちやすい
- ただし、長時間学習が前提 (colab 6h, M4 13h) となり、計算資源がボトルネック

---

# 結論

## まとめ
- LDM は 128px 以上の画像で有効
- 圧縮率は latent が 32x32 以上になるように選ぶのが重要

---

# 参考文献
1. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. Advances in Neural Information Processing Systems (NeurIPS).
2. Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models*. International Conference on Learning Representations (ICLR).
3. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
