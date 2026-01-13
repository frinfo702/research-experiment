# 使い方
```
cd DDPM
```

# 依存関係インストール
```
uv sync
```

# ローカルで動作確認（M4 Mac）
デフォルトは速度と品質をバランス重視で実行する`balanced`になっている

```
./scripts/run_baseline.sh # 
```

速度と品質のトレードオフで切り替えができる

```
PROFILE=turbo   ./scripts/run_baseline.sh
PROFILE=fast    ./scripts/run_baseline.sh
PROFILE=balanced ./scripts/run_baseline.sh
PROFILE=quality ./scripts/run_baseline.sh

```

# 本番学習（クラウド）
```
uv run python main.py train --timesteps 1000 --beta-schedule linear --device cuda
```

# サンプル生成
```
uv run python main.py sample --checkpoint outputs/ddpm_baseline/checkpoints/checkpoint_latest.pt
```

# FID評価
```
uv run python main.py evaluate --checkpoint outputs/ddpm_baseline/checkpoints/checkpoint_latest.pt
```

# Ablation実験
```
uv run python main.py ablation train
uv run python main.py ablation evaluate
```
