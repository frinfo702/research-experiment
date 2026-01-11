# 使い方
```
cd DDPM
```

# 依存関係インストール
```
uv sync
```

# ローカルで動作確認（M4 Mac）
```
./scripts/run_baseline_debug.sh
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
