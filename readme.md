Background:
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

1. Macbook m4
2. GPU with CUDA

思路与MPS加速实践（PyTorch）

- 目标与指标
  - 对 `SalePrice` 做对数变换（`log1p`），在对数空间训练；验证指标用对数空间的 RMSE（接近竞赛的 RMSLE）。
- 特征预处理
  - 数值列：中位数填补缺失 + 标准化。
  - 类别列：众数填补 + One-Hot 编码（忽略未知类别）。
  - 以上用 `sklearn` 的 `ColumnTransformer` 与 `Pipeline` 组合完成。
- 验证策略
  - 先用 `train/valid` 切分跑通（默认 90/10）。后续可切到 KFold 做更稳健的验证与集成。
- 模型（PyTorch）
  - 简单的 MLP 回归器（ReLU 多层全连接），损失为 MSE；优化器 Adam，`weight_decay` 做 L2 正则；`ReduceLROnPlateau` 与 early stopping。
- Apple Silicon/MPS 要点（适用于 MacBook M4）
  - 自动优先选择 `mps` 设备：若不可用回退到 CPU；也可用 `--device mps` 强制。
  - 使用 `float32`（已在脚本中确保），避免 `float64` 导致性能差。
  - DataLoader 默认 `num_workers=0` 更通用；需要时再升到 2/4 试速。
  - 如果显存不够，降低 `--batch_size`。

快速开始

1) 使用 uv 安装依赖（推荐）

```bash
# 安装 uv（二选一）
brew install uv                        # 如果你用 Homebrew
# 或者：
curl -LsSf https://astral.sh/uv/install.sh | sh

# 准备 Python 与虚拟环境（可选固定版本）
uv python install 3.11                 # 可选：安装并使用 3.11
uv venv -p 3.11                        # 或直接 `uv venv`

# 同步依赖（基于 pyproject.toml）
uv sync

# 也可沿用 requirements.txt（可选）
uv pip sync requirements.txt
```

2) 训练并生成提交文件（自动使用 MPS，如果可用）

```bash
# 无需手动激活虚拟环境，直接用 uv 运行
uv run python src/train_mps.py --device mps --epochs 200 --batch_size 512 --hidden_dim 256 --layers 3
```

树模型 + 特征工程（推荐的更强基线）

- 我们新增了 `src/train_tree.py`：包含常用特征工程（面积/房龄/总卫浴/门廊、质量有序映射、是否存在标记等）、偏度列 `log1p` 处理，以及 5 折 KFold 交叉验证。
- 默认模型 `--model hgb` 使用 `HistGradientBoostingRegressor`（sklearn 内置、无需额外依赖）；也可选 `rf`/`xgb`/`lgbm`（后两者需额外安装）。

运行示例：

```bash
# 5 折 CV + 生成提交（submissions/submission_tree_hgb.csv）
uv run python src/train_tree.py --model hgb --folds 5

# 切换随机森林（CPU 很快）
uv run python src/train_tree.py --model rf --folds 5

# 可选安装并使用 XGBoost/LightGBM（需要时再装）
uv add xgboost           # 如需 XGBoost
uv add lightgbm          # 如需 LightGBM
uv run python src/train_tree.py --model xgb --folds 5
uv run python src/train_tree.py --model lgbm --folds 5
```

CUDA/大显存 GPU（48GB）一键运行与调参

- MLP（PyTorch，混合精度 + 随机搜索 + K 折集成，推荐在 48GB 上运行）

```bash
# 一键：搜索 30 次 + 5 折集成，自动使用 CUDA AMP 与更高并行度
uv run python src/train_mps.py \
  --device cuda --amp \
  --search_trials 30 --kfolds 5 \
  --epochs 300 --num_workers 8 --batch_size 4096
```

- 树模型（sklearn，随机搜索）

```bash
# HistGradientBoosting 随机搜索 80 次 + 5 折
uv run python src/train_tree.py --model hgb --tune --n_iter 80 --folds 5

# 可选：安装并使用 XGBoost（如需 GPU 版）
uv add xgboost
uv run python src/train_tree.py --model xgb --tune --n_iter 80 --folds 5
```

可选项与提示

- `--remove_outliers/--no_remove_outliers`：是否按常见做法移除极端 `GrLivArea`（默认移除）。
- `--log_target`：对 `SalePrice` 取对数训练（默认开启，和竞赛 RMSLE 对齐）。
- CV 会打印每折与均值的 `RMSE(log)`，便于快速比较模型与特征改动。

- 读取数据路径：`data/train.csv` 和 `data/test.csv`
- 训练完成后，PyTorch 版本会输出 `submissions/submission_pytorch_mlp.csv`（若使用 K 折则带 `_k{K}` 后缀；若 `--log_target`，则已 `expm1` 还原）。
- 树模型会输出到 `submissions/submission_tree_{model}.csv`，若 `--tune` 则带 `_tuned` 后缀。

常用可调参数

- `--epochs`：训练轮数（默认 200，带 early stopping）。
- `--batch_size`：批大小（默认 512；显存不足请调小）。
- `--hidden_dim` / `--layers` / `--dropout`：MLP 结构。
- `--lr` / `--weight_decay`：优化器与正则。
- `--val_size`：验证集占比（默认 0.1）。
- `--device`：强制设备（`cpu|cuda|mps`）。
- `--log_target`：是否对 `SalePrice` 做 `log1p` 训练（默认开）。

下一步提升方向（按收益推荐）

- 验证与集成
  - 切换到 KFold（如 K=5），取各折平均预测；或对多个 MLP/结构做简单集成。
- 特征工程（强烈建议）
  - 面积类汇总：`TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
  - 年份衍生：`HouseAge = YrSold - YearBuilt`，`RemodAge = YrSold - YearRemodAdd`，`IsRemodeled` 标记。
  - 卫浴合并：`TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
  - 车库/地下室/门廊等是否存在布尔特征；社区（`Neighborhood`）均值编码（注意泄露，用交叉验证目标编码）。
- 数据清洗
  - 移除明显离群点（例如较小 `GrLivArea` 却异常高价的样本）。
  - 对偏度大的数值列做 Box-Cox/对数变换以提升线性可分性。
- 训练技巧
  - OneCycleLR/余弦退火等学习率策略；合适的 `dropout/weight_decay`；增大 batch size（若显存允许）。
- 模型对比
  - 与树模型（LightGBM/XGBoost/CatBoost）做基线对比，后续可做 stacking/blending。

代码位置

- 训练与提交脚本：`src/train_mps.py`
