# House Prices Regression - Advanced Ensemble Solution

## Background
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

## 硬件支持
- ✅ Macbook M4 (Apple Silicon MPS)
- ✅ GPU with CUDA (推荐：大显存GPU如L20 48GB)
- ✅ CPU fallback

## 项目结构
```
House-prices-regression/
├── data/
│   ├── train.csv          # 训练数据
│   └── test.csv           # 测试数据
├── src/
│   ├── train_tree.py      # 树模型基线（XGBoost/LightGBM/RF/HGB）
│   ├── train_ensemble.py  # CPU/GPU ensemble (XGBoost + LightGBM)
│   ├── train_gpu_ensemble.py  # 完整GPU加速ensemble (NN + XGBoost + LightGBM)
│   └── train_mps.py       # PyTorch MLP (支持MPS/CUDA/CPU)
├── submissions/           # 生成的提交文件
├── pyproject.toml         # 依赖管理
└── readme.md
```

## 核心思路与方法

### 1. 特征工程（Feature Engineering）

- 目标与指标
  - 对 `SalePrice` 做对数变换（`log1p`），在对数空间训练；验证指标用对数空间的 RMSE（接近竞赛的 RMSLE）。
**基础特征**：
- 面积汇总：`TotalSF`, `TotalPorchSF`
- 时间特征：`HouseAge`, `RemodAge`, `IsRemodeled`, `GarageAge`
- 卫浴统计：`TotalBath`（全浴+半浴权重）
- 存在性标记：`HasBsmt`, `HasGarage`, `HasFireplace`, `HasPool`
- 质量有序编码：将 Ex/Gd/TA/Fa/Po 映射为数值

**高级特征**：
- 交互特征：`QualArea = OverallQual × GrLivArea`, `QualBathArea`, `BathArea`
- 多项式特征：关键数值特征的平方、立方
- 比率特征：`BathPerArea`, `LotAreaRatio`, `SFperBath`
- 偏度处理：对高偏度数值特征（>0.75）进行 log1p 转换

**数据清洗**：
- 领域规则填充：对特定缺失值按竞赛规则处理（如 NA → "None"）
- 离群点移除：去除 `GrLivArea >= 4000` 的异常样本
- 社区填充：`LotFrontage` 按 `Neighborhood` 中位数填充

### 2. 模型集成（Ensemble Learning）

采用 **Stacking** 策略，结合多个模型的优势：

**Level 1 Base Models**：
- **Deep Neural Network (PyTorch)**：512→256→128→64 全连接网络，BatchNorm + Dropout
- **XGBoost**：梯度提升树，GPU加速（device='cuda'）
- **LightGBM**：轻量级梯度提升，GPU训练

**Level 2 Meta-Learner**：
- **Ridge Regression**：基于 OOF predictions 的线性集成

**优势**：
- 神经网络捕获非线性交互
- 树模型处理稀疏和分类特征
- Ridge元学习器自动学习最优权重

### 3. GPU 加速优化

**训练加速**：
- XGBoost: `device='cuda'`, `tree_method='hist'`
- LightGBM: `device='gpu'`（should be careful with OpenCL for GPU）
- PyTorch: CUDA张量 + DataLoader批处理
- Early stopping：避免过拟合并节省时间

**显存优化**：
- 48GB L20 可同时训练大batch size（512-4096）
- 神经网络支持混合精度训练（AMP）
- 5折交叉验证并行处理


## 快速开始

### 1. 安装依赖

```bash
# 安装 uv 包管理器（二选一）
brew install uv                        # macOS Homebrew
# 或：
curl -LsSf https://astral.sh/uv/install.sh | sh

# 准备环境并安装依赖
uv python install 3.11                 # 安装 Python 3.11
uv sync                                # 同步依赖（基于 pyproject.toml）
```

### 2. 选择训练脚本

#### 🚀 方案一：GPU 完整 Ensemble（推荐，最佳性能）
**适用**：48GB GPU (L20/A100)，目标 RMSE ~0.11-0.12

```bash
# 深度神经网络 + XGBoost + LightGBM 三模型 stacking
uv run python src/train_gpu_ensemble.py --folds 5

# 训练集额外并入 AmesHousing.csv（可选）
uv run python src/train_gpu_ensemble.py --folds 5 --include-ames --ames-path data/AmesHousing.csv
```

**特点**：
- 5折交叉验证，自动stacking
- 完整特征工程（354维）
- GPU加速训练（NN + 树模型）
- 输出：`submissions/submission_gpu_ensemble.csv`

---

#### ⚡ 方案二：树模型基线（快速调参）
**适用**：CPU/GPU，快速迭代

```bash
# XGBoost（GPU加速 + 超参数搜索）
uv run python src/train_tree.py --model xgb --gpu --folds 5

# 训练集额外并入 AmesHousing.csv（可选）
uv run python src/train_tree.py --model xgb --gpu --folds 5 --include-ames --ames-path data/AmesHousing.csv

# 带超参数搜索（20次迭代）
uv run python src/train_tree.py --model xgb --tune --n_iter 20 --folds 3 --gpu

# LightGBM（需先安装）
uv add lightgbm
uv run python src/train_tree.py --model lgbm --gpu --folds 5

# HistGradientBoosting（sklearn，CPU友好）
uv run python src/train_tree.py --model hgb --folds 5
```

**参数说明**：
- `--gpu`: 启用GPU加速（XGBoost/LightGBM）
- `--tune`: 超参数搜索（RandomizedSearchCV）
- `--n_iter`: 搜索迭代次数（默认40）
- `--folds`: K折交叉验证折数（默认5）
- `--include-ames`: 训练集并入 `AmesHousing.csv`
- `--ames-path`: 指定 AmesHousing 数据路径（默认 `data/AmesHousing.csv`）

---

#### 🍎 方案三：PyTorch MLP（Apple Silicon 优化）
**适用**：MacBook M4 (MPS加速)

```bash
# 使用 MPS 加速
uv run python src/train_mps.py --device mps --epochs 200 --batch_size 512

# CUDA GPU
uv run python src/train_mps.py --device cuda --epochs 300 --batch_size 1024

# 训练集额外并入 AmesHousing.csv（可选）
uv run python src/train_mps.py --device mps --include-ames --ames-path data/AmesHousing.csv
```

---

## 输出文件

训练完成后，提交文件保存在 `submissions/` 目录：

| 文件名 | 对应脚本 | 说明 |
|--------|---------|------|
| `submission_gpu_ensemble.csv` | `train_gpu_ensemble.py` | GPU完整ensemble |
| `submission_tree_xgb_tuned.csv` | `train_tree.py --model xgb --tune` | XGBoost调参版 |
| `submission_tree_hgb.csv` | `train_tree.py --model hgb` | HGB基线 |
| `submission_pytorch_mlp.csv` | `train_mps.py` | PyTorch MLP |

## 可调参数

### train_gpu_ensemble.py
- `--folds`: K折交叉验证数（默认5）
- `--seed`: 随机种子（默认42）

### train_tree.py
- `--model`: 模型选择（hgb/rf/xgb/lgbm，默认hgb）
- `--folds`: 交叉验证折数（默认5）
- `--tune`: 启用超参数搜索
- `--n_iter`: 搜索迭代次数（默认40）
- `--gpu`: 启用GPU加速
- `--log_target`: 目标对数变换（默认开启）
- `--remove_outliers`: 移除离群点（默认开启）

### train_mps.py
- `--device`: 设备选择（cpu/cuda/mps，默认自动）
- `--epochs`: 训练轮数（默认200）
- `--batch_size`: 批大小（默认512）
- `--hidden_dim`: 隐藏层维度（默认256）
- `--layers`: 隐藏层数（默认3）
- `--lr`: 学习率（默认0.001）
- `--dropout`: Dropout率（默认0.3）

## 性能优化建议

### 如何进一步提升 (目标 < 0.11)

1. **增加模型多样性**
   - 添加CatBoost到ensemble
   - 尝试不同的neural network架构
   - 使用TabNet等表格专用模型

2. **高级特征工程**
   - Target encoding for neighborhood
   - 更多交互特征组合
   - 时序特征（建造年份周期性）
   - 外部数据融合

3. **优化stacking策略**
   - 多层stacking（Level 3）
   - 使用更复杂的meta-learner（如LightGBM）
   - Out-of-fold predictions optimization

4. **超参数深度调优**
   - Optuna/Hyperopt替代RandomizedSearchCV
   - 增加搜索空间和迭代次数
   - 针对ensemble权重的grid search


## 项目依赖

核心库：
- `numpy`, `pandas`: 数据处理
- `scikit-learn`: 预处理、CV、meta-learner
- `torch`: 神经网络训练
- `xgboost`: 梯度提升树（GPU支持）
- `lightgbm`: 轻量级GBDT（GPU支持）
- `tqdm`: 进度条

## 参考资料

- [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/latest/gpu/)
- [LightGBM GPU Tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

## License

MIT
