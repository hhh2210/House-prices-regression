# LightGBM GPU 贝叶斯优化使用指南

## 新增功能

已成功集成 **Optuna** 贝叶斯优化框架，相比原有的 RandomizedSearchCV，具有以下优势：

### Optuna 优势
- 🚀 **更智能的搜索**：使用 TPE (Tree-structured Parzen Estimator) 算法，而非随机搜索
- ✂️ **自动剪枝**：通过 MedianPruner 提前终止表现差的试验，节省时间
- 📊 **实时进度**：显示最佳得分和完成/剪枝试验统计
- 🎯 **更高效率**：通常比随机搜索快 3-10 倍找到最优参数

## 使用命令

### 1. LightGBM + GPU + Optuna 优化（推荐）
```bash
# 基础版本 - 50 次试验（默认）
uv run python src/train_tree.py --model lgbm --gpu --tune

# 深度搜索 - 100 次试验
uv run python src/train_tree.py --model lgbm --gpu --tune --n_iter 100

# 包含 AmesHousing 数据增强
uv run python src/train_tree.py --model lgbm --gpu --tune  --include-ames --n_jobs 10
```

### 2. 使用传统 RandomizedSearchCV
```bash
python src/train_tree.py --model lgbm --gpu --tune --tune-method random --n_iter 100
```

### 3. 其他模型支持
```bash
# XGBoost + GPU
python src/train_tree.py --model xgb --gpu --tune --n_iter 50

# RandomForest（无 GPU 加速）
python src/train_tree.py --model rf --tune --n_iter 50

# HistGradientBoosting
python src/train_tree.py --model hgb --tune --n_iter 50
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `hgb` | 模型类型：`lgbm`, `xgb`, `rf`, `hgb` |
| `--gpu` | `False` | 启用 GPU 加速（仅 LightGBM/XGBoost） |
| `--tune` | `False` | 启用超参数优化 |
| `--tune-method` | `optuna` | 优化方法：`optuna` 或 `random` |
| `--n_iter` | `50` | 试验次数（optuna）或迭代次数（random） |
| `--folds` | `5` | 交叉验证折数 |
| `--include-ames` | `False` | 使用 AmesHousing 数据增强 |

## LightGBM 参数搜索空间

Optuna 会在以下范围内智能搜索最佳参数：

```python
{
    "n_estimators": [1000, 5000],          # 树的数量
    "learning_rate": [0.01, 0.2],          # 学习率（对数分布）
    "num_leaves": [15, 127],               # 叶子节点数
    "subsample": [0.5, 1.0],               # 样本采样率
    "colsample_bytree": [0.5, 1.0],        # 特征采样率
    "reg_lambda": [0.1, 2.0],              # L2 正则化
    "min_child_samples": [5, 50],          # 叶子最小样本数
}
```

## 输出示例

```
============================================================
Optuna Bayesian Optimization: 50 trials × 5 folds
Using TPE sampler with median pruner
============================================================

Optuna Trials: 100%|████████████| 50/50 [15:32<00:00, best=0.1234]

Best CV RMSE(log): 0.123456
Best params: {'n_estimators': 2500, 'learning_rate': 0.0345, ...}

Optuna Statistics:
  Completed trials: 42
  Pruned trials: 8
```

## 预计耗时

使用 GPU 加速的情况下：

| 试验次数 | 预计耗时 |
|----------|----------|
| 50 次 | 10-20 分钟 |
| 100 次 | 20-40 分钟 |
| 200 次 | 40-80 分钟 |

*注：剪枝机制会显著减少实际耗时，表现差的试验会提前终止。*

## 最佳实践

1. **初次探索**：使用 50 次试验快速找到较优区域
2. **精细调优**：在好的区域基础上运行 100-200 次试验
3. **GPU 加速**：确保安装了支持 GPU 的 LightGBM 版本
4. **数据增强**：`--include-ames` 可能提升泛化能力

## 技术细节

- **采样器**：TPESampler with seed for reproducibility
- **剪枝器**：MedianPruner (n_startup_trials=5, n_warmup_steps=2)
- **优化方向**：minimize RMSE(log)
- **CV策略**：KFold with shuffle

## 依赖项

```toml
optuna>=4.1.0
lightgbm>=4.6.0
xgboost  # 可选，仅使用 XGBoost 时需要
```
