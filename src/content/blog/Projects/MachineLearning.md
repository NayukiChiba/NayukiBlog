---
title: 机器学习全汇总项目
date: 2026-05-24
category: 项目
tags:
  - 高级教程
  - Scikit-learn
  - 总结
description: 机器学习全汇总项目架构文档——目录结构、核心抽象与模块分层。
image: https://img.yumeko.site/file/blog/cover/1780581808315.webp
status: published
---


# 项目架构

## 本章目标

1. 理解项目的整体目录结构和模块分层。
2. 建立从源码到文档的全局导航。

本项目是一个机器学习算法教学代码库，覆盖分类、回归、聚类、降维、概率模型和集成学习六大任务类型。每个算法配有 9 篇文档，代码遵循**声明式流水线架构**。

**核心哲学**：显式优于隐式，教学清晰度优于工程复用度。所有步骤（加载、切分、预处理、训练、评估、可视化）在 Runner 中显式编排。

---

## 1. 目录结构

```
Machine-Learning-Algorithms/
├── main.py                          # CLI 统一入口
├── config.py                        # 路径常量与配置
├── src/
│   └── mlAlgorithms/
│       ├── core/                    # 核心抽象
│       │   ├── pipelineSpec.py      #   PipelineSpec —— 流水线声明
│       │   ├── datasetSpec.py       #   DatasetSpec —— 数据集声明
│       │   ├── registry.py          #   Registry —— 简单注册表
│       │   ├── runContext.py        #   RunContext —— 运行时上下文
│       │   ├── runResult.py         #   RunResult —— 运行结果
│       │   ├── taskTypes.py         #   枚举：TaskType / RunnerType / DataKind
│       │   └── artifactManager.py   #   产物路径管理
│       ├── catalog/                 # 注册表（声明式配置聚合）
│       │   ├── pipelines.py         #   PIPELINE_REGISTRY —— 所有流水线声明
│       │   └── datasets.py          #   DATASET_REGISTRY —— 所有数据集声明
│       ├── datasets/                # 数据层
│       │   ├── datasetCatalog.py    #   统一构建所有 DatasetSpec
│       │   ├── tabular/             #   表格数据工厂
│       │   └── sequence/            #   序列数据工厂（HMM）
│       ├── training/                # 训练层
│       │   ├── classification/      #   分类模型
│       │   ├── regression/          #   回归模型
│       │   ├── clustering/          #   聚类模型
│       │   ├── dimensionality/      #   降维模型
│       │   └── probabilistic/       #   概率模型（GMM / HMM）
│       ├── workflows/               # 运行器层
│       │   ├── executor.py          #   按 RunnerType 分发
│       │   ├── baseRunner.py        #   共享辅助函数
│       │   ├── classificationRunner.py
│       │   ├── regressionRunner.py
│       │   ├── clusteringRunner.py
│       │   ├── dimensionalityRunner.py
│       │   └── probabilisticRunner.py
│       ├── evaluation/              # 评估层
│       ├── visualization/           # 可视化层
│       │   ├── data/                #   训练前数据可视化
│       │   └── result/              #   训练后结果可视化
│       └── analysis/                # 数据探索层
├── docs/                            # 文档
│   ├── appendix/                    #   项目架构（本目录）
│   ├── foundations/                 #   基础库教程
│   ├── classification/              #   分类算法文档
│   ├── regression/                  #   回归算法文档
│   ├── clustering/                  #   聚类算法文档
│   ├── ensemble/                    #   集成学习文档
│   ├── dimensionality/              #   降维算法文档
│   └── probabilistic/               #   概率模型文档
└── outputs/                         # 运行产物（图像/报告）
```

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                       main.py (CLI)                         │
│          list / run / suite / analyze                       │
└─────────────────────┬───────────────────────────────────────┘
                      │ 查找
          ┌───────────┴───────────┐
          │   PIPELINE_REGISTRY   │  <- PipelineSpec x 20
          │   DATASET_REGISTRY    │  <- DatasetSpec x 20
          └───────────┬───────────┘
                      │ 分发
          ┌───────────┴───────────┐
          │     executor.py       │  <- 按 RunnerType 路由
          └───────────┬───────────┘
                      │
     ┌──────┬──────┬──┴───┬──────┬──────────┐
     │      │      │      │      │          │
  Class.  Regr.  Clust.  Dim.  Prob.    (Runner)
     │      │      │      │      │
     └──────┴──────┴──────┴──────┴──────────┘
                      │
          ┌───────────┴───────────┐
          │  buildRunContext()    │  <- 数据加载 + 探索
          │  makeSplit()          │  <- 切分
          │  applyPreprocessor()  │  <- 标准化
          │  callTrainer()        │  <- 训练
          │  evaluate()           │  <- 评估 + 打印
          │  plot*()              │  <- 可视化
          └───────────────────────┘
```

---

# 核心抽象

## 本章目标

1. 理解 `PipelineSpec` 的全部 16 个字段及其配置方式。
2. 理解 `DatasetSpec`、`Registry`、`RunContext`、`RunResult` 的职责。
3. 掌握 `TaskType`、`RunnerType`、`DataKind` 三种枚举。

---

## 1. PipelineSpec —— 流水线声明

`PipelineSpec` 是本项目最核心的数据类。一个实例完整描述一条可执行算法流水线——数据来源、训练函数、预处理、评估和可视化全部在此声明。

**定义**：[`src/mlAlgorithms/core/pipelineSpec.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/src/mlAlgorithms/core/pipelineSpec.py)

### 1.1 字段速览

| 字段                     | 类型              | 说明                                                   | 示例                        |
| ---------------------- | --------------- | ---------------------------------------------------- | ------------------------- |
| `id`                   | `str`           | 流水线唯一标识，格式 `{domain}.{algorithm}`                    | `"regression.svr"`        |
| `taskType`             | `TaskType`      | 任务类型——决定数据探索报告的格式                                    | `TaskType.REGRESSION`     |
| `datasetId`            | `str`           | 关联的数据集 ID——必须与 `DatasetSpec.id` 一致                   | `"regression.svr"`        |
| `runnerType`           | `RunnerType`    | 运行器类型——`executor.py` 据此分发                            | `RunnerType.REGRESSION`   |
| `trainer`              | `Callable`      | 训练函数——接收训练数据，返回模型（或 `dict`）                          | `trainSvrRegressionModel` |
| `preprocessor`         | `str `\|`None`  | 预处理方式——`"standardScaler"` 或 `None`                   | `"standardScaler"`        |
| `splitter`             | `str`\|`None`   | 切分策略——`"randomSplit"` / `"stratifiedSplit"` / `None` | `"randomSplit"`           |
| `predictor`            | `str` \| `None` | 后处理策略——分类/聚类/LDA 特有                                  | `"default"`               |
| `evaluator`            | `str` \| `None` | 评估配置名称                                               | `"default"`               |
| `analysisProfile`      | `str`           | 分析报告类型——按 TaskType 选择合适的分析器                          | `"regression"`            |
| `dataPlots`            | `list[str]`     | 训练前数据可视化列表                                           | `["correlationHeatmap"]`  |
| `resultPlots`          | `list[str]`     | 训练后结果可视化列表                                           | `["featureImportance"]`   |
| `diagnostics`          | `list[str]`     | 诊断性可视化列表                                             | `["learningCurve"]`       |
| `outputKey`            | `str`           | 输出子目录名——产物存放到 `outputs/{outputKey}/`                 | `"svr"`                   |
| `optionalDependencies` | `tuple[str]`    | 可选依赖——缺失时跳过而非崩溃                                      | `("hmmlearn",)`           |
| `metadata`             | `dict`          | 额外配置——多模型标记、工厂函数等                                    | `{"multiModel": True}`    |

### 1.2 注册示例

```python
PipelineSpec(
    "regression.linear_regression",    # pipeline ID
    TaskType.REGRESSION,               # 任务类型
    "regression.linear_regression",    # dataset ID
    RunnerType.REGRESSION,             # 运行器类型
    trainLinearRegressionModel,        # 训练函数
    None,                               # 预处理 — 无标准化
    "randomSplit",                      # 切分策略
    "default",                          # 后处理
    "regression",                       # analysisProfile
    "regression",                       # evaluator
    ["correlationHeatmap", "featureTargetScatter"],  # dataPlots
    ["featureImportance"],              # resultPlots
    ["learningCurve"],                  # diagnostics
    "linear_regression",                # outputKey
    metadata={
        "learningCurveEstimatorFactory": _buildLearningCurveFactory(
            "regression.linear_regression"
        )
    },
)
```

### 1.3 理解重点

- 一个 `PipelineSpec` 就是一个算法的**完整配置清单**——Runner 不需要任何额外信息即可执行。
- `preprocessor`、`dataPlots`、`resultPlots`、`diagnostics` 都是**声明式列表**——Runner 遍历列表逐项执行，新增可视化只需在列表中添加名称。
- `metadata` 是扩展点——`multiModel`、`learningCurveEstimatorFactory`、`visualModelFactory` 等特殊需求都通过它传递。

---

## 2. DatasetSpec —— 数据集声明

描述一个数据集的加载方式与元信息。

**定义**：[`src/mlAlgorithms/core/datasetSpec.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/src/mlAlgorithms/core/datasetSpec.py)

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `str` | 数据集唯一标识 |
| `taskType` | `TaskType` | 所属任务类型 |
| `dataKind` | `DataKind` | 数据形态——`TABULAR` 或 `SEQUENCE` |
| `loader` | `Callable[[], DataFrame]` | 数据加载函数——每次调用返回新 DataFrame |
| `targetColumn` | `str \| None` | 标签列名 |
| `featureColumns` | `list[str] \| None` | 手动指定特征列——`None` 时自动从 targetColumn 推断 |
| `description` | `str` | 数据集中文描述 |

**关键方法**：

| 方法 | 说明 |
|---|---|
| `load()` | 调用 `loader()` 返回新 DataFrame |
| `resolveFeatureColumns(data)` | 根据 `featureColumns` / `targetColumn` 解析特征列名 |

### 理解重点

- `loader` 每次调用返回**全新** DataFrame——避免多次运行之间的状态污染。
- `featureColumns=None` 时自动推断：排除 `targetColumn` 外的所有列即为特征列。
- `dataKind=SEQUENCE` 仅用于 HMM——影响数据探索报告的生成方式。

---

## 3. Registry —— 简单注册表

基于字典的泛型注册表，是 `PIPELINE_REGISTRY` 和 `DATASET_REGISTRY` 的底层实现。

**定义**：[`src/mlAlgorithms/core/registry.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/src/mlAlgorithms/core/registry.py)

| 方法 | 说明 |
|---|---|
| `register(itemId, item)` | 注册对象——重复 ID 抛出 `KeyError` |
| `get(itemId)` | 获取对象——未注册抛出 `KeyError` |
| `keys()` | 返回所有已注册 ID |
| `values()` | 返回所有已注册对象 |
| `contains(itemId)` | 判断条目是否已注册 |

### 理解重点

- `Registry` 是泛型类——`Registry[PipelineSpec]` 和 `Registry[DatasetSpec]` 共享同一实现。
- 两条 `Registry` 在模块导入时完成注册——CLI 启动时即可直接查询。

---

## 4. RunContext —— 运行时上下文

一次流水线运行的共享状态容器——贯穿 Runner 的整个执行周期。

**定义**：[`src/mlAlgorithms/core/runContext.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/src/mlAlgorithms/core/runContext.py)

| 字段 | 类型 | 说明 |
|---|---|---|
| `spec` | `PipelineSpec` | 当前执行的流水线声明 |
| `datasetSpec` | `DatasetSpec` | 关联的数据集声明 |
| `data` | `DataFrame` | 完整原始数据 |
| `features` | `DataFrame \| None` | 特征列子集 |
| `target` | `Series \| None` | 标签列——无监督任务为 `None` |
| `outputDir` | `Path` | 产物输出目录 |
| `randomState` | `int` | 全局随机种子（42） |
| `analysisReport` | `Any \| None` | 数据探索报告——`runAnalysis()` 填充 |
| `extras` | `dict` | 扩展字段——Runner 间传递额外数据 |

### 理解重点

- `RunContext` 由 `buildRunContext()` 创建——加载数据、解析特征/标签、创建输出目录。
- `analysisReport` 在 Runner 执行早期填充——后续步骤可访问探索结果。

---

## 5. RunResult —— 运行结果

一次流水线执行的产物容器。

**定义**：[`src/mlAlgorithms/core/runResult.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/src/mlAlgorithms/core/runResult.py)

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | `Any` | 训练完成的模型——或 `dict[str, 模型]`（多模型模式） |
| `predictions` | `Any \| None` | 预测值数组 |
| `scores` | `Any \| None` | 预测分数（`predict_proba` / `decision_function`） |
| `metrics` | `dict` | 评估指标字典 |
| `artifacts` | `list[Path]` | 产物文件路径列表（PNG 图像等） |
| `extras` | `dict` | 扩展字段 |

### 理解重点

- `artifacts` 逐步累积——每生成一张图就 `appendArtifact()` 追加。
- 多模型模式下 `model` 是 `dict`，`metrics` 的键与模型名对应。

---

## 6. 枚举类型

**定义**：[`src/mlAlgorithms/core/taskTypes.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/src/mlAlgorithms/core/taskTypes.py)

### 6.1 TaskType（任务类型）

决定数据探索报告的生成方式和算法的领域归属。

| 枚举值 | 含义 | 包含算法 |
|---|---|---|
| `CLASSIFICATION` | 分类 | 逻辑回归、决策树、SVC、朴素贝叶斯、KNN、随机森林、Bagging、GBDT、LightGBM |
| `REGRESSION` | 回归 | 线性回归、SVR、决策树回归、正则化回归、XGBoost |
| `CLUSTERING` | 聚类 | KMeans、DBSCAN |
| `DIMENSIONALITY` | 降维 | PCA、LDA |
| `PROBABILISTIC` | 概率模型 | GMM（EM）、HMM |

### 6.2 RunnerType（运行器类型）

与 `TaskType` 值一一对应。`executor.py` 根据它分发到对应的 Runner 函数：

```
RunnerType.CLASSIFICATION   -> runClassificationPipeline()
RunnerType.REGRESSION       -> runRegressionPipeline()
RunnerType.CLUSTERING       -> runClusteringPipeline()
RunnerType.DIMENSIONALITY   -> runDimensionalityPipeline()
RunnerType.PROBABILISTIC    -> runProbabilisticPipeline()
```

### 6.3 DataKind（数据形态）

| 枚举值 | 含义 | 使用场景 |
|---|---|---|
| `TABULAR` | 表格数据——每行一个样本，每列一个特征 | 除 HMM 外的所有算法 |
| `SEQUENCE` | 序列数据——不等长观测序列 | HMM |

# 模块分层

## 本章目标

1. 理解六层模块的职责边界——数据层、训练层、流水线注册层、运行器层、评估层、可视化层。
2. 理清各层之间的调用关系和数据流向。

---

## 1. 数据层（`datasets/`）

**职责**：生成或加载数据集，返回 `pandas.DataFrame`。不涉及任何预处理或切分。

### 1.1 核心文件

| 文件 | 职责 |
|---|---|
| `datasetCatalog.py` | `buildDatasetSpecs()` 构建全部 20 个 `DatasetSpec` |
| `tabular/classificationDatasets.py` | `ClassificationDatasetFactory` —— 6 个分类数据集 |
| `tabular/regressionDatasets.py` | `RegressionDatasetFactory` —— 4 个回归数据集 |
| `tabular/clusteringDatasets.py` | `ClusteringDatasetFactory` —— 2 个聚类数据集 |
| `tabular/ensembleDatasets.py` | `EnsembleDatasetFactory` —— 4 个集成数据集 |
| `tabular/dimensionalityDatasets.py` | `DimensionalityDatasetFactory` —— 2 个降维数据集 |
| `sequence/probabilisticDatasets.py` | `ProbabilisticDatasetFactory` —— 2 个概率模型数据集 |

### 1.2 数据工厂一览

| 工厂类 | 负责的数据集 |
|---|---|
| `ClassificationDatasetFactory` | 逻辑回归（线性可分二分类）、决策树（blob 多分类）、SVC（同心圆二分类）、朴素贝叶斯（Iris）、KNN（双月牙二分类）、随机森林（高维多分类） |
| `RegressionDatasetFactory` | 线性回归（合成线性房价）、SVR（Friedman1 非线性）、决策树回归（California Housing）、正则化回归（diabetes + 共线 + 噪声） |
| `ClusteringDatasetFactory` | KMeans（球形多簇）、DBSCAN（双月牙非线性） |
| `EnsembleDatasetFactory` | Bagging（高噪声双月牙）、GBDT（中等难度多分类）、XGBoost（回归）、LightGBM（高维四分类） |
| `DimensionalityDatasetFactory` | PCA（高维低秩合成）、LDA（Wine） |
| `ProbabilisticDatasetFactory` | EM/GMM（混合高斯）、HMM（离散序列） |

### 1.3 关键设计

- 每个 `load*Dataset()` 方法独立生成数据——**无共享状态**，无全局变量。
- 数据在 `DatasetSpec.load()` 时才会实际生成——**惰性加载**。
- 所有数据方法使用 `random_state` 参数保证**可复现性**。

---

## 2. 训练层（`training/`）

**职责**：封装 scikit-learn（及 hmmlearn / xgboost / lightgbm）模型的构建与 `fit()` 调用。**不包含预处理、切分和评估**。

### 2.1 核心文件

| 文件 | 包含的训练函数 |
|---|---|
| `classification/classificationModels.py` | 逻辑回归、决策树、SVC、朴素贝叶斯、KNN、随机森林、Bagging、GBDT、LightGBM |
| `regression/regressionModels.py` | 线性回归、SVR、决策树回归、正则化回归（Lasso/Ridge/ElasticNet）、XGBoost |
| `clustering/clusteringModels.py` | KMeans、DBSCAN |
| `dimensionality/dimensionalityModels.py` | PCA、LDA |
| `probabilistic/probabilisticModels.py` | GMM（EM）、HMM |

### 2.2 训练函数一览（回归示例）

| 函数 | 模型类 | 核心超参数 | 行数 | 特殊性 |
|---|---|---|---|---|
| `trainLinearRegressionModel` | `LinearRegression` | 无 | 3 | 无参构造 |
| `trainSvrRegressionModel` | `SVR` | `C=10.0, epsilon=0.1, kernel='rbf', gamma='scale'` | 2 | 本仓库最短 |
| `trainDecisionTreeRegressionModel` | `DecisionTreeRegressor` | `max_depth=6, min_samples_split=6, min_samples_leaf=3` | ~5 | — |
| `trainRegularizationModels` | `Lasso / Ridge / ElasticNet` | `alpha=0.15/2.0/0.2, l1_ratio=0.5` | ~10 | 返回 `dict` |
| `trainXgboostRegressionModel` | `XGBRegressor` | `n_estimators=300, learning_rate=0.05, ...` | ~15 | 可选依赖 |

### 2.3 关键设计

- 训练函数是**薄封装**——仅构建 + `fit()`，不做数据预处理或评估。
- 签名不强制统一——`callTrainer()` 通过 `inspect.signature` 按需过滤关键字参数。
- 正则化回归的训练函数返回 `dict[str, 模型]`——触发 Runner 的多模型模式。

---

## 3. 流水线注册层（`catalog/`）

**职责**：将所有 `PipelineSpec` 和 `DatasetSpec` 集中注册到全局 `Registry`。

### 3.1 核心文件

| 文件 | 职责 |
|---|---|
| `pipelines.py` | 声明全部 20 条 `PipelineSpec` + 工厂函数 |
| `datasets.py` | 调用 `buildDatasetSpecs()` 注册全部 `DatasetSpec` |

### 3.2 工厂函数

| 工厂 | 用途 | 说明 |
|---|---|---|
| `_buildLearningCurveFactory(pipelineId)` | 学习曲线 | 返回创建**未训练**模型实例的 lambda——供 CV 内部多次 fit |
| `_buildVisualModelFactory(pipelineId)` | 决策边界可视化 | 返回创建模型实例的 lambda——供二维投影上的边界拟合 |

### 3.3 关键设计

- 两条 `Registry` 是**全局单例**——CLI 入口通过它们查找流水线和数据集。
- 工厂函数将模型构造参数固化在 lambda 中——确保学习曲线和可视化使用的超参数与训练一致。
- 新增算法的 `PipelineSpec` 直接追加到 `PIPELINE_REGISTRY` 的注册循环中。

---

## 4. 运行器层（`workflows/`）

**职责**：编排一次完整流水线执行——从数据加载到产物输出。是项目的**执行核心**。

### 4.1 核心文件

| 文件 | 职责 |
|---|---|
| `executor.py` | 按 `RunnerType` 分发到对应的 Runner 函数 |
| `baseRunner.py` | 共享辅助函数——数据加载、切分、标准化、训练调用、二维投影等 |
| `classificationRunner.py` | 分类流水线——含决策边界和 ROC 逻辑 |
| `regressionRunner.py` | 回归流水线——含多模型模式和学习曲线逻辑 |
| `clusteringRunner.py` | 聚类流水线——含标签对齐和参数扫描逻辑 |
| `dimensionalityRunner.py` | 降维流水线——含 PCA 训练曲线和 LDA 分类评估 |
| `probabilisticRunner.py` | 概率模型流水线——含 GMM 分量扫描和 HMM 解码 |

### 4.2 执行链（以回归为例）

```
buildRunContext()              # 加载数据 -> 解析特征/标签 -> 创建输出目录
- runAnalysis()             #   数据探索 -> 终端打印报告
- [dataPlots] 遍历          #   训练前可视化（相关性热力图等）
- makeSplit()               #   切分训练集/测试集
- applyPreprocessor()       #   标准化（如配置了 standardScaler）
- callTrainer()             #   训练 -> 返回模型
- predict()                 #   预测
- evaluate()                #   评估 + 打印指标
- [resultPlots] 遍历        #   训练后可视化（特征重要性等）
- [diagnostics] 遍历        #   诊断可视化（学习曲线等）
```

### 4.3 baseRunner 关键函数

| 函数 | 作用 |
|---|---|
| `buildRunContext(spec, datasetSpec)` | 加载数据、解析特征/标签、创建输出目录 -> 返回 `RunContext` |
| `runAnalysis(context)` | 按 `TaskType` / `DataKind` 构建并打印数据探索报告 |
| `makeSplit(X, y, splitter, randomState)` | 切分数据——支持 `randomSplit`、`stratifiedSplit`、`None` |
| `applyPreprocessor(splitData, preprocessor)` | 仅在 `standardScaler` 时执行 `fit_transform` / `transform` |
| `callTrainer(trainer, *args, **kwargs)` | 调用训练函数——自动按签名过滤关键字参数 |
| `prepare2dProjection(...)` | 为二维可视化准备 PCA 投影 + 边界模型拟合 |
| `collectScoreOutput(model, XTestProcessed)` | 收集 `predict_proba` 或 `decision_function` 输出 |
| `makeRunResult(model, predictions, scores, metrics)` | 构建 `RunResult` |
| `appendArtifact(result, artifact)` | 向 `RunResult` 追加产物文件路径 |

### 4.4 多模型模式

当 `PipelineSpec.metadata["multiModel"] == True` 时（仅正则化回归）：

1. Runner 将训练返回值视为 `dict[str, 模型]`。
2. 循环 `models.items()`——每个模型独立预测、评估、生成产物。
3. 各模型使用不同的输出子目录（通过 `resolveOutputDir(modelName)`）。

---

## 5. 评估层（`evaluation/`）

**职责**：计算并打印模型评估指标到终端。

| 文件 | 支持指标 |
|---|---|
| `classificationEvaluator.py` | Accuracy、Precision、Recall、F1、ROC-AUC |
| `regressionEvaluator.py` | R^2、MSE、RMSE、MAE、Explained Variance |
| `clusteringEvaluator.py` | Silhouette、Davies-Bouldin、Calinski-Harabasz |
| `dimensionalityEvaluator.py` | 累计解释方差比（PCA） |
| `sequenceEvaluator.py` | HMM 评分 |

---

## 6. 可视化层（`visualization/`）

**职责**：绘制并保存各类图表。分为训练前（`data/`）和训练后（`result/`）两个子模块。

| 子模块 | 包含图表 |
|---|---|
| `data/` | 分类分布、特征空间散点图、相关性热力图 |
| `result/` 分类 | 混淆矩阵、ROC 曲线、特征重要性、决策边界、分类结果展示 |
| `result/` 回归 | 残差图、回归结果展示、学习曲线、特征重要性、树结构 |
| `result/` 聚类 | 聚类结果散点图、KMeans 惯性曲线、DBSCAN k-距离图 / epsilon 扫描 |
| `result/` 降维 | PCA 训练曲线 |
| `result/` 序列 | HMM 状态解码图 |

# CLI 与流水线

## 本章目标

1. 掌握 CLI 四种命令的用法和执行流程。
2. 了解全部 20 条流水线的配置速览。

---

## 1. CLI 入口

**入口文件**：[`main.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/main.py)

### 1.1 命令一览

```bash
# 列出所有可用流水线
python main.py list

# 运行单个流水线
python main.py run <pipelineId>
python main.py run regression.linear_regression
python main.py run classification.svc
python main.py run probabilistic.hmm

# 运行一组流水线（按 domain 前缀筛选）
python main.py suite <groupName>
python main.py suite all              # 全部 20 条
python main.py suite classification   # 分类（6 条）
python main.py suite regression       # 回归（4 条）
python main.py suite ensemble         # 集成（4 条）
python main.py suite clustering       # 聚类（2 条）
python main.py suite dimensionality   # 降维（2 条）
python main.py suite probabilistic    # 概率模型（2 条）

# 仅做数据探索——加载数据并打印统计报告，不训练
python main.py analyze <pipelineId>
```

### 1.2 执行流程

```
main()
- list    -> _printPipelineList()
      - 遍历 PIPELINE_REGISTRY -> 按 ID 排序 -> 逐条打印摘要
- run     -> _runPipeline(id)
      - PIPELINE_REGISTRY.get(id)  -> PipelineSpec
      - DATASET_REGISTRY.get(datasetId) -> DatasetSpec
      - ensureOptionalDependencies()    -> 检查可选依赖
      - executePipeline(spec, ds)       -> 分发到 Runner
- suite   -> _runSuite(group)
      - 按 domain 前缀筛选 pipelineId 列表
      - 逐个 _runPipeline() -> 打印 [OK] / [FAIL] / [SKIP]
- analyze -> _analyzePipeline(id)
               加载数据 -> 构建探索报告 -> 终端打印
```

### 1.3 可选依赖处理

部分流水线依赖可选第三方库：

| 依赖 | 涉及的流水线 |
|---|---|
| `hmmlearn` | `probabilistic.hmm` |
| `xgboost` | `ensemble.xgboost` |
| `lightgbm` | `ensemble.lightgbm` |

当可选依赖未安装时：
- CLI 打印 `[SKIP] <pipelineId>: 缺少可选依赖` 而非崩溃。
- `suite` 模式下其他流水线继续正常执行。

---

## 2. 全部流水线速览

### 2.1 分类（Classification）— 6 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `classification.logistic_regression` | `LogisticRegression(max_iter=1000)` | `standardScaler` | 线性可分二分类 | 决策边界、ROC |
| `classification.decision_tree` | `DecisionTreeClassifier(max_depth=6)` | `None` | blob 多分类 | 决策边界、树结构 |
| `classification.svc` | `SVC(rbf, probability=True)` | `standardScaler` | 同心圆二分类 | 决策边界、ROC |
| `classification.naive_bayes` | `GaussianNB` | `standardScaler` | Iris 真实数据 | 决策边界 |
| `classification.knn` | `KNeighborsClassifier(5)` | `standardScaler` | 双月牙二分类 | 决策边界 |
| `classification.random_forest` | `RandomForestClassifier(100)` | `None` | 高维多分类 | 特征重要性、决策边界 |

### 2.2 回归（Regression）— 4 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `regression.linear_regression` | `LinearRegression` | `None` | 合成线性房价（200 x 3） | 学习曲线、系数对照 |
| `regression.svr` | `SVR(rbf, C=10, $\varepsilon$=0.1)` | `standardScaler` | Friedman1 非线性（200 x 10） | 学习曲线、支持向量数 |
| `regression.decision_tree` | `DecisionTreeRegressor(max_depth=6)` | `None` | California Housing（20640 x 8） | 学习曲线、树结构 |
| `regression.regularization` | `Lasso / Ridge / ElasticNet` | `standardScaler` | diabetes + 共线 + 噪声（442 x 21） | 多模型对比、近零系数 |

### 2.3 聚类（Clustering）— 2 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `clustering.kmeans` | `KMeans` | `standardScaler` | 球形多簇 | K 值扫描（惯性曲线） |
| `clustering.dbscan` | `DBSCAN` | `standardScaler` | 双月牙非线性 | epsilon 扫描、k-距离图 |

### 2.4 降维（Dimensionality）— 2 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `dimensionality.pca` | `PCA` | `standardScaler` | 高维低秩合成 | 累计解释方差训练曲线 |
| `dimensionality.lda` | `LDA` | `standardScaler` | Wine 真实数据（13 维 -> 2 维） | 分类评估（混淆矩阵 + ROC） |

### 2.5 集成学习（Ensemble）— 4 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `ensemble.bagging` | `BaggingClassifier(DT, n=30)` | `standardScaler` | 高噪声双月牙二分类 | 决策边界 |
| `ensemble.gbdt` | `GradientBoostingClassifier` | `standardScaler` | 中等难度多分类 | 特征重要性 |
| `ensemble.xgboost` | `XGBRegressor(n=300, lr=0.05)` | `None` | 回归数据 | 特征重要性 |
| `ensemble.lightgbm` | `LGBMClassifier(n=80)` | `standardScaler` | 高维四分类 | 特征重要性 |

### 2.6 概率模型（Probabilistic）— 2 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `probabilistic.em` | `GaussianMixture` | `standardScaler` | GMM 混合数据 | 分量数扫描（BIC/AIC） |
| `probabilistic.hmm` | `CategoricalHMM` | `None` | 离散序列 | 无图形化输出（仅终端日志） |

---

## 3. 回归流水线对比矩阵

四条回归流水线展示了四种典型的配置组合：

| 配置维度 | 线性回归 | SVR | 决策树回归 | 正则化回归 |
|---|---|---|---|---|
| 标准化 | 无 | **有** | 无 | **有** |
| 学习曲线 | **有** | **有** | **有** | 无 |
| 特征重要性 | **有（coef_）** | **无（RBF 核）** | **有** | **有** |
| 树结构 | 无 | 无 | **有** | 无 |
| 多模型 | 无 | 无 | 无 | **有（3 个）** |
| 训练函数行数 | 3 | 2 | ~5 | ~10 |

# 扩展指南

## 本章目标

1. 掌握新增一个算法流水线的完整步骤。
2. 理解 `PipelineSpec` 各字段的填写规则。
3. 理解项目的关键设计决策及其理由。

---

## 1. 新增算法步骤

新增一个算法需要在 6 个位置添加代码：

### 步骤 1：数据层

在对应的 `*DatasetFactory` 中新增 `load*Dataset()` 方法。

```python
# 示例：src/mlAlgorithms/datasets/tabular/regressionDatasets.py
def loadNewAlgorithmDataset(self) -> DataFrame:
    """加载新算法数据。"""
    rng = np.random.RandomState(self.randomState)
    # ... 生成或加载数据 ...
    return DataFrame({...})
```

### 步骤 2：数据注册

在 [`datasetCatalog.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/src/mlAlgorithms/datasets/datasetCatalog.py) 的 `buildDatasetSpecs()` 中添加 `DatasetSpec`：

```python
DatasetSpec(
    "regression.new_algorithm",       # 与 PipelineSpec.datasetId 一致
    TaskType.REGRESSION,
    DataKind.TABULAR,
    regression.loadNewAlgorithmDataset,
    "price",                           # 标签列名
    None,                              # 自动推断特征列
    "新算法数据描述",
),
```

### 步骤 3：训练层

在对应的 `*Models.py` 中新增训练函数：

```python
def trainNewAlgorithmModel(XTrain, yTrain, randomState: int = 42):
    """训练新算法。"""
    model = SomeModel(param1=..., param2=..., random_state=randomState)
    model.fit(XTrain, yTrain)
    return model
```

### 步骤 4：流水线注册

在 [`pipelines.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/src/mlAlgorithms/catalog/pipelines.py) 的 `PIPELINE_REGISTRY` 注册循环中添加 `PipelineSpec`：

```python
PipelineSpec(
    "regression.new_algorithm",
    TaskType.REGRESSION,
    "regression.new_algorithm",
    RunnerType.REGRESSION,
    trainNewAlgorithmModel,
    "standardScaler",                  # 或 None
    "randomSplit",
    "default",
    "regression",
    "regression",
    ["correlationHeatmap", "featureTargetScatter"],
    ["featureImportance"],
    ["learningCurve"],
    "new_algorithm",
    metadata={
        "learningCurveEstimatorFactory": _buildLearningCurveFactory(
            "regression.new_algorithm"
        )
    },
),
```

### 步骤 5：工厂函数

如需学习曲线，在 `_buildLearningCurveFactory()` 中添加映射：

```python
"regression.new_algorithm": lambda: SomeModel(param1=..., random_state=42),
```

如需二维决策边界可视化（分类模型），在 `_buildVisualModelFactory()` 中添加类似映射。

### 步骤 6：文档

在 `docs/{domain}/{algorithm}/` 下创建 9 个文件：

```
docs/regression/new_algorithm/
├── index.md                       # 概述 + 定位对比 + 文件导航
├── 01-mathematics.md              # 数学原理 + 数学-代码映射
├── 02-data.md                     # 数据构成
├── 03-intuition.md                # 思路与直觉
├── 04-model.md                    # 模型构建
├── 05-training-and-prediction.md  # 训练与预测
├── 06-evaluation.md               # 评估与诊断
├── 07-implementation.md           # 工程实现
└── 08-exercises-and-references.md # 练习与参考文献
```

---

## 2. PipelineSpec 字段填写指南

| 字段 | 填写方式 |
|---|---|
| `id` | `{domain}.{algorithm}`——如 `"regression.svr"` |
| `taskType` | `TaskType` 枚举——决定数据探索报告格式 |
| `datasetId` | 与 `DatasetSpec.id` 完全一致 |
| `runnerType` | `RunnerType` 枚举——通常与 taskType 相同 |
| `trainer` | 训练函数引用——不调用，只传递 |
| `preprocessor` | `"standardScaler"`：RBF 核模型和正则化模型必须；`None`：树模型和线性回归 |
| `splitter` | `"randomSplit"`（回归/聚类）；`"stratifiedSplit"`（分类）；`None`（无需切分，如 HMM） |
| `predictor` | `"default"`（回归/聚类）；`"ldaClassifier"`（LDA）；`"hmmPredictor"`（HMM）；`"gmmPredictor"`（GMM） |
| `evaluator` | `"default"`（回归/分类）；`"transformOnly"`（PCA 等纯变换） |
| `analysisProfile` | 与 evaluator 相同或更具体（如 `"regression"`、`"classification"`） |
| `dataPlots` | 从已有列表选择：`correlationHeatmap` / `featureTargetScatter` / `classDistribution` / `labeledScatter2d` / `featureSpace2d` / `featureSpace3d` / `rawScatter2d` |
| `resultPlots` | 从已有列表选择：`featureImportance` / `confusionMatrix` / `rocCurve` / `classificationResult` / `decisionBoundary` |
| `diagnostics` | 从已有列表选择：`learningCurve` / `treeStructure` / `kmeansSweep` / `dbscanKDistance` / `dbscanEpsSweep` / `gmmComponentSweep` / `pcaTrainingCurve` |
| `outputKey` | 输出子目录名——产物保存到 `outputs/{outputKey}/` |
| `optionalDependencies` | 可选依赖包名——如 `("hmmlearn",)` `("xgboost",)` `("lightgbm",)` |
| `metadata` | `multiModel: True`（多模型）；`learningCurveEstimatorFactory`（学习曲线工厂）；`visualModelFactory`（决策边界工厂） |

---

## 3. 关键设计决策

### 3.1 为什么不用 scikit-learn Pipeline

当前代码**显式编排**每一步，而非使用 `sklearn.pipeline.Pipeline`：

- **教学透明**：每一步的输入输出都是命名清晰的中间变量——`X_train_s`、`y_pred`、`splitData`——方便调试和理解。
- **灵活性**：多模型模式、学习曲线工厂、二维投影等需求难以在标准 Pipeline 中表达。
- **类型边界清晰**：`RunContext` 和 `splitData` 字典提供了明确的状态边界。

### 3.2 为什么使用 Registry 模式

- **声明式配置**：所有流水线在 `pipelines.py` 一个文件中集中声明，一目了然。
- **CLI 解耦**：CLI 通过 `PIPELINE_REGISTRY.get(id)` 获取配置，不依赖 import 路径。
- **零修改扩展**：新增算法只需添加 `PipelineSpec` 条目，无需修改 CLI、Executor 或 Runner。

### 3.3 多模型模式（multiModel）

仅正则化回归使用。训练函数返回 `dict[str, 模型]` 时：

1. Runner 检测 `metadata["multiModel"] == True` -> 进入多模型分支。
2. 循环 `models.items()` -> 每个模型独立预测、评估、生成独立产物文件。
3. 各模型使用不同的输出子目录（`resolveOutputDir(modelName)`）。

### 3.4 学习曲线工厂

学习曲线内部的交叉验证需要**未训练的模型实例**——每次 CV 折必须从头训练。

`_buildLearningCurveFactory()` 返回 `lambda: SomeModel(...)`，确保每次调用都创建全新实例，避免状态污染。

### 3.5 训练函数签名不强制统一

不同模型的构造器参数差异很大（如 `SVR` 不需要 `random_state`，`Lasso` 需要）。`callTrainer()` 通过 `inspect.signature` 动态过滤关键字参数：

```python
def callTrainer(trainer, *args, **kwargs):
    signature = inspect.signature(trainer)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return trainer(*args, **accepted)
```

这样 `randomState` 只传给需要它的训练函数，不会因多余关键字参数而报错。

### 3.6 数据层不负责标准化

数据层只返回原始 DataFrame——标准化由运行器层的 `applyPreprocessor()` 执行。理由：

- 数据层的数据可被多个流水线复用（不同流水线可能需要不同的预处理）。
- 标准化发生在切分**之后**——`fit_transform` 仅用于训练集，`transform` 用于测试集。
- 保持数据工厂的纯粹性——只关心数据来源，不关心数据如何被使用。
