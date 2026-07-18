# CIFAR-10 Image Classification

基于 PyTorch 实现的 CIFAR-10 图像分类项目，包含 SimpleCNN 和 ResNet-18 两种网络架构，支持 TensorBoard 可视化训练过程和完整的评估指标。

## 项目结构

```
CIFAR-10/
├── models/                    # 模型定义目录
│   ├── __init__.py            # 模型模块导出文件，将 SimpleCNN 和 ResNet18 暴露给外部调用
│   ├── simple_cnn.py          # 简单卷积神经网络模型定义
│   └── resnet18.py            # ResNet-18 残差网络模型定义
├── dataset/                   # CIFAR-10 数据集目录（不上传到 GitHub）
│   └── cifar-10-batches-py/   # CIFAR-10 官方数据集文件
├── my_test/                   # 自定义测试图片目录（不上传到 GitHub）
├── checkpoints/               # 模型检查点保存目录（不上传到 GitHub）
├── runs/                      # TensorBoard 日志目录（不上传到 GitHub）
├── results/                   # 评估结果保存目录（不上传到 GitHub）
├── train.py                   # 训练脚本：支持训练 SimpleCNN 和 ResNet-18
├── evaluate.py                # 评估脚本：计算准确率、精确率、召回率、F1、混淆矩阵
├── predict_my_test.py         # 自定义图片预测脚本：对 my_test/ 目录下的图片进行分类预测
├── .gitignore                 # Git 忽略文件配置
└── README.md                  # 项目说明文档
```

## 文件详细说明

### 模型目录 (models/)

#### `models/__init__.py`
- **作用**：作为模型模块的入口文件，导出 `SimpleCNN` 和 `ResNet18` 类
- **内容**：使用 `from .simple_cnn import SimpleCNN` 和 `from .resnet18 import ResNet18` 将两个模型类导入，并通过 `__all__` 列表暴露接口，方便外部通过 `from models import SimpleCNN, ResNet18` 直接导入

#### `models/simple_cnn.py`
- **作用**：定义简单的卷积神经网络 `SimpleCNN` 类
- **结构**：
  - `conv1`: 3→32 通道的卷积层，3×3 卷积核，padding=1
  - `bn1`: Batch Normalization 层，加速训练收敛
  - `conv2`: 32→64 通道的卷积层，3×3 卷积核，padding=1
  - `bn2`: Batch Normalization 层
  - `conv3`: 64→128 通道的卷积层，3×3 卷积核，padding=1
  - `bn3`: Batch Normalization 层
  - `pool`: MaxPool2d 池化层，2×2 池化核，步长为 2
  - `fc1`: 全连接层，128×4×4 → 512
  - `fc2`: 全连接层，512 → 10（分类输出）
  - `dropout`: Dropout 层，随机丢弃 50% 神经元，防止过拟合
- **前向传播**：卷积 → BN → ReLU → 池化，重复三次后接入全连接层

#### `models/resnet18.py`
- **作用**：定义 ResNet-18 残差网络 `ResNet18` 类
- **结构**：
  - `BasicBlock`: 基础残差块，包含两个 3×3 卷积层和 shortcut 连接
  - `conv1`: 输入层，3→64 通道的卷积层
  - `bn1`: Batch Normalization 层
  - `layer1-layer4`: 四个残差层，通道数依次为 64、128、256、512
  - `avgpool`: 全局平均池化层，将特征图降为 1×1
  - `fc`: 全连接层，512 → 10（分类输出）
- **特点**：通过残差连接解决深度网络训练时的梯度消失问题

### 训练脚本 (train.py)

- **作用**：统一的模型训练入口，支持训练 SimpleCNN 和 ResNet-18
- **功能**：
  - 命令行参数解析（模型类型、训练轮数、批次大小、学习率等）
  - 数据加载与预处理（含数据增强：随机水平翻转、随机裁剪）
  - 模型初始化与设备分配（自动检测 GPU/CPU）
  - 损失函数定义（交叉熵损失）
  - 优化器配置（SGD + Momentum）
  - 学习率调度（Cosine Annealing）
  - 训练循环（前向传播 → 计算损失 → 反向传播 → 参数更新）
  - 每轮训练后在测试集上评估准确率
  - 保存最佳模型和最终模型
  - TensorBoard 日志记录（训练/测试损失、准确率、学习率）

### 评估脚本 (evaluate.py)

- **作用**：对训练好的模型进行全面评估，计算多种评估指标
- **功能**：
  - 加载指定模型和权重文件
  - 在测试集上进行预测，收集真实标签和预测结果
  - 计算评估指标：
    - 准确率（Accuracy）
    - 精确率（Precision）：Macro 和 Weighted
    - 召回率（Recall）：Macro 和 Weighted
    - F1 分数（F1-Score）：Macro 和 Weighted
    - 混淆矩阵（Confusion Matrix）
  - 打印评估结果到控制台
  - 生成混淆矩阵热力图（PNG 格式）
  - 生成评估指标报告（TXT 格式）

### 预测脚本 (predict_my_test.py)

- **作用**：对自定义图片进行分类预测
- **功能**：
  - 加载指定模型和权重文件
  - 遍历 `my_test/` 目录下的所有图片文件
  - 对每张图片进行预处理（Resize 到 32×32、归一化）
  - 执行预测，输出：
    - 预测类别和置信度
    - Top-3 预测结果及对应的置信度
- **使用场景**：测试自己收集的图片，验证模型的实际效果

### 配置文件

#### `.gitignore`
- **作用**：配置 Git 版本控制时忽略的文件和目录
- **忽略内容**：
  - `dataset/`: 数据集目录（文件较大，且可从官方下载）
  - `my_test/`: 自定义测试图片（用户本地文件）
  - `checkpoints/`: 模型检查点（文件较大）
  - `runs/`: TensorBoard 日志（自动生成）
  - `results/`: 评估结果（自动生成）
  - Python 编译文件（`__pycache__/`、`*.pyc`）
  - IDE 配置文件（`.idea/`、`.vscode/`）
  - 操作系统文件（`Thumbs.db`、`.DS_Store`）

## 环境依赖

- Python 3.8+
- PyTorch 1.9+
- torchvision
- tensorboard
- scikit-learn
- seaborn
- matplotlib
- PIL (Pillow)
- numpy

安装依赖：
```bash
pip install torch torchvision tensorboard scikit-learn seaborn matplotlib pillow numpy
```

## 数据集

CIFAR-10 数据集包含 60,000 张 32x32 彩色图片，分为 10 个类别，每类 6,000 张图片。
- 训练集：50,000 张
- 测试集：10,000 张

10 个类别：`plane`（飞机）、`car`（汽车）、`bird`（鸟）、`cat`（猫）、`deer`（鹿）、`dog`（狗）、`frog`（青蛙）、`horse`（马）、`ship`（船）、`truck`（卡车）

数据集下载后放入 `dataset/` 目录下，结构为 `dataset/cifar-10-batches-py/`。

## 模型介绍

### SimpleCNN
简单的三层卷积神经网络，包含 BatchNorm 和 Dropout：
- 3 层卷积层（32 → 64 → 128 通道）
- Batch Normalization
- Max Pooling
- 2 层全连接层
- Dropout (0.5)

### ResNet-18
经典的 ResNet-18 残差网络，适用于 CIFAR-10 的版本：
- 4 个残差层（64 → 128 → 256 → 512 通道）
- BasicBlock 残差块
- 全局平均池化
- 全连接分类头

## 使用方法

### 1. 训练模型

训练 SimpleCNN：
```bash
python train.py --model simple_cnn --epochs 20 --batch_size 128 --lr 0.01
```

训练 ResNet-18：
```bash
python train.py --model resnet18 --epochs 30 --batch_size 128 --lr 0.01
```

训练参数说明：
- `--model`: 模型架构，`simple_cnn` 或 `resnet18`
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 初始学习率
- `--momentum`: SGD 动量
- `--weight_decay`: 权重衰减
- `--data_root`: 数据集路径
- `--save_dir`: 模型保存路径
- `--log_dir`: TensorBoard 日志路径

### 2. TensorBoard 可视化

启动 TensorBoard：
```bash
tensorboard --logdir=runs
```

在浏览器中打开 `http://localhost:6006` 查看训练曲线：
- Loss/Train: 训练损失
- Loss/Test: 测试损失
- Accuracy/Train: 训练准确率
- Accuracy/Test: 测试准确率
- Learning_Rate: 学习率变化

### 3. 模型评估

评估 SimpleCNN：
```bash
python evaluate.py --model simple_cnn --checkpoint checkpoints/simple_cnn_best.pth
```

评估 ResNet-18：
```bash
python evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_best.pth
```

评估结果会保存在 `results/` 目录下，包括：
- 指标文本报告
- 混淆矩阵热力图

### 4. 自定义图片预测

预测 `my_test/` 目录下的图片：
```bash
python predict_my_test.py --model simple_cnn
```

或使用 ResNet-18：
```bash
python predict_my_test.py --model resnet18
```

## 评估指标定义

本项目使用以下评估指标来衡量模型性能：

### 1. 准确率 (Accuracy)

**定义**：预测正确的样本数占总样本数的比例。

**公式**：
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**说明**：
- 衡量模型整体预测正确的比例
- 适合类别分布均衡的数据集
- CIFAR-10 各类别数量均衡，准确率是重要指标

### 2. 精确率 (Precision)

**定义**：在所有被预测为正类的样本中，真正为正类的比例。

**公式**：
```
Precision = TP / (TP + FP)
```

**说明**：
- 衡量模型预测为正类的样本中，有多少是真正的正类
- 关注"预测的准确性"，即模型说"是"的时候有多大把握
- Macro 精确率：各类别精确率的算术平均
- Weighted 精确率：按各类别样本数加权平均

### 3. 召回率 (Recall / Sensitivity)

**定义**：在所有真实为正类的样本中，被模型正确预测为正类的比例。

**公式**：
```
Recall = TP / (TP + FN)
```

**说明**：
- 衡量模型能找出多少正类样本
- 关注"找得全不全"
- Macro 召回率：各类别召回率的算术平均
- Weighted 召回率：按各类别样本数加权平均

### 4. F1 分数 (F1-Score)

**定义**：精确率和召回率的调和平均数。

**公式**：
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**说明**：
- 综合考虑精确率和召回率
- 取值范围 [0, 1]，越大越好
- 当精确率和召回率都高时，F1 分数才会高
- Macro F1：各类别 F1 的算术平均
- Weighted F1：按各类别样本数加权平均

### 5. 混淆矩阵 (Confusion Matrix)

**定义**：一个 N×N 的矩阵，N 为类别数，展示模型预测结果与真实标签的对比。

**说明**：
- 行表示真实类别，列表示预测类别
- 对角线上的元素表示预测正确的样本数
- 非对角线元素表示被错误分类的样本数
- 可以直观看到模型容易混淆哪些类别

### 指标符号说明

| 符号 | 含义 |
|------|------|
| TP (True Positive) | 真正例：真实为正类，预测为正类 |
| TN (True Negative) | 真负例：真实为负类，预测为负类 |
| FP (False Positive) | 假正例：真实为负类，预测为正类 |
| FN (False Negative) | 假负例：真实为正类，预测为负类 |

## 训练技巧

- **数据增强**：随机水平翻转、随机裁剪
- **优化器**：SGD + Momentum
- **学习率调度**：Cosine Annealing
- **权重衰减**：L2 正则化
- **Batch Normalization**：加速训练、稳定分布

## 参考文献

- Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
