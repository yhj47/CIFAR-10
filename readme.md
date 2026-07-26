# CIFAR-10 Image Classification
## 一、项目整体目录结构
### CIFAR-10
    |----dataset
    |----models
    |    |---__init__.py
    |    |---simplecnn.py
    |    |---resnet.py
    |----runs
    |----results
    |----checkpoints
    |----.gitignore
    |----readme.md
    |----train.py
    |----evaluate.py
## 二、SimpleCNN
### 卷积块 1：
    Conv2d (3,16)+BN+ReLU+MaxPool
### 卷积块 2：
    Conv2d (16,32)+BN+ReLU+MaxPool
### 卷积块 3：
    Conv2d (32,64)+BN+ReLU+MaxPool
### 展平 Flatten
### 全连接层 + Dropout (0.5) 防过拟合
### 最终全连接输出 10 个类别得分

# SimpleCNN 混淆矩阵
### 测试集混淆矩阵
  <div align ="center">
  <img src ="results/SimpleCNN_run1_confusion_matrix.png" width="700">
  </div>

##   三、ResNet18
### 3.1 网络架构
ResNet18 基础模块为 `BasicBlock`，两层 3×3 卷积

1. 网络层级结构
- 输入头：3×3卷积 + BN，将3通道原图映射至64通道特征；
- 4组残差层：layer1(64通道)、layer2(128通道)、layer3(256通道)、layer4(512通道)，每组堆叠2个BasicBlock；
- 下采样：layer2~layer4 通过 stride=2 完成尺寸减半；
- 维度匹配：通道/尺寸不匹配时，shortcut 使用 1×1卷积+BN 校正维度；
- 输出头：自适应全局平均池化 + 线性层，输出10分类结果。
### 混淆矩阵
  <div align ="center">
  <img src ="results/ResNet18_run1_confusion_matrix.png" width="700">
  </div>

## 四、ResNet34
### 网络架构ResNet34
ResNet34使用 `BasicBlock` 基础残差块，无Bottleneck瓶颈结构
1. 网络层级结构
- 输入头与ResNet18完全一致：3×3卷积+BN，输出64通道特征；
- 4组残差层堆叠数量调整为 `[3, 4, 6, 3]`；
- layer1：3个64通道BasicBlock；layer2：4个128通道BasicBlock；layer3：6个256通道BasicBlock；layer4：3个512通道BasicBlock；
- 下采样、shortcut维度校正、池化与分类头逻辑和ResNet18保持统一；
### ResNet34 混淆矩阵
   <div align ="center">
   <img src ="results/ResNet34_run1_confusion_matrix.png" width="700">
   </div>
