import os
import torch
import numpy as np                                   #用numpy来生成混淆矩阵confusion_matrix，以及进行acc的加减法等
import matplotlib.pyplot as plt                      #matplotlib工具箱里的pyplot是常用的python绘图工具
import seaborn as sns                                #seaborn是用来美化pyplot的图像
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from sklearn.metrics import confusion_matrix         #从Scikit_learn工具包里，调用confusion_matrix函数
from models import SimpleCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'当前运行设备：{device}')

RESULT_DIR = 'results'                                #dir是directory缩写
WEIGHT_PATH =os.path.join(RESULT_DIR,'best_model.pth')
                                                     #把最佳权重放在WEIGHT_PATH
CM_SAVE_PATH = os.path.join(RESULT_DIR,'confusion_matrix.png')

os.makedirs(RESULT_DIR,exist_ok=True)

#---------------------数据与类型的配置-------------------------------------
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#----------------------评估集的预处理-------------------------------------
test_transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(    mean = [0.4914, 0.4822, 0.4465],
                                           std = [0.2023, 0.1994, 0.2010])
                 ])
test_dataset = datasets.CIFAR10(
    root='./dataset',
    train=False,
    transform=test_transform,
    download=False
)
test_dataloader = DataLoader(
                  test_dataset,
                  batch_size= 64,
                  shuffle = False,
                  num_workers= 0 ,
                  pin_memory = True if torch.cuda.is_available() else False
                  )

#--------------加载模型权重----------------------------------------
if not os.path.exists(WEIGHT_PATH):
    raise FileNotFoundError(f'未找到权重{WEIGHT_PATH}\n请先进行训练后再进行评估')

model = SimpleCNN(num_classes = 10).to(device)
model.load_state_dict(torch.load(WEIGHT_PATH,map_location=device,weights_only = False))
model.eval()                                 #开启评估模式
print(f'成功加载权重：{WEIGHT_PATH}')

all_labels = []                              #把每个batch的真值labels按顺序放在这里
all_preds  = []                              #把predict的结果放在这里面
total_num = 0
correct_num = 0                              #任何自定义变量必须先设置初值，才能在后面进行调用

with torch.no_grad():                        #禁用梯度计算，不需要反向传播
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, dim = 1)

        total_num += labels.size(0)
        correct_num += (predicted == labels).sum().item()


        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
test_accuracy = correct_num/total_num
print(f'\n整体测试集的准确率为{test_accuracy:.4f}({correct_num}/{total_num}')

cm = confusion_matrix(all_labels, all_preds)
                                            #先把混淆矩阵的横坐标画为真值，纵坐标画为preds值
plt.figure(figsize = (10,8))                #创建一个Matplot画布，宽10，长8

sns.heatmap(
          cm,                               #混淆矩阵
          annot=True,                       #显示格子里的数值
          fmt='d',                          #数据类型为十进制整数
          cmap = 'Blues',                   #配色为蓝色
          xticklabels = CLASS_NAMES,
          yticklabels = CLASS_NAMES
)

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix (CIFAR-10)", fontsize=14, pad=15)
plt.xticks(rotation=45, ha="right")         # x轴标签旋转，避免文字重叠
plt.yticks(rotation=0)
plt.tight_layout()                          # 自动调整布局，防止标签被截断

plt.savefig(CM_SAVE_PATH, dpi=300, bbox_inches="tight")
plt.close()                                 # 关闭画布，释放内存

print(f"混淆矩阵已保存至: {CM_SAVE_PATH}")
print("评估完成。")