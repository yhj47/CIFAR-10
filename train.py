import argparse
import os                                      #os能够对项目文件夹进行创建、储存等操作
from calendar import EPOCH

import torch
import torch.nn as nn
import torch.optim as optim                    #调用torch的优化器
from torch.utils.data import DataLoader        #DataLoader负责把数据打包传送给神经网络
from torchvision import datasets,transforms    #dataset负责去找项目里有没有dataset，如果没有的话，将联网下载
                                               #transforms是负责给照片进行旋转、裁剪等操作的工具包
from torch.utils.tensorboard import SummaryWriter
from models import SimpleCNN


#超参数配置
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(f'训练使用设备：{device}')                  # f-string这里固定语法，后有{}要进行计算替代时候必须用f

batch_size = 64
learning_rate = 1e-3                           #学习率提前设定？
epochs = 15
patience = 3                                   #防止过拟合，可以实现early stop
lr_factor = 0.5                                #衰减因子，当触发降学习率时，将LR*衰减因子
lr_patience = 3                                #耐心值，当3个循环验证集的精度没有提升时，触发学习率下降


#-----------创建所需文件夹--------------
os.makedirs('runs', exist_ok = True)
os.makedirs('results', exist_ok = True)

writer = SummaryWriter(log_dir='./runs')       #调用这个SuammaryWriter工具来记录训练日志
Best_Weight_Path = os.path.join('runs','best_model.pth')
                                               #把最佳权重参数记录在runs里面，文件名为“best_model.pth”
Latest_Weight_Path = os.path.join('runs','latest_checkpoint.pth')
                                               #把最近一次训练权重记录在runs里面，文件名为"latest_checkpoint.pth"
                                               #防止训练丢失，否则需要从头训练
#--------------数据预处理----------------------------------------------------------------------
#=====================transform数据增强========================================================
train_transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),  #先把照片四周补上4圈0，然后随机裁剪出一张32*32大小
    transforms.RandomHorizontalFlip(),         #把照片随机水平翻转
    transforms.ToTensor(),                     #把PIL照片转化为tensor格式
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=train_transform)
                                               #去项目文件夹里找CIFAR10数据集
val_set   = datasets.CIFAR10(root='./dataset', train=False,download=True, transform=val_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
                                               #用DataLoader加载训练数据集和测试数据集，一捆的数目已经确定好
                                               #shuffle指是否打乱顺序
                                               #num_worker指的是是否需要在主线开几个支线来提前加载照片，用于照片尺寸过大
#--------------------------模型、损失、学习率、优化的调度--------------------------------------------------------------
model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()              #nn.CrossEntropyLoss是Pytorch.nn内置的一个损失函数器，专门用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min', factor=lr_factor, patience=lr_patience)
                                               #最后一个是一个监督者，具体没看懂
#------------------------训练参数-----------------------------------------------------------------------------------
best_acc = 0.0                                 #最佳准确率记录器
early_stop_counter = 0                         #早停计数器


#-----------------------训练循环-----------------------

for epoch in range(epochs):
    model.train()                              #训练模型
    train_loss_sum = 0                         #将训练集的总损失初始化为0
    for images,lables in train_loader: