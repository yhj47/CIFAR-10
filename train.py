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
from models import SimpleCNN,ResNet18,ResNet34,ResNet50,ResNet101



#-------------实验运行参数设置----------------------------------------
parser = argparse.ArgumentParser(description='CIFAR10_Training')
parser.add_argument('--exp_name', type=str, default='ResNet34_run1', help='实验标识名:模型名_run1')
parser.add_argument('--model', type=str, default='ResNet34', help='模型名称')
args = parser.parse_args()
#------------------------------------以上的都没看懂在干啥-----------------------------------

#超参数配置
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(f'训练使用设备：{device}')                  # f-string这里固定语法，后有{}要进行计算替代时候必须用f

args = parser.parse_args()

# ========== 新增：自动递增实验序号 ==========
# 如果用户没手动传 exp_name，就自动生成 模型名_runN 的格式，自动找下一个序号
if args.exp_name == parser.get_default('exp_name'):
    base_name = args.model  # 以模型名为基础名
    run_id = 1
    # 循环检测 checkpoints 文件夹里是否已有对应序号的文件
    while True:
        candidate_ckpt = os.path.join('checkpoints', f'{base_name}_run{run_id}_latest_checkpoint.pth')
        if not os.path.exists(candidate_ckpt):
            break
        run_id += 1
    exp_name = f'{base_name}_run{run_id}'
    print(f'未指定实验名，自动生成：{exp_name}')
else:
    exp_name = args.exp_name

batch_size = 64
learning_rate = 1e-3                           #初始学习率提前设定，训练后期衰减学习率提高精度
epochs = 10                                    #可修改参数
patience = 3                                   #防止过拟合，可以实现early stop
lr_factor = 0.5                                #衰减因子，当触发降学习率时，将LR*衰减因子
lr_patience = 3                                #耐心值，当3个循环验证集的精度没有提升时，触发学习率下降


#-----------创建所需文件夹--------------
os.makedirs('runs', exist_ok = True)#存放TensorBoard训练日志，包含每轮的损失函数、学习率等
os.makedirs('results', exist_ok = True)#存放最佳模型权重以及CM混淆矩阵
os.makedirs('checkpoints', exist_ok = True)#存放训练的checkpoints

writer = SummaryWriter(log_dir=os.path.join('runs',exp_name))       #调用这个SummaryWriter工具来记录训练日志


Best_Weight_Path = os.path.join('results',f'{exp_name}_best_model.pth')
                                               #把最佳权重参数记录在results里面，文件名为“best_model.pth”
Latest_Weight_Path = os.path.join('checkpoints',f'{exp_name}_latest_checkpoint.pth')
                                               #把最近一次训练权重记录在checkpoints里面，文件名为"latest_checkpoint.pth"
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

model_dic = {'SimpleCNN':SimpleCNN,'ResNet18':ResNet18,'ResNet34':ResNet34,'ResNet50':ResNet50,'ResNet101':ResNet101}            #这是models的字典，把新家进来的网络写在里面

model_class = model_dic[args.model]
model = model_class(num_classes=10).to(device)


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
    for images,lables in train_loader:         #遍历数据集，按照loader的批次，一个batch64张照片，lables是0~9
        images = images.to(device)
        lables = lables.to(device)             #输送到GUP上计算
        optimizer.zero_grad()                  #清空上一轮训练计算的梯度
        outputs = model(images)
        loss = criterion(outputs,lables)       #平均损失
        loss.backward()                        #反向传播
        optimizer.step()                       #更新参数：优化器Adam参照梯度以及设置的学习率，更新网络的values以及bias
        train_loss_sum += loss.item()          #累加损失：train_loss_sum=train_loss_sum+loss,且终值被item设定为浮点数
    train_loss = train_loss_sum / len(train_loader)
                                               #一个epoch结束了，计算所有batches的平均loss，len就是batches的个数

    #---------------------验证阶段-----------------------
    model.eval()                               #模型开启evaluation模式，意味着dropout停止工作
    val_loss_sum = 0.0
    correct = 0                                #eval阶段识别正确的个数
    total = 0
    with torch.no_grad():                      #关闭梯度计算，停止value和bias的更新
        for images,lables in val_loader:
            images = images.to(device)
            lables = lables.to(device)
            outputs = model(images)
            loss = criterion(outputs,lables)
            val_loss_sum += loss.item()
            _, pred = torch.max(outputs, dim = 1)
                                               #torch.max是在一维上寻找最大值，将最大值本身索引会 = 前面
                                               #_把最大值本身给忽略了，把这个索引赋给pred，是类别的0~9
            total += lables.size(0)
            correct += (pred == lables).sum().item()
    val_loss = val_loss_sum / len(val_loader)
    val_acc = correct / total

    scheduler.step(val_loss)

    #--------------TensorBoard记录-------------------------
    writer.add_scalar('Loss/Train', train_loss, epoch)
                                              #训练集平均损失
    writer.add_scalar('Loss/Val', val_loss, epoch)
                                              #验证集平均损失
    writer.add_scalar('Accuracy/Val', val_acc, epoch)
                                              #验证集准确率
    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch)
                                              #当前学习率
    print(f"Epoch[{epoch+1:02d}/{epochs}]"
          f"TrainLoss:{train_loss:.4f} | ValLoss:{val_loss:.4f} | ValAcc:{val_acc:.4f} "
          f"LR:{optimizer.param_groups[0]['lr']:.6f}")
    #--------------------checkpoints保存---------------------------------------
    checkpoint = {
        "epoch": epoch ,
        "model_state_dict": model.state_dict(),          #保存此时的values和bias
        "optimizer_state_dict": optimizer.state_dict(),  #保存此时优化器参数
        "best_acc": best_acc
    }
    torch.save(checkpoint, Latest_Weight_Path)

    #------------------出现更高精度，保存最优权重------------------------------
    if val_acc > best_acc:
        best_acc = val_acc
        early_stop_counter = 0
        torch.save(model.state_dict(), Best_Weight_Path)
        print(f" New Best! Save best model. Best Acc: {best_acc:.4f}")
                                                          # best_acc:.4f
                                                          #.4f表示这里数据类型为浮点数4位小数
    else:
        early_stop_counter += 1
        print(f"️ No improvement. EarlyStop counter: {early_stop_counter}/{patience}")

        # ========== Early Stopping 触发 ==========
        if early_stop_counter >= patience:
            print(f"\n Early Stopping Triggered! No improvement over {patience} epochs.")
            break

writer.close()
print(f"\nTraining Finished. Best Val Accuracy = {best_acc:.4f}")
print(f"Best Model Path: {Best_Weight_Path}")
print(f"Latest Checkpoint Path: {Latest_Weight_Path}")