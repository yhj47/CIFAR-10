import torch
import torch.nn as nn

class SimpleCNN(nn.Module):                       #自定义函数名字，习惯首字母大写
      def __init__(self,num_classes=10):               #定义网络的输出通道有几个，num_classes仅表示在这里的分类任务，其实可以自定义名字
          super().__init__()                        #super是我们模型与nn.Module沟通的中间人。这一句是在告知torch我现在要开始正式搭建网络了

          self.feature = nn.Sequential(                 #这里是在说，self这条pipeline的最开端是feature部分，这个名字可以自定义
                                                    #nn.Sequential是在说外部输入进来之后，在后面的layers按照固定顺序自动走下去

          #Conv1
           nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,bias=False),
                                                    #这是第一个layer，Conv2d表示kernel走的方向是2维度的，左右与上下
                                                    #in/out_channels是固定用法，kernel_size也是固定用法，表示这个window是3*3
                                                    #核心：输出通道数=（输入通道数-kernel_size+2*padding）/stride+1
           nn.BatchNorm2d(16),
           nn.ReLU(inplace = True),                      #激活函数，inplace指用ReLU后的值直接替换激活前，节省内存
           nn.MaxPool2d(kernel_size=2, stride=2),        #池化层，保证通道数不变，压缩画质，只保留输入图中的峰值部分，弱化噪声
          #Conv2
           nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,bias=False),
           nn.BatchNorm2d(32),
           nn.ReLU(inplace = True),
           nn.MaxPool2d(kernel_size=2,stride=2),
          #Conv3
           nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,bias = False),
           nn.BatchNorm2d(64),
           nn.ReLU(inplace = True),
           nn.MaxPool2d(kernel_size=2,stride=2),
          )                                             #现在feature提取层定义完毕

          self.classifier = nn.Sequential(
           nn.Linear(64*4*4,512),
           nn.ReLU(inplace = True),
           nn.Dropout(p=0.5),                           #使上一个Linear层有p个神经元随即失活，来防止过拟合，dropout仅在训练时开启，测试阶段失效
           nn.Linear(512,num_classes),
          )

      def forward(self,x):
          x = self.feature(x)
          x = torch.flatten(x,start_dim=1)          #flatten层把 x(batch_size,3,32,32)flat成一维的tensor（batch_size,3*32*32）
          x = self.classifier(x)
          return x