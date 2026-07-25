from .simple_cnn import SimpleCNN
from .resnet import ResNet18,ResNet34,ResNet50,ResNet101

__all__ = ['SimpleCNN','ResNet18','ResNet34','ResNet50','ResNet101']                #这里是负责从上面的模型中，告诉系统，哪些model是可以被调用的，没写的被冻结不允许访问