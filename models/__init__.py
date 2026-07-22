from .simple_cnn import SimpleCNN

__all__ = [SimpleCNN]                #这里是负责从上面的模型中，告诉系统，哪些model是可以被调用的，没写的被冻结不允许访问