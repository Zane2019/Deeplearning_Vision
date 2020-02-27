#! usr/bin/ python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

Class Net (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)

    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),2)
        x=F.max_pool2d(F.relu(self.conv2(x),2))
        return x

#网络定义好了之后，就涉及传入参数，算误差，反向传播，更新权重。
net=Net()

output=net(input)



compute_loss=nn.MSELoss()  #定义损失函数方法
#如果想用自己的损失函数，必须也把loss在forward（）里算，最后返回return loss.

loss=compute_loss(true_value,output)


#更新权重：
#1.先算loss对于输入x的偏导，当然网络好几层，这个x指的是每一层的输入，而不是最开始的输入input
# 2.对【1】的结果再乘以一个步长（这样就相当于得到一个对参数w的修改量）
# 3.用w减掉这个修改量，完成一次对参数w的修改。
loss.backward()  #这一步得到对参数W一步更新量，算是一次反向传播。

#第二步，第三步怎么实现，通过优化器来实现，让优化器来自动实现对网络权重w更新

from torch import optim

optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
#每次迭代前，把optimizer里存的梯度清零一下（因为w已经更新过的更新量下一次就不需要用了

#所以基本顺序就是，
# #先定义网络：写网络Net的class，声明网络的实例net=Net()
# #定义优化器optimizer=optim.xx(net.parameters(),lr=0.1)
# #定义损失函数（自己写或者直接用官方的，compute_loss=nn.MSELoss()或者其他
# # 定义完之后，开始一次一次的循环：
# # # 1.先清空优化器里的梯度信息，optimizer.zero_grad()
# # # 2.将input传入，out=net(input),正向传播
# # # 3.算损失，loss=compute_loss(target,output) ##这里target就是参考标准值，需要自己准备，和之前传入的input一一对应
# # # 4.误差反向传播 loss.backward()
# # # 5.更新参数optimizer.step()
for t in range(100):
    prediction=net(x)
    loss=compute_loss(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
C=optim.SGD()
