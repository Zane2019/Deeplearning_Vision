import torch.nn as nn  # 各种层类型的实现
import torch.nn.functional as F　  # 各种层函数的实现，与层类型对应，如卷积函数，池化函数，归一化函数等等
import torch.optim as optim  # 实现各种优化算法的包

import argparse

from torchvision import datasets, transforms

######################################################
########### 1.构建网络并实例化        ##################
###########　　　　之自定义Net并实例化 ###################
######################################################

# 自定义模型类需要继承nn.Module,且你至少需要写__init__和forward两个函数


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 初始化层类型
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # 定义前向传播
        x = F.relu(F.max_pool2d(self.conv1(x), 2)
        x=F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x=x.view(-1, 320)  # 类似于np.reshape
        x=F.relu(self.fc1(x))
        x=F.dropout(x, training=self.training)
        x=self.fc2(x)
        return F.log_softmax(x, dim=1)

# 实例化自定义模型

model=Net().to(device)

######################################################
########### 1.构建网络并实例化        ##################
###########　　　　之基于顺序容器构建 ###################
######################################################

# example of using Sequential
model=nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)

# example of using Sequential with OrderedDict

model=nn.Sequential(
    OrderedDict(
        [('conv1', nn.Conv2d(1, 20, 5)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(20, 64, 5)),
        ('relu2', nn.ReLU())
        ]
    )
)

#######################################################
################## ２ 定义训练函数 ######################
#######################################################

def train(args, model, device, train_loader, optimizer, epoch):  # 还可以添加loss_func参数
    model.train()  # 必备!!!!将模型设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):  # 从数据加载器中迭代一个batch数据
        data, target=data.to(device), target.to(device)  # 将数据存储cpu或者GPU
        optimizer.zero_grad()  # 清楚所有优化梯度
        output=model(data)  # 喂入数据，前向传播输出
        loss=F.nll_loss(output, target)  # 调用损失函数计算损失
        loss.backward()  # 反向传播
        optimizer.step() 　  # 更新参数
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss:{:.6f}'.format(\
                epoch, batch_idx*len(data), len(train_loader.dataset),\
                    100.*batch_idx/len(train_loader), loss.item()))


#######################################################
################## 3 定义测试函数 ######################
#######################################################
def test(args, model, device, test_loader):
    model.eval()  # 　必备！！！！！将模型设置为评估模式
    test_loss=0
    correct=0
    with torch.no_grad():  # 禁用梯度计算
        data, target=data.to(device), target.to(device)
        output=model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred=output.max(1, keepdim=True)[1]  # 得到最大log概率的索引
        correct += pred.eq(target.view_as(pred)).sum().item()　  # 统计预测正确的个数

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(\
    test_loss, correct, len(test_loader.dataset),\
    100. * correct / len(test_loader.dataset)))

#######################################################
################## 4 训练和测试 ########################
#######################################################

def main():
    # Training setting
    parser=argparse.ArgumentParser(description="Pytorch MNIST Example")

    parser.add_argument('--batch-size', type=int, default=64,
                        metavar='N', help='input batch size for trainning(default:64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N', help='input batch size for testing (default:1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        metavar='N', help='number of epochs to train(default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status(default: 10)')
    args=parser.parse_args()

    use_cuda=not args.no_cuda and torch.cuda.is_avaiable() #根据输入参数和实际cuda的有无决定是否使用GPU

    torch.manual_seed(args.seed)　#设置随机种子，保证可重复性

    device=torch.device('cuda' if use_cuda else 'cpu')
    # 设置数据加载的子进程数；是否返回之前将张量复制到cuda的页锁定内存
    kwargs={'num_works':1,'pin_memory':True} if use_cuda else {}

    train_loader=torch.utils.data.DataLoader(
       datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs
    )

    model = Net().to(device) # 实例化网络模型
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) # 实例化求解器
	
	for epoch in range(1, args.epochs + 1): # 循环调用train() and test() 进行epoch迭代
	    train(args, model, device, train_loader, optimizer, epoch)
	    test(args, model, device, test_loader)