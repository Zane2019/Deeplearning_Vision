
# Pytorch

`torch.Tensor`是一种包含单一数据类型元素的多维矩阵。torch定义了其中CPU tensor类型和八种 GPU tensor类型

数据类型| CPU tensor|GPU tensor
--|:--:|--:
32位浮点|torch.FloatTensor|torch.cuda.FloatTensor
64位浮点|torch.DoubleTensor|torch.cuda.DoubleTensor
16位浮点|N/A|torch.cuda.HalfTensor
8位整型（无符号）|torch.ByteTensor|torch.cuda.ByteTensor
8位整型|torch.CharTensor|torch.cuda.CharTensor
16位整型|torch.ShortTensor|torch.cuda.ShortTensor
32位整型|torch.IntTensor|torch.cuda.IntTensor
64位整型|torch.LongTensor|torch.cuda.LongTensor

`torch.Tensor`是默认的tensor类型`torch.FloatTensor`的简称。


`requireds_grad`、`volatile`、`no_grad`、`detach()`介绍

1. requireds_grad
   Variable变量的requires_grad的属性默认为False,若一个节点requires_grad被设置为True，那么所有依赖它的节点的requires_grad都为True
2. volatile=True是Variable的另一个重要的标识，它能够将所有依赖它的节点全部设为volatile=True，**其优先级比requires_grad=True高**。因而volatile=True的节点不会求导，即使requires_grad=True，也不会进行反向传播，**对于不需要反向传播的情景(inference，测试推断)，该参数可以实现一定速度的提升，并节省一半的显存**，因为其不需要保存梯度。
该属性已经在0.4版本中被移除了，并提示你可以使用with torch.no_grad()代替该功能
3. `detach`，如果x为中间变量，`x'=x.detach`表示创建一个与x相同，但`requires_grad==False`的variable((实际上是把x’ 以前的计算图 grad_fn 都消除了)，x’ 也就成了叶节点。原先反向传播时，回传到x时还会继续，而现在回到x’处后，就结束了，不继续回传求到了。另外值得注意, x (variable类型) 和 x’ (variable类型)都指向同一个Tensor ,即 x.data
而detach_() 表示不创建新变量，而是直接修改 x 本身。
4. `retain_graph` ，每次 `backward()` 时，默认会把整个计算图free掉。一般情况下是每次迭代，只需一次 `forward()` 和一次 `backward()` ,前向运算`forward()` 和反向传播`backward()`是成对存在的，一般一次`backward()`也是够用的。但是不排除，由于自定义loss等的复杂性，需要一次`forward()`，多个不同loss的`backward()`来累积同一个网络的grad,来更新参数。于是，若在当前`backward()`后，不执行`forward()` 而可以执行另一个`backward()`，需要在当前`backward()`时，指定保留计算图，即`backward(retain_graph)`



**！注意：** 会改变tensor的函数操作会用一个下划线后缀来标示。比如，torch.FloatTensor.abs_()会在原地计算绝对值，并返回改变后的tensor，而tensor.FloatTensor.abs()将会在一个新的tensor中计算结果。

## inplace operation

在 pytorch 中, 有两种情况不能使用 inplace operation:

对于 requires_grad=True 的 叶子张量(leaf tensor) 不能使用 inplace operation
对于在 求梯度阶段需要用到的张量 不能使用 inplace operation
