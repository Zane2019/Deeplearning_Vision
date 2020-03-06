# Faster CNN
> 转自https://zhuanlan.zhihu.com/p/31426458，部分删减
> 
RCNN系列的神经网络在图像检测和分类领域具有高效率，并且其mAP（mean Average Precision）分数比之前的技术更高.

经过RCNN,Fast RCNN的几点，经过R-CNN和Fast RCNN的积淀，Ross B. Girshick在2016年提出了新的Faster RCNN，在结构上，Faster RCNN已经将特征抽取(feature extraction)，proposal提取，bounding box regression(rect refine)，classification都整合在了一个网络中，使得综合性能有较大提高，在检测速度方面尤为明显。

![](/img/Faster_RCNN_Structure.png)

从图上看来，Faster RCNN主要分为四个主要内容：

1. Conv layers。Faster RCNN首先使用一组基础的conv+relu+pooling提取feature maps.该feature maps共享用于后续RPN层和全连接层。
2. Region Proposal Networks. RPN网络用于生成region proposal（每张图给出大概20000个候选框）。该层通过softmax判断anchor属于positive/negative。再利用bounding box regression修正anchors获得精确的proposal.
3. ROI Pooling。该层收集输入的feature map和proposal。综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。
4. Classfication。利用proposal feature map 计算proposal的类别，同时再次bounding box regression获得检测狂最终精确的位置。

## Content

[TOC]


以VGG16模型中的faster_rcnn_test.pt的网络结构讲解
![avatar](/img/Faster_RCNN_vgg16.png)

##  Conv layers

Conv layers包含了conv，pooling，relu三种层。以python版本中的VGG16模型中的faster_rcnn_test.pt的网络结构为例，如图2，Conv layers部分共有13个conv层，13个relu层，4个pooling层。
所有的conv层都是：kernel_size=3，pad=1
所有的pooling层都是：kernel_size=2，stride=2
> 在Faster RCNN Conv layers中对所有的卷积都做了扩边处理（pad=1，即填充一圈0），导致原图变为(M+2)x(N+2)大小，再做3x3卷积后输出MxN。正是这种设置，导致Conv layers中的conv层不改变输入和输出矩阵大小.pooling层kernel_size=2，stride=2。这样每个经过pooling层的MxN矩阵，都会变为(M/2)*(N/2)大小。综上所述，在整个Conv layers中，conv和relu层不改变输入输出大小，只有pooling层使输出长宽都变为输入的1/2

一个MxN大小的矩阵经过Conv layers固定变为(M/16)x(N/16)

## Region Proposal NetWorks(RPN)

经典的检测方法生成检测框都非常耗时，如OpenCV adaboost使用滑动窗口+图像金字塔生成检测框；或如RCNN使用SS(Selective Search)方法生成检测框。Faster RCNN则抛弃了传统的滑动窗口和SS方法，直接使用RPN生成检测框，这也是Faster RCNN的巨大优势，能极大提升检测框的生成速度。
![avatar](/img/Faster_RCNN_RPN.png)

上图展示了RPN网络的具体结构。可以看到RPN网络实际分为2条线，上面一条通过softmax分类anchors获得foreground和background（检测目标是foreground），下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。而最后的Proposal层则负责综合foreground anchors和bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。



### Anchors

anchors实际上是一组举行。其中每行的4个值[x1,y1,x2,y2]代表矩形左上和右下角点坐标。9个矩形共有3种形状，长宽比为大约为：width:height = [1:1, 1:2, 2:1]三种，如图6。实际上通过anchors就引入了检测中常用到的多尺度方法。
![avatar](/img/Faster_RCNN_Anchors.png)
>关于anchors size，其实是根据检测图像设置的。在python demo中，会把任意大小的输入图像reshape成800x600（即图2中的M=800，N=600）。再回头来看anchors的大小，anchors中长宽1:2中最大为352x704，长宽2:1中最大736x384，基本是cover了800x600的各个尺度和形状.

遍历Conv layers计算获得的feature maps，为每一个点都配备这9种anchors作为初始的检测框。这样做获得检测框很不准确，不用担心，后面还有2次bounding box regression可以修正检测框位置。

![avatar](/img/Faster_RCNN_Anchors_2.png)
> 在原文中使用的是ZF model中，其Conv Layers中最后的conv5层num_output=256，对应生成256张特征图，所以相当于feature map每个点都是256-d
在conv5之后，做了rpn_conv/3x3卷积且num_output=256，相当于每个点又融合了周围3x3的空间信息（猜测这样做也许更鲁棒？反正我没测试），同时256-d不变（如图4和图7中的红框）
假设在conv5 feature map中每个点上有k个anchor（默认k=9），而每个anhcor要分foreground和background，所以每个点由256d feature转化为cls=2k scores；而每个anchor都有[x, y, w, h]对应4个偏移量，所以reg=4k coordinates
补充一点，全部anchors拿去训练太多了，训练程序会选取256个合适的anchors进行训练（**什么是合适的anchors**)后续有解释



### Softmax判定foreground与background

一副MxN大小的矩阵送入Faster RCNN网络后，到RPN网络变为(M/16)x(N/16)，不妨设W=M/16，H=N/16。在进入reshape与softmax之前，先做了1x1卷积，如下图：

![avatar](/img/Faster_RCNN_rpn_classfication.png)
以看到其num_output=18，也就是经过该卷积的输出图像为18xWxH大小
这也就刚好对应了feature maps每一个点都有9个anchors，同时每个anchors又有可能是foreground和background，所有这些信息都保存WxHx(9x2)大小的矩阵。为何这样做？后面接softmax分类获得foreground anchors，也就相当于初步提取了检测目标候选区域box（一般认为目标在foreground anchors中）

**那么为何要在softmax前后都接一个reshape layer？** 其实只是为了便于softmax分类，至于具体原因这就要从caffe的实现形式说起了。在caffe基本数据结构blob中以如下形式保存数据：
blob=[batch_size, channel，height，width]
对应至上面的保存bg/fg anchors的矩阵，其在caffe blob中的存储形式为[1, 2*9, H, W]。而在softmax分类时需要进行fg/bg二分类，所以reshape layer会将其变为[1, 2, 9*H, W]大小，即单独“腾空”出来一个维度以便softmax分类，之后再reshape回复原状。贴一段caffe softmax_loss_layer.cpp的reshape函数的解释，非常精辟：
```cpp
//"Number of labels must match number of predictions; "  
//"e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "  
//"label count (number of labels) must be N*H*W, "  
//"with integer values in {0, 1, ..., C-1}.";  
```
综上所述，RPN网络中利用anchors和softmax初步提取出foreground anchors作为候选区域.


### Bounding box regression原理
介绍bounding box regression数学模型及原理。如下图所示绿色框为飞机的Ground Truth(GT)，红色为提取的foreground anchors，那么即便红色的框被分类器识别为飞机，但是由于红色的框定位不准，这张图相当于没有正确的检测出飞机。所以我们希望采用一种方法对红色的框进行微调，使得foreground anchors和GT更加接近。
![avatar](/img/Faster_RCNN_rpb_bounding_box.png)

对于窗口一般使用四维向量(x, y, w, h)表示，分别表示窗口的中心点坐标和宽高。对于下图，红色的框A代表原始的Foreground Anchors，绿色的框G代表目标的GT，我们的目标是寻找一种关系，使得输入原始的anchor A经过映射得到一个跟真实窗口G更接近的回归窗口G'，即：给定A=(Ax, Ay, Aw, Ah)，寻找一种映射f，使得f(Ax, Ay, Aw, Ah)=(G'x, G'y, G'w, G'h)，其中(G'x, G'y, G'w, G'h)≈(Gx, Gy, Gw, Gh)。
![avatar](/img/Faster_RCNN_rpb_bounding_box2.png)

那么经过何种变换才能从图6中的A变为G'呢？ 比较简单的思路就是:
1. 先做平移：
   $$
   \begin{aligned}
       G'_x=A_w\cdot d_x(A)+A_x \\
       G'_y=A_h\cdot d_y(A)+A_y
   \end{aligned}
   $$
2. 再做缩放：
   $$
   \begin{aligned}
   G'_w=A_w \cdot exp(d_w(A)) \\
   G'_h=A_h \cdot exp(d_h(A))
   \end{aligned}
   $$
观察上面4个公式发现，需要学习的是dx(A)，dy(A)，dw(A)，dh(A)这四个变换。当输入的anchor与GT相差较小时，可以认为这种变换是一种线性变换， 那么就可以用线性回归来建模对窗口进行微调（注意，只有当anchors和GT比较接近时，才能使用线性回归模型，否则就是复杂的非线性问题了）。对应于Faster RCNN原文，平移量(tx, ty)与尺度因子(tw, th)如下
$$
\begin{aligned}
t_x=(x-x_a)/w_a,\\
t_y=(y-y_a)/h_a,\\
t_w=log(w/w_a),\\
t_h=log(h/h_a)
\end{aligned}
$$
接下来的问题就是如何通过线性回归获得dx(A)，dy(A)，dw(A)，dh(A)了。线性回归就是给定输入的特征向量X, 学习一组参数W, 使得经过线性回归后的值跟真实值Y（即GT）非常接近，即Y=WX。对于该问题，输入X是一张经过num_output=1的1x1卷积获得的feature map，定义为Φ；同时还有训练传入的GT，即(tx, ty, tw, th)。输出是dx(A)，dy(A)，dw(A)，dh(A)四个变换。那么目标函数可以表示为:
$$
d_*(A)=w^T_x*\Phi(A)
$$

其中Φ(A)是对应anchor的feature map组成的特征向量，w是需要学习的参数，d(A)是得到的预测值（*表示 x，y，w，h，也就是每一个变换对应一个上述目标函数）。为了让预测值(tx, ty, tw, th)与真实值最小，得到损失函数：
$$Loss=\sum_i^N(t_*^i-\hat w\cdot \Phi(A^i))^2$$
函数优化目标为：
$$w_*=\argmin_{\hat{w_*}}\sum^N_i(t_*^i-\hat w_*^T\cdot\Phi(A^i))^2+\lambda||\hat{ w_*}||^2$$

### 对Proposals进行bounding box regression


![avatar](/img/Faster_RCNN_rpb_bounding_box_regression.png)
可以看到其num_output=36，即经过该卷积输出图像为WxHx36，在caffe blob存储为[1, 36, H, W]。与上文中fg/bg anchors存储为[1, 18, H, W]类似，这里相当于feature maps每个点都有9个anchors，每个anchors又都有4个用于回归的[dx(A)，dy(A)，dw(A)，dh(A)]变换量。利用上面的的计算公式即可从foreground anchors回归出proposals。

### Proposal Layer
Proposal Layer负责综合所有[dx(A)，dy(A)，dw(A)，dh(A)]变换量和foreground anchors，计算出精准的proposal，送入后续RoI Pooling Layer。

Proposal Layer有3个输入：fg/bg anchors分类器结果rpn_cls_prob_reshape，对应的bbox reg的[dx(A)，dy(A)，dw(A)，dh(A)]变换量rpn_bbox_pred，以及im_info；另外还有参数feat_stride=16，这和图4是对应的。

首先解释im_info。对于一副任意大小PxQ图像，传入Faster RCNN前首先reshape到固定MxN，im_info=[M, N, scale_factor]则保存了此次缩放的所有信息。然后经过Conv Layers，经过4次pooling变为WxH=(M/16)x(N/16)大小，其中feature_stride=16则保存了该信息。所有这些数值都是为了将proposal映射回原图而设置的，如图12，毕竟检测就是为了在原图上画一个框而已。

> Proposal Layer forward（caffe layer的前传函数）按照以下顺序依次处理：
再次生成anchors，并对所有的anchors做bbox reg位置回归（注意这里的anchors生成顺序和之前是即完全一致的）
按照输入的foreground softmax scores由大到小排序anchors，提取前pre_nms_topN(e.g. 6000)个anchors。即提取修正位置后的foreground anchors
利用feat_stride和im_info将anchors映射回原图，判断fg anchors是否大范围超过边界，剔除严重超出边界fg anchors。
进行nms（nonmaximum suppression，非极大值抑制）
再次按照nms后的foreground softmax scores由大到小排序fg anchors，提取前post_nms_topN(e.g. 300)结果作为proposal输出。
之后输出proposal=[x1, y1, x2, y2]，注意，由于在第三步中将anchors映射回原图判断是否超出边界，所以这里输出的proposal是对应MxN输入图像尺度的，这点在后续网络中有用。另外我认为，严格意义上的检测应该到此就结束了，后续部分应该属于识别了~

RPN网络结构就介绍到这里，总结起来就是：
**生成anchors -> softmax分类器提取fg anchors -> bbox reg回归fg anchors -> Proposal Layer生成proposals**

## ROI pooling 

而RoI Pooling层则负责收集proposal，并计算出proposal feature maps，送入后续网络。从图3中可以看到Rol pooling层有2个输入：

原始的feature maps
RPN输出的proposal boxes（大小各不相同）

为什么需要ROI Pooling?
对于传统的CNN（如AlexNet，VGG），当网络训练好后输入的图像尺寸必须是固定值，同时网络输出也是固定大小的vector or matrix。如果输入图像大小不定，这个问题就变得比较麻烦。
回忆RPN网络生成的proposals的方法：对foreground anchors进行bound box regression，那么这样获得的proposals也是大小形状各不相同，即也存在上述问题。所以Faster RCNN中提出了RoI Pooling解决这个问题（需要说明，RoI Pooling其实是从SPP发展而来）

### ROI Pooling 原理

分析之前先来看看RoI Pooling Layer的caffe prototxt的定义：
```cpp
layer {  
  name: "roi_pool5"  
  type: "ROIPooling"  
  bottom: "conv5_3"  
  bottom: "rois"  
  top: "pool5"  
  roi_pooling_param {  
    pooled_w: 7  
    pooled_h: 7  
    spatial_scale: 0.0625 # 1/16  
  }  
}  
```
其中有新参数pooled_w=pooled_h=7，另外一个参数spatial_scale=1/16。

RoI Pooling layer forward过程：在之前有明确提到：proposal=[x1, y1, x2, y2]是对应MxN尺度的，所以首先使用spatial_scale参数将其映射回MxN大小的feature maps尺度（这里来回多次映射，是有点绕）；之后将每个proposal水平和竖直都分为7份，对每一份都进行max pooling处理。这样处理后，即使大小不同的proposal，输出结果都是7x7大小，实现了fixed-length output。
![avatar](/img/Faster_RCNN_proposal.png)

## Classfication
Classification部分利用已经获得的proposal feature maps，通过full connect层与softmax计算每个proposal具体属于那个类别（如人，车，电视等），输出cls_prob概率向量；同时再次利用bounding box regression获得每个proposal的位置偏移量bbox_pred，用于回归更加精确的目标检测框。Classification部分网络结构如图
![avatar](/img/Faster_RCNN_classfication.png)

从PoI Pooling获取到7x7=49大小的proposal feature maps后，送入后续网络，可以看到做了如下2件事：

通过全连接和softmax对proposals进行分类，这实际上已经是识别的范畴了
再次对proposals进行bounding box regression，获取更高精度的rect box


## QA

1. **为何有ROI Pooling还要把输入图片resize到固定大小的MxN?**
 
 由于引入ROI Pooling，从原理上说Faster R-CNN确实能够检测任意大小的图片。但是由于在训练的时候需要使用大batch训练网络，而不同大小输入拼batch在实现的时候代码较为复杂，而且当时以Caffe为代表的第一代深度学习框架也不如Tensorflow和PyTorch灵活，所以作者选择了把输入图片resize到固定大小的800x600。这应该算是历史遗留问题。