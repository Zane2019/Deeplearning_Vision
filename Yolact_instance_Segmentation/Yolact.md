
# Yolact:实时实例分割

> 对于目前的实例分割方法，基本都是在two-stage目标检测算法上加一个 instance  segmentation模块，就像Mask-RCNN一样，但是由于对feature的多次repool，导致速度很慢。

## Yolact

Yolact为了保证速度，设计了讲个分支网络，并行地进行一下操作

- **Prediction Head**分支生成候选框的类别confidence、anchor的location和protitype mask的coefficient
- **Protonet**为每张图片生成k个prototype mask。在代码中k=32。prototype和coefficient的数量相等。
Prototype 意为“原型”。COCO 数据集共有 81 类物体，但每张图片只有 32 个 prototype mask，图片中无论什么类型的实例都可以通过这些 prototype mask 与 coefficient 相乘获得其自己的 mask。这就是“原型”的含义

![](/img/yolact_architecture.png)



