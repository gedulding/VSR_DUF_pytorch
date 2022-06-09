# VSR_DUF_pytorch
According to the author's paper and code, it is converted to pytorch implementation. This code can be used as the basis for reproducing this paper. And the code content can be used as a template

本代码主要是对论文：Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation。论文进行复现。根据原作者提供的代码，以及一些代码摹本，将该论文复现为pythorch实现。

## 论文相关内容

***论文地址***：https://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf

 **本人对论文内容的阅读理解**：https://blog.csdn.net/Gedulding/article/details/124856566

## 代码说明：
本文主要是对模型中16层深度的代码进行了复现，根据原文作者的代码内容，进行了16层和x4倍的实现。包含前期的数据处理内容，已经基本复现结束，不过在本机上无法进行训练，所以没有进行训练测试，进行了数据size和shape的比较。
### 模型部分：(全貌运行查看)
![在这里插入图片描述](https://img-blog.csdnimg.cn/5c369e0f18b7438480ac14775c3bdfe9.png#pic_center =400x500)
### 代码运行说明：

       >***python main.py --R --L -- resume --path_to_checkpoint***

       >例如执行 ：python main.py --R 4 --L 16

--R 表示Upscaling factor，目前文中使用x4倍数，修改倍数，需要手动修改代码内容，替换为输入参数
--L 表示Network depth，目前根据论文复现为16层深度
--resume 表示是否执行断点恢复
--path_to_checkpoint  指定已经保存模型ckpt的路径

### 代码结构说明：
下图包含了本次代码的基本结构，其中工具类中包括一些本次未使用的，例如图像下采样，以及YCbCr等转化，也包含了作者使用的PSNR计算方法。

![在这里插入图片描述](https://img-blog.csdnimg.cn/a56199ff64624401b9613253ff84da3e.png)

### 代码缺陷说明：
1，代码部分未完全进行实现，主要是集中在最后一部分，获得了filter和高频信息后需要级联，然后恢复出高频图像内容，可以对比原文作者在test.py中G函数的内容实现。已预留此部分内容在net.py中。![在这里插入图片描述](https://img-blog.csdnimg.cn/34a150aa8644492d8483e026324476c2.png)

2，文章中输入模型的图像大小为32*32，我直接输入了完整图像，未进行裁剪。
## 论文实现效果
![在这里插入图片描述](https://img-blog.csdnimg.cn/27811c3ca8e74cbbaf4dcce2f95aa0dd.png#pic_center)
模型参数和运行效率比对：

![运行效率和参数比对](https://img-blog.csdnimg.cn/bb2f116dde384191bca38ec9cbe331fe.png#pic_center)
## 代码参考
https://github.com/yhjo09/VSR-DUF

https://github.com/IgorSusmelj/pytorch-styleguide
