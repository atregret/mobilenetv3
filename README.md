# MobileNetV3描述

MobileNetV3结合硬件感知神经网络架构搜索（NAS）和NetAdapt算法，可以移植到手机CPU上运行。

[论文](https://arxiv.org/pdf/1905.02244)：Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al."Searching for mobilenetv3."In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324.2019.

# 模型架构

MobileNetV3总体网络架构如下：

[链接](https://arxiv.org/pdf/1905.02244)

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小: 146G, 1330k 1000类彩色图像
    - 训练: 140G, 1280k张图片
    - 测试: 6G, 50k张图片
- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)
- 本模型使用启智平台的算力进行训练

# 脚本说明

## 文件结构

```python
├── MobileNetV3
  ├── README.md     # MobileNetV3相关描述
  ├── src
  │   ├──config.py      # 参数配置
  │   ├──dataset.py     # 创建数据集
  │   ├──loss.py        # 损失函数
  │   ├──lr_generator.py     # 配置学习率
  │   ├──mobilenetV3.py      # MobileNetV3架构
  │   ├──monitor.py          # 监控网络损失和其他数据
  |   └──weight_convert.py   # 权重转换
  ├── eval.py       # 评估脚本
  ├── export.py     # 模型格式转换脚本
  ├── inference.py      # 推理脚本
  ├── train_zhisuan.py     # 启智平台训练脚本
  └── train.py      # 训练脚本
```
## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```python
'num_classes': 1000,                       # 数据集类别数
'image_height': 224,                       # 输入图像高度
'image_width': 224,                        # 输入图像宽度
'batch_size': 512,                         # 数据批次大小
'epoch_size': 360,                         # 模型迭代次数
'warmup_epochs': 4,                        # warmup epoch数量
'lr': 0.05,                                # 学习率
'momentum': 0.9,                           # 动量参数
'weight_decay': 4e-5,                      # 权重衰减率
'label_smooth': 0.1,                       # 标签平滑因子
'loss_scale': 1024,                        # loss scale
'save_checkpoint': True,                   # 是否保存ckpt文件
'save_checkpoint_epochs': 4,               # 每迭代相应次数保存一个ckpt文件
'keep_checkpoint_max': 20,                  # 保存ckpt文件的最大数量
'save_checkpoint_path': "./checkpoint",    # 保存ckpt文件的路径
'export_file': "mobilenetv3",        # export文件
'export_format': "MINDIR",                 # export格式
"device": "Ascend",                         # 设备类型
```

## 训练过程

### 启动

使用python脚本进行训练。

```shell
    python train.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR]
```

### 结果

ckpt文件将存储在 `./checkpoint` 路径下，训练日志将被记录到 `train.txt` 中。训练日志部分示例如下：

```shell
epoch: [  0/360], step:[ 2501/ 2502], loss:[4.621/4.621], time:[1272146.158], lr:[0.012]
epoch time: 1345476.922, per step time: 537.761, avg loss: 4.621
epoch: [  1/360], step:[ 2501/ 2502], loss:[4.202/4.202], time:[1144570.961], lr:[0.025]
epoch time: 1144589.638, per step time: 457.470, avg loss: 4.202
epoch: [  2/360], step:[ 2501/ 2502], loss:[4.219/4.219], time:[1124961.782], lr:[0.037]
epoch time: 1124993.566, per step time: 449.638, avg loss: 4.219
epoch: [  3/360], step:[ 2501/ 2502], loss:[4.255/4.255], time:[1140670.391], lr:[0.050]
epoch time: 1141726.227, per step time: 456.325, avg loss: 4.255
epoch: [  4/360], step:[ 2501/ 2502], loss:[4.165/4.165], time:[1109755.133], lr:[0.050]
epoch time: 1109775.335, per step time: 443.555, avg loss: 4.165
epoch: [  5/360], step:[ 2501/ 2502], loss:[4.081/4.081], time:[1112947.974], lr:[0.050]
epoch time: 1112961.252, per step time: 444.829, avg loss: 4.081
······
······
epoch: [354/360], step:[ 2501/ 2502], loss:[2.370/2.370], time:[990176.584], lr:[0.000]
epoch time: 990190.818, per step time: 395.760, avg loss: 2.370
epoch: [355/360], step:[ 2501/ 2502], loss:[2.223/2.223], time:[988829.701], lr:[0.000]
epoch time: 989896.396, per step time: 395.642, avg loss: 2.223
epoch: [356/360], step:[ 2501/ 2502], loss:[2.256/2.256], time:[1001421.960], lr:[0.000]
epoch time: 1001428.299, per step time: 400.251, avg loss: 2.256
epoch: [357/360], step:[ 2501/ 2502], loss:[2.288/2.288], time:[994389.219], lr:[0.000]
epoch time: 994394.936, per step time: 397.440, avg loss: 2.288
epoch: [358/360], step:[ 2501/ 2502], loss:[2.387/2.387], time:[993284.317], lr:[0.000]
epoch time: 993290.133, per step time: 396.998, avg loss: 2.387
epoch: [359/360], step:[ 2501/ 2502], loss:[2.343/2.343], time:[994759.766], lr:[0.000]
epoch time: 995520.157, per step time: 397.890, avg loss: 2.343

```

## 评估过程

### 启动

使用python脚本进行评估。

```shell
    python eval.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR] --checkpoint_path [PATH_CHECKPOINT]

```

### 结果

```shell
result: {'top_5_accuracy': 0.8687065972222222, 'top_1_accuracy': 0.669921875, 'loss': 2.332750029034085} ckpt= ./mobilenetV3-360_2502.ckpt
```


