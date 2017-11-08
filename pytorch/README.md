这是天池AI医疗大赛第一赛季PyTorch的实现方案。

##流程

主要流程：

1. 分割网络： 3D的分割，疑似结节的像素点标1，正常组织的像素点标0，目标是从3D的数据中找出疑似结节的位置
2. 连通块查找： 根据分割网络得到的mask图，利用数字图像的处理方式，找出疑似的结节的位置与半径（只要利用skimage库）
3. 分类网络（假阳性衰减）：对找出来的许多疑似节点进行分类，给每个疑似结节一个概率

## 主要文件介绍
- `collecton/`：一些脚本文件，包括计算froc分数的（`cal_froc.py`）,生成提交的csv文件等
- `data/`：数据加载相关，主要是`dataset.py`。
- `models/`: 各个模型的定义文件，包括分类和分割模型，实验了较多的模型
- `utils/`: 可视化工具等
- `config.py`: 配置文件

## 运行

- 训练分割
```Bash
python train.py main --args=......
```

- 分割测试（找出疑似的结节）
```Bash
python test_seg.py seg
```

- 训练分类
```Bash
python train.py main --cls
```

- 测试分类（给每个结节打分）
```Bash
python test_class.py doTest
```
