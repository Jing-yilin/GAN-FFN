# GAN-FFN

This is a GAN-based feature Fusion Network.
- GPU: V100-32GB

GAN-FNN的训练过程分为2个步骤：
1. 通过GAN实现融合特征生成器的训练
2. 将训练好的实现融合特征生成器接入Backbone，实现情感分类。

`train_IEMOCAP_dialoguernn.py`以DialogueRNN作为Backbone。

## 不进行融合的DialogueRNN表现

```
Loss 0.971 accuracy 62.05
              precision    recall  f1-score   support

           0     0.3902    0.2222    0.2832     144.0
           1     0.8066    0.8000    0.8033     245.0
           2     0.5413    0.5807    0.5603     384.0
           3     0.6566    0.6412    0.6488     170.0
           4     0.6523    0.7592    0.7017     299.0
           5     0.5914    0.5774    0.5843     381.0

    accuracy                         0.6205    1623.0
   macro avg     0.6064    0.5968    0.5969    1623.0
weighted avg     0.6122    0.6205    0.6134    1623.0

[[ 32.   5.  17.   0.  90.   0.]
 [  5. 196.  13.   4.   0.  27.]
 [ 14.  17. 223.  17.  29.  84.]
 [  0.   7.  16. 109.   0.  38.]
 [ 31.   2.  36.   0. 227.   3.]
 [  0.  16. 107.  36.   2. 220.]]

```

# Introduction of model

We now introduce our Generative Adversarial Network-based Feature Fusion Network (GAN-FFN), which generates fused features that can be further used by other sentiment recognition networks. GAN-FFN consists of two parts - a **fused feature generator group** and a **feature discriminator group**. The fused feature generator group includes text, visual, and acoustic fused feature generators, and the feature discriminator group includes text, visual, and acoustic feature discriminators.

# Environment

- python 3.8
- pytorch
- pytorch-geometric
- tensorboardX
- sklearn
- matplotlib
- pandas
- numpy
- openpyxl

## install requirements
```bash
python -m pip install pandas numpy sklearn matplotlib tensorboardX openpyxl
python -m pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/pyg_lib-0.1.0%2Bpt111cu113-cp38-cp38-linux_x86_64.whl
python -m pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
python -m pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
python -m pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl
python -m pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
python -m pip install torch-geometric
```

## Bugs
遇到报错：
```text
ImportError: cannot import name 'container_abcs' from 'torch._six' 
```
解决方法：
修改`/Users/zephyr/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch_geometric/data/dataloader.py`文件,将`from torch._six import container_abcs, string_classes, int_classes`改为`import collections.abc as container_abcs`即可。