## GAN-FFN

We now introduce our Generative Adversarial Network-based Feature Fusion Network (GAN-FFN), which generates fused features that can be further used by other sentiment recognition networks. GAN-FFN consists of two parts - a **fused feature generator group** and a **feature discriminator group**. The fused feature generator group includes text, visual, and acoustic fused feature generators, and the feature discriminator group includes text, visual, and acoustic feature discriminators.

## Best performance

`train_IEMOCAP.py`:
![GAN-loss](https://img.jing10.top/uPic/202307012248371688222917mFKocc.jpg)
```text
Test performance..
Loss 1.025 F1-score 59.65
              precision    recall  f1-score   support

           0     0.4048    0.3542    0.3778     144.0
           1     0.7887    0.6245    0.6970     245.0
           2     0.5603    0.5443    0.5522     384.0
           3     0.5534    0.6706    0.6064     170.0
           4     0.6451    0.7659    0.7003     299.0
           5     0.5827    0.5643    0.5733     381.0

    accuracy                         0.5983    1623.0
   macro avg     0.5891    0.5873    0.5845    1623.0
weighted avg     0.6011    0.5983    0.5965    1623.0

[[ 51.   9.   7.   2.  73.   2.]
 [  6. 153.  36.   4.   4.  42.]
 [ 25.  23. 209.  20.  42.  65.]
 [  2.   0.  10. 114.   0.  44.]
 [ 32.   0.  35.   2. 229.   1.]
 [ 10.   9.  76.  64.   7. 215.]]
```

## Environment
- Device: V100-32GB * 1卡
- python 3.8

## 在autodl的服务器上训练

```bash
source /etc/network_turbo # 打开加速
git config --global user.name "YOUR_USERNAME" # 设置git用户名
git config user.email "YOUR_EMAIL" # 设置git邮箱

python -m pip install pandas numpy sklearn matplotlib tensorboardX openpyxl
python -m pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/pyg_lib-0.1.0%2Bpt111cu113-cp38-cp38-linux_x86_64.whl
python -m pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
python -m pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
python -m pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl
python -m pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
python -m pip install torch-geometric
```

## Performance

```text
Loss 1.025 F1-score 59.65
              precision    recall  f1-score   support

           0     0.4048    0.3542    0.3778     144.0
           1     0.7887    0.6245    0.6970     245.0
           2     0.5603    0.5443    0.5522     384.0
           3     0.5534    0.6706    0.6064     170.0
           4     0.6451    0.7659    0.7003     299.0
           5     0.5827    0.5643    0.5733     381.0

    accuracy                         0.5983    1623.0
   macro avg     0.5891    0.5873    0.5845    1623.0
weighted avg     0.6011    0.5983    0.5965    1623.0
`

## Bugs
### 1. torch._six
```text
遇到报错：
```text
ImportError: cannot import name 'container_abcs' from 'torch._six' 
```
解决方法：
修改`/Users/zephyr/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch_geometric/data/dataloader.py`文件,将`from torch._six import container_abcs, string_classes, int_classes`改为`import collections.abc as container_abcs`即可。


### 2. 训练精度问题
经本人测试，目前如果使用一张V100-32GB卡训练，F1-score=59.56；如果使用两张V100-32GB卡训练，F1-score明显下降，原因不明。