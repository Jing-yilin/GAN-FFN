## GAN-FFN

We now introduce our Generative Adversarial Network-based Feature Fusion Network (GAN-FFN), which generates fused features that can be further used by other sentiment recognition networks. GAN-FFN consists of two parts - a **fused feature generator group** and a **feature discriminator group**. The fused feature generator group includes text, visual, and acoustic fused feature generators, and the feature discriminator group includes text, visual, and acoustic feature discriminators.

## Environment

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

## Bugs
遇到报错：
```text
ImportError: cannot import name 'container_abcs' from 'torch._six' 
```
解决方法：
修改`/Users/zephyr/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch_geometric/data/dataloader.py`文件,将`from torch._six import container_abcs, string_classes, int_classes`改为`import collections.abc as container_abcs`即可。
