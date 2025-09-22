### 激活函数
- ReLU 激活（非线性激活函数）
`torch.relu(input_tensor)`
$f(x) = max(0, x)$
ReLu函数相比于Sigmoid函数和Tanh函数具有更强的非线性拟合能力 没有梯度消失 能够最大化的发挥神经元的筛选能力
实际收敛速度较快，比 Sigmoid/tanh 快很多

- Sigmoid 激活
`output = torch.sigmoid(input_tensor)`
是常用的连续、平滑的s型激活函数，也被称为逻辑（Logistic）函数。可以将一个实数映射到（0，1）的区间，用来做二分类
$ \sigma(x) = \frac{1}{1 + e^{-x}} $

- Tanh 激活
`output = torch.tanh(input_tensor)`
值域为 (-1,1)
$ f(x) = \tanh (x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} $

- [文献](https://zhuanlan.zhihu.com/p/360980567)

### 损失函数
- 均方误差损失
`criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')`
- size_average (bool, 可选)已弃用
默认情况下，损失在批次中的每个损失元素上取平均（True）；否则（False），在每个小批次中对损失求和。
默认值是 True。
- reduce (bool, 可选)已弃用
默认情况下，损失根据 size_average 参数进行平均或求和。
当 reduce 为 False 时，返回每个批次元素的损失，并忽略 size_average 参数。
默认值是 True。
- reduction (str, 可选):
指定应用于输出的归约方式。
可选值为 'none'、'mean'、'sum'。
'none'：不进行归约 返回每个样本的损失组成的张量 
'mean'：输出的和除以输出的元素总数 损失是所有样本损失的平均值
'sum'：输出的元素求和 损失是所有样本损失的和
注意：size_average 和 reduce 参数正在被弃用，同时指定这些参数中的任何一个都会覆盖 reduction 参数。
默认值是 'mean'。
- 示例
```
import torch
import torch.nn as nn

# 定义输入和目标张量
input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

# 使用 nn.MSELoss 计算损失（reduction='mean'）
criterion_mean = nn.MSELoss(reduction='mean')
loss_mean = criterion_mean(input, target)
print(f"Loss with reduction='mean': {loss_mean.item()}")

# 使用 nn.MSELoss 计算损失（reduction='sum'）
criterion_sum = nn.MSELoss(reduction='sum')
loss_sum = criterion_sum(input, target)
print(f"Loss with reduction='sum': {loss_sum.item()}")

# 使用 nn.MSELoss 计算损失（reduction='none'）
criterion_none = nn.MSELoss(reduction='none')
loss_none = criterion_none(input, target)
print(f"Loss with reduction='none': {loss_none}")
```
```
>>> Loss with reduction='mean': 0.25
>>> Loss with reduction='sum': 1.0
>>> Loss with reduction='none': tensor([[0.2500, 0.2500],
        [0.2500, 0.2500]], grad_fn=<MseLossBackward0>)
```
- 交叉熵损失
`criterion = nn.CrossEntropyLoss()`

- 二分类交叉熵损失
`criterion = nn.BCEWithLogitsLoss()`

### 优化器
