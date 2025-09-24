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
[文献](https://zhuanlan.zhihu.com/p/704991790)

### 优化器
`import torch.optim as optim`

- SGD 优化器
`optimizer = optim.SGD(model.parameters(), lr=0.01)`

- Adam 优化器
`optimizer = optim.Adam(model.parameters(), lr=0.001)`

### torch基本函数
1. `torch.rand(*size, dtype=None, device=None, requires_grad=False)`
   - size：指定张量的形状，可以是多个整数值，表示每个维度的大小。例如：(3, 2) 表示生成一个 3x2 的张量。
   - dtype（可选）：指定返回张量的数据类型。默认是 torch.float32。
   - device（可选）：指定张量存放的设备（如 CPU 或 GPU）。例如，可以指定为 torch.device('cuda') 或 torch.device('cpu')。
   - requires_grad（可选）：布尔值，默认值为 False。如果设置为 True，则返回的张量会记录操作，以便进行自动求导（通常用于训练神经网络时）。



### 张量
1. 创建

| 方法                         | 说明                                          | 示例代码                                      |
|------------------------------|-----------------------------------------------|-----------------------------------------------|
| `torch.tensor(data)`          | 从 Python 列表或 NumPy 数组创建张量。           | `x = torch.tensor([[1, 2], [3, 4]])`           |
| `torch.zeros(size)`           | 创建一个全为零的张量。                         | `x = torch.zeros((2, 3))`                     |
| `torch.ones(size)`            | 创建一个全为 1 的张量。                        | `x = torch.ones((2, 3))`                      |
| `torch.empty(size)`           | 创建一个未初始化的张量。                       | `x = torch.empty((2, 3))`                     |
| `torch.rand(size)`            | 创建一个服从均匀分布的随机张量，值在 [0, 1) 之间。 | `x = torch.rand((2, 3))`                      |
| `torch.randn(size)`           | 创建一个服从正态分布的随机张量，均值为 0，标准差为 1。 | `x = torch.randn((2, 3))`                     |
| `torch.arange(start, end, step)` | 创建一个一维序列张量，类似于 Python 的 `range`。   | `x = torch.arange(0, 10, 2)`                  |
| `torch.linspace(start, end, steps)` | 创建一个在指定范围内均匀间隔的序列张量。         | `x = torch.linspace(0, 1, 5)`                 |
| `torch.eye(n)`                | 创建一个单位矩阵（对角线为 1，其他为 0）。       | `x = torch.eye(3)`                            |
| `torch.from_numpy(ndarray)`   | 将 NumPy 数组转换为张量。                      | `x = torch.from_numpy(np.array([1, 2, 3]))`    |

2. 属性

| 属性                    | 说明                                    | 示例代码                  |
|-------------------------|-----------------------------------------|---------------------------|
| `.shape`                | 获取张量的形状                          | `tensor.shape`             |
| `.size()`               | 获取张量的形状                          | `tensor.size()`            |
| `.dtype`                | 获取张量的数据类型                      | `tensor.dtype`             |
| `.device`               | 查看张量所在的设备（CPU/GPU）            | `tensor.device`            |
| `.dim()`                | 获取张量的维度数                        | `tensor.dim()`             |
| `.requires_grad`        | 是否启用梯度计算                        | `tensor.requires_grad`     |
| `.numel()`              | 获取张量中的元素总数                    | `tensor.numel()`           |
| `.is_cuda`              | 检查张量是否在 GPU 上                   | `tensor.is_cuda`           |
| `.T`                    | 获取张量的转置（适用于 2D 张量）         | `tensor.T`                 |
| `.item()`               | 获取单元素张量的值                      | `tensor.item()`            |
| `.is_contiguous()`      | 检查张量是否连续存储                    | `tensor.is_contiguous()`   |

3. 基本操作

| 操作 | 说明 | 示例代码 |
| :--- | :--- | :--- |
| `+`, `-`, `*`, `/` | 元素级加法、减法、乘法、除法。 | `z = x + y` |
| `torch.matmul(x, y)` | 矩阵乘法。 | `z = torch.matmul(x, y)` |
| `torch.dot(x, y)` | 向量点积（仅适用于1D张量）。 | `z = torch.dot(x, y)` |
| `torch.sum(x)` | 求和。 | `z = torch.sum(x)` |
| `torch.mean(x)` | 求均值。 | `z = torch.mean(x)` |
| `torch.max(x)` | 求最大值。 | `z = torch.max(x)` |
| `torch.min(x)` | 求最小值。 | `z = torch.min(x)` |
| `torch.argmax(x, dim)` | 返回最大值的索引（指定维度）。 | `z = torch.argmax(x, dim=1)` |
| `torch.softmax(x, dim)` | 计算softmax（指定维度）。 | `z = torch.softmax(x, dim=1)` |

4. 形态操作

| 操作 | 说明 | 示例代码 |
| :--- | :--- | :--- |
| `x.view(shape)` | 改变张量的形状（不改变数据）。 | `z = x.view(3, 4)` |
| `x.reshape(shape)` | 类似于view，但更灵活。 | `z = x.reshape(3, 4)` |
| `x.t()` | 转置矩阵。 | `z = x.t()` |
| `x.unsqueeze(dim)` | 在指定维度添加一个维度。 | `z = x.unsqueeze(0)` |
| `x.squeeze(dim)` | 去掉指定维度为1的维度。 | `z = x.squeeze(0)` |
| `torch.cat((x, y), dim)` | 按指定维度连接多个张量。 | `z = torch.cat((x, y), dim=1)` |

