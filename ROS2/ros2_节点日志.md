# 📝 ROS 2 节点日志 (Logger) 核心用法笔记

## 1. 核心概念：为什么不用 `print()`？
在 ROS 2 开发中，强烈建议使用 `node.get_logger()` 替代 Python 自带的 `print()`。
* **标准化格式**：自动附加时间戳、日志级别和发送该日志的**节点名称**，在多节点并发运行时极大地提升了可读性。
* **系统级集成**：日志不仅输出在终端，还会通过 ROS 2 网络广播，可以被图形化工具（如 `rqt_console`）捕获、过滤，并自动持久化存储到本地文件中。

---

## 2. 日志的 5 个基础级别 (Levels)
ROS 2 提供了 5 种不同严重程度的日志级别。它们在终端中通常会以不同的颜色显示，方便快速定位问题。

| 级别 | 严重程度 | 用法场景 | 代码示例 |
| :--- | :--- | :--- | :--- |
| **DEBUG** | 最低 | 仅用于开发阶段排查详细变量值，默认**不显示**。 | `node.get_logger().debug('当前 x 的值为: 5')` |
| **INFO** | 正常 | 汇报程序的正常运行状态和进度（最常用）。 | `node.get_logger().info('节点已成功启动！')` |
| **WARN** | 警告 | 出现潜在异常，但程序仍可继续运行（如信号弱）。 | `node.get_logger().warning('传感器延迟较高！')` |
| **ERROR** | 错误 | 发生严重错误，某项具体功能失效，需要介入。 | `node.get_logger().error('无法连接到摄像头！')` |
| **FATAL** | 致命 | 导致整个程序崩溃或必须强制退出的毁灭性错误。 | `node.get_logger().fatal('内存溢出，系统即将关闭！')` |

---

## 3. 进阶用法：控制输出频率 (🔥 极其推荐)
在循环（如定时器回调或控制循环）中直接使用 `info()` 会导致终端瞬间被刷屏。ROS 2 提供了非常优雅的频率控制方法：

### 3.1 节流输出 (Throttle)
无论这行代码每秒被执行多少次，它都**最多只打印一次**。非常适合持续监测状态。
```python
# 第一个参数是限制的秒数 (例如 1.0 表示 1 秒内最多打印 1 次)
node.get_logger().info('机器人正在以 1m/s 的速度前进...', throttle_duration_sec=1.0)
```

### 3.2 仅输出一次 (Once)
无论这行代码被触发多少次，在整个节点的生命周期中，**只在第一次执行时打印**。
```python
node.get_logger().info('已成功进入巡航模式！', once=True)
```

### 3.3 跳过输出 (Skip First)
忽略第一次触发，从第二次开始正常打印。
```python
node.get_logger().info('正在接收连续数据流...', skip_first=True)
```

---

## 4. 完整的代码应用模板
以下是一个包含多种日志用法的标准节点模板：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

class LogDemoNode(Node):
    def __init__(self):
        super().__init__('log_demo_node')
        
        # 1. 基础打印：节点启动时的欢迎信息
        self.get_logger().info('你好，LogDemoNode 已启动！')
        
        # 2. 警告打印
        self.get_logger().warning('注意：当前使用的是模拟数据。')
        
        # 3. 创建一个高频定时器 (每 0.1 秒执行一次 = 10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        # 进阶用法：强制限制为每 2 秒打印一次，避免刷屏
        self.get_logger().info('系统状态良好，正在持续处理数据...', throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = LogDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('检测到退出指令，准备关闭...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 5. 调试与排查工具
* **图形化查看日志**：在终端输入 `ros2 run rqt_console rqt_console`，可以打开一个清晰的 GUI 窗口，对所有节点的日志进行筛选、排除和级别过滤。
* **本地日志文件路径**：如果程序崩溃，可以在 `~/.ros/log/` 目录下找到按时间分类保存的 `.log` 历史文件。

# 🛠️ ROS 2 日志系统进阶配置笔记

除了修改输出格式，ROS 2 还提供了多种方式来深度定制日志行为。

---

## 1. 深度定制输出格式 (RCUTILS_CONSOLE_OUTPUT_FORMAT)*(日志格式环境变量)
这是你在图片中看到的用法。除了 `{message}`，你还可以自由组合以下占位符（Tokens）：

| 占位符 | 说明 |
| :--- | :--- |
| `{severity}` | 日志级别（DEBUG, INFO, WARN...） |
| `{name}` | 节点/记录器名称 |
| `{message}` | 实际输出的日志文字 |
| `{time}` | 时间戳（秒） |
| `{time_as_nanoseconds}` | 时间戳（纳秒） |
| `{function_name}` | 调用该日志的**函数名** |
| `{file_name}` | 源代码**文件名** |
| `{line_number}` | 源代码**行号** |

**💡 调试神器推荐格式：**
```bash
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{time}] [{name}] [{function_name}:{line_number}]: {message}"
```
# ROS 2 日志格式配置符号说明
## 符号作用
1. `{}`：语法核心，RCUTILS识别的占位符标记，包裹的关键词（如severity/time）会被实际日志数据替换；
2. `[]`：纯视觉分隔符，无语法意义，用于区分日志字段，提升可读性；
3. `:`：纯语义/视觉分隔符，无语法意义，分隔元数据与核心日志消息。

## 关键注意
- 仅`{}`是功能必需符号，`[]`/`:`为可选美化符号；
- 配置仅对当前Shell会话生效，永久生效需写入~/.bashrc。
---

## 2. 运行时配置：修改日志级别 (Log Level)
有时候你不需要修改代码，只想在启动时看到更多（或更少）的信息。

### 方式 A：启动命令参数
在运行节点时直接指定级别：
```bash
# 将整个节点的日志级别设为 DEBUG
ros2 run my_pkg my_node --ros-args --log-level debug

# 仅修改特定节点的日志级别
ros2 run my_pkg my_node --ros-args --log-level village_li:=warn
```

### 方式 B：通过 Python 代码设置
在节点内部，根据逻辑动态修改：
```python
import rclpy
from rclpy.logging import LoggingSeverity

# ... 在节点类的方法中 ...
self.get_logger().set_level(LoggingSeverity.DEBUG)
```

---

## 3. 环境控制：外观与行为
这些环境变量通常写在 `~/.bashrc` 中：

| 环境变量 | 作用 | 取值示例 |
| :--- | :--- | :--- |
| **`RCUTILS_COLORIZED_OUTPUT`** | 强制开启/关闭彩色日志 | `1` (开), `0` (关) |
| **`RCUTILS_LOGGING_USE_STDOUT`** | 默认输出到标准输出而非标准错误 | `1` (是) |
| **`ROS_LOG_DIR`** | 自定义日志文件保存路径 | `export ROS_LOG_DIR=~/my_logs` |

---

## 4. 自动化工具：rqt_console
如果你觉得终端里的文字还是太乱，可以使用这个图形化工具：
```bash
ros2 run rqt_console rqt_console
```
* **功能**：它可以自动收集网络上所有节点的日志，并提供**搜索、过滤、级别高亮**等功能，是复杂项目调试的必备工具。

---

## 5. 永久保存配置
如果你想让这些配置（比如格式修改）在每次开机时都生效，请将其加入到你的 Linux 配置文件中：
1. `nano ~/.bashrc`
2. 在文件末尾添加：`export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"`
3. 保存退出并执行：`source ~/.bashrc`
