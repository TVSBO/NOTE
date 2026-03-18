# ROS 2 基础核心用法笔记

本笔记涵盖了 ROS 2 的基础概念及高频使用场景，代码示例结合了 C++ 和 Python 的主流用法，适用于快速回顾和日常开发查阅。

---

## 1. 工作空间与功能包管理

在使用 ROS 2 时，所有代码都需要组织在工作空间（Workspace）和功能包（Package）中。

### 创建工作空间
```bash
# 创建工作空间目录
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 创建功能包
```bash
# 创建一个 C++ 功能包
ros2 pkg create --build-type ament_cmake my_cpp_pkg --dependencies rclcpp std_msgs

# 创建一个 Python 功能包
ros2 pkg create --build-type ament_python my_py_pkg --dependencies rclpy std_msgs
```

### 编译与环境变量
```bash
cd ~/ros2_ws
# 编译整个工作空间 (推荐安装 colcon 编译工具)
colcon build

# 仅编译指定包
colcon build --packages-select my_cpp_pkg

# 刷新环境变量 (每次新开终端或编译新节点后都需要)
source install/setup.bash
```

---

## 2. 话题通信 (Topic - Pub/Sub)

话题是单向的、基于发布/订阅模型的数据流。这里以 **C++** 为例实现一个简单的发布者和订阅者。

### 2.1 发布者 (Publisher) - C++
```cpp
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node {
public:
  MinimalPublisher() : Node("minimal_publisher"), count_(0) {
    // 创建发布者，指定话题名称 "topic" 和队列长度 10
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    // 创建定时器，每 500ms 触发一次 timer_callback
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback() {
    auto message = std_msgs::msg::String();
    message.data = "Hello ROS 2! Count: " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
```

### 2.2 订阅者 (Subscriber) - C++
```cpp
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalSubscriber : public rclcpp::Node {
public:
  MinimalSubscriber() : Node("minimal_subscriber") {
    // 创建订阅者，绑定回调函数
    subscription_ = this->create_subscription<std_msgs::msg::String>(
      "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, std::placeholders::_1));
  }

private:
  void topic_callback(const std_msgs::msg::String & msg) const {
    RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg.data.c_str());
  }
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}
```

---

## 3. 服务通信 (Service/Client)

服务是双向的、基于请求/响应（Request/Response）同步或异步模型。这里以 **Python** 为例，使用标准服务类型 `example_interfaces/srv/AddTwoInts`。

### 3.1 服务端 (Service Server) - Python
```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3.2 客户端 (Service Client) - Python
```python
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        # 等待服务上线
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        # 异步调用
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info('Result: %d' % response.sum)
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 4. Launch 文件启动机制

Launch 文件用于一次性配置并启动多个节点。ROS 2 推荐使用 Python 编写 Launch 文件。

```python
# 文件路径: my_py_pkg/launch/my_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_cpp_pkg',      # 功能包名
            executable='talker',       # 可执行文件/节点入口
            name='custom_talker',      # 重命名节点 (可选)
            output='screen',           # 日志输出到终端
            parameters=[
                {'my_param': 'value'}  # 传递参数
            ]
        ),
        Node(
            package='my_cpp_pkg',
            executable='listener',
            name='custom_listener',
            output='screen'
        )
    ])
```
*启动命令:* `ros2 launch my_py_pkg my_launch.py`

---

## 5. 常用 CLI 调试命令速查表

在终端中调试和排查 ROS 2 系统的利器：

### 节点操作 (Node)
* `ros2 node list` : 列出当前运行的所有节点
* `ros2 node info <node_name>` : 查看节点订阅、发布的话题及服务等详细信息

### 话题操作 (Topic)
* `ros2 topic list -t` : 列出所有话题及其消息类型
* `ros2 topic echo <topic_name>` : 实时打印话题内容
* `ros2 topic hz <topic_name>` : 查看话题的发布频率
* `ros2 topic pub <topic_name> <msg_type> "<data>"` : 手动向话题发布消息 (如：`ros2 topic pub /chatter std_msgs/msg/String "data: 'Hello'"`)

### 服务操作 (Service)
* `ros2 service list` : 列出所有活跃的服务
* `ros2 service type <service_name>` : 查看服务类型
* `ros2 service call <service_name> <srv_type> "<request_data>"` : 终端手动调用服务

### 接口与包 (Interface & Pkg)
* `ros2 interface show <type_name>` : 查看消息或服务类型的内部数据结构定义
* `ros2 pkg list` : 列出系统中安装的所有功能包