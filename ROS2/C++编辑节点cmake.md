# 📝 纯 CMake 构建 ROS 2 C++ 节点核心笔记

在不使用完整的 ROS 2 功能包（不写 `package.xml`，不用 `colcon`）的情况下，我们依然可以通过最纯粹的 CMake 流程来编译一个 ROS 2 节点。这通常用于单文件测试或底层原理学习。

---

## 📂 1. 准备文件目录
在你当前的测试文件夹（例如 `~/chapt2`）中，只需要这两个核心文件：
1. **源码文件**：`ros2_cpp_node.cpp`
2. **构建图纸**：`CMakeLists.txt`

---

## 📄 2. 核心代码模板

### C++ 源码 (`ros2_cpp_node.cpp`)
保持节点定义的唯一性，最简结构如下：
```cpp
#include "rclcpp/rclcpp.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("my_node");
    RCLCPP_INFO(node->get_logger(), "节点启动成功！");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
```

### CMake 构建图纸 (`CMakeLists.txt`)
强烈建议使用 `ament_target_dependencies` 来挂载 ROS 2 依赖，这比传统的 `target_link_libraries` 更简洁且不易报错。
```cmake
cmake_minimum_required(VERSION 3.8)
project(ros2_cpp)

# 1. 寻找 ROS 2 核心库
find_package(rclcpp REQUIRED)

# 2. 指定用哪个 cpp 文件编译出哪个可执行程序
add_executable(ros2_cpp_node ros2_cpp_node.cpp)

# 3. 将 ROS 2 的头文件和库一键链接到你的程序上
ament_target_dependencies(ros2_cpp_node rclcpp)
```

---

## ⚙️ 3. 标准编译与运行流程 (Out-of-source Build)

为了不让 CMake 生成的临时文件弄乱你的源代码，强烈推荐使用**“外部构建”**（即新建一个 build 文件夹专门存放编译垃圾）。

请依次在终端执行以下命令：

**第一步：创建并进入专门的构建车间**
```bash
mkdir build
cd build
```

**第二步：让 CMake 读取上一层目录的图纸并生成 Makefile**
```bash
cmake ..
```

**第三步：开始真正的编译**
```bash
make
```

**第四步：运行生成的可执行文件**
```bash
./ros2_cpp_node
```

---

## 💡 4. 清理环境
如果你的编译环境乱了，或者想重新来过，直接把 `build` 文件夹整个删掉即可，源码完全不会受影响：
```bash
cd ..
rm -rf build
```


# 📝 ROS 2 CMakeLists.txt 依赖配置对比笔记

在 ROS 2 的 C++ 开发中，为节点配置依赖项（头文件与库文件）是 `CMakeLists.txt` 的核心任务。目前主要有两种写法。

## 1. 传统 CMake 底层写法 (繁琐但清晰)
传统方法需要手动将 `find_package` 找到的路径变量分别配置给目标文件。这种方式让你能清楚地看到编译器的底层行为。

```cmake
# 1. 寻找包，自动生成 _INCLUDE_DIRS 和 _LIBRARIES 变量
find_package(rclcpp REQUIRED)

# 2. 生成可执行文件
add_executable(ros2_cpp_node ros2_cpp_node.cpp)

# 3. 手动添加头文件搜索路径 (告诉编译器去哪看 #include)
target_include_directories(ros2_cpp_node PUBLIC ${rclcpp_INCLUDE_DIRS})

# 4. 手动链接二进制库文件 (告诉链接器去哪找具体实现)
target_link_libraries(ros2_cpp_node ${rclcpp_LIBRARIES})
```

## 2. ROS 2 现代推荐写法 (简洁高效)
使用 `ament_cmake` 提供的专用宏 `ament_target_dependencies`。这是 ROS 2 官方极力推荐的最佳实践。

```cmake
# 1. 寻找包
find_package(rclcpp REQUIRED)

# 2. 生成可执行文件
add_executable(ros2_cpp_node ros2_cpp_node.cpp)

# 3. 【一键挂载】自动处理头文件、库文件以及嵌套依赖
ament_target_dependencies(ros2_cpp_node rclcpp)
```

## 3. 核心对比总结

| 特性 | 传统 `target_...` 方法 | 现代 `ament_target_dependencies` |
| :--- | :--- | :--- |
| **代码量** | 至少两行，依赖越多代码越长。 | **仅需一行**，多个依赖直接并排写。 |
| **易错率** | 高。极易拼错长变量名。 | **极低**。直接写包名（如 `rclcpp`）即可。 |
| **依赖传递** | 需手动处理复杂的底层依赖关系。 | **自动解析**并链接底层隐藏依赖。 |
| **适用场景** | 引入非 ROS 的纯 C++ 第三方库（如 OpenCV）。 | 引入标准的 ROS 2 功能包。 |

**💡 最佳实践原则：**
只要是引用 ROS 2 内部的功能包，**永远优先使用 `ament_target_dependencies`**。

