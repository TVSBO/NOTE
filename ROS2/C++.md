# C++ 命令行参数输出核心笔记
## 一、核心元素作用
| 符号/变量       | 作用说明                                                                 |
|-----------------|--------------------------------------------------------------------------|
| `argc`          | main函数第一个参数（int型），命令行参数总数（包含程序名本身）             |
| `std::cout`     | 需包含`<iostream>`头文件，标准输出流对象，用于向控制台输出数据            |
| `<<`            | 流插入运算符，将右侧数据（字符串/数字/变量）插入输出流，支持链式调用      |
| `std::endl`     | 输出操纵符，实现换行+刷新输出缓冲区（等价于`\n + flush`）                |

## 二、基础用法示例
```cpp
#include <iostream>
//std::cout 是 C++ 标准库 <iostream> 中定义的输出流对象
// main函数标准形式（支持命令行参数）
int main(int argc, char* argv[]) {
    // 输出参数数量
    std::cout << "参数数量" << argc << std::endl;
    return 0;
}
//例子
std::cout << "姓名：" << "张三" << " 年龄：" << 20;
// 输出结果：姓名：张三 年龄：20
```

# 📝 C++ 命令行参数

## 1. 安全的代码模板
处理参数时，**必须先判断 `argc` 的数量**，再读取 `argv`，这样程序才不会崩溃。

```cpp
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    std::cout << "参数数量: " << argc << std::endl;
    std::cout << "程序名字: " << argv[0] << std::endl;

    // 安全锁：只有当参数数量大于 1 时，才去读取第二个参数
    if (argc > 1) 
    {
        std::string arg1 = argv[1];
        if (arg1 == "--help")
        {
            std::cout << "欢迎来到程序帮助中心！" << std::endl;
        }
    }
    else
    {
        std::cout << "提示: 你可以尝试加上 --help 运行我" << std::endl;
    }

    return 0;
}
```

## 2. 编译与运行流程 (使用 g++)
因为这段代码没有用到 ROS 2 的库，最快的方法是直接用 `g++` 编译。

**第一步：编译代码**
假设你的文件叫 `demo.cpp`，把它编译成名为 `demo` 的程序：
```bash
g++ demo.cpp -o demo
```

**第二步：测试运行**

* **场景 A：不加参数**
```bash
./demo
```
> 输出：
> 参数数量: 1
> 提示: 你可以尝试加上 --help 运行我

* **场景 B：加上参数**
```bash
./demo --help
```
> 输出：
> 参数数量: 2
> 欢迎来到程序帮助中心！