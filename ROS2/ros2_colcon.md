# colcon
1. colcon 是 "collective construction" 的缩写，是 ROS 2 开发中用于构建、测试和安装多个软件包的命令行工具。它是一个通用的构建工具，不仅支持 C++ (CMake)，也支持 Python (setuptools)。
2. 工作空间结构 (Workspace)
在使用 colcon 之前，通常需要建立一个工作空间（如 ros2_ws）。构建完成后，目录结构如下：
~~~
ros2_ws/
├── src/          # 源代码目录（用户手动创建，放入各种功能包）
├── build/        # 编译中间文件（colcon 自动创建）
├── install/      # 编译后的可执行文件、库和环境变量脚本（colcon 自动创建）
└── log/          # 构建过程中的日志记录（colcon 自动创建）
~~~
命令,功能说明
| 命令        | 功能说明 |
| -------- | ---- |
| `colcon build` | 编译工作空间中的所有功能包。 |
| `colcon build --packages-select <pkg_name>` | 只编译指定的一个或多个包。开发时最常用，可大幅节省时间。 |
| `colcon build --symlink-install` | 使用符号链接安装。对于 Python 包或更改资源文件（如 launch 文件、配置文件）时，无需重新编译即可生效。 |
| `colcon test` | 运行工作空间内所有包的测试用例。 |
| `colcon graph` | 以可视化形式显示包之间的依赖关系（需提前安装相关插件）。 |
