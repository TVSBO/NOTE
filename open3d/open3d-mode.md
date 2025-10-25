## 基础函数
1. 点云写入文件
   ```
    open3d.io.write_point_cloud(
    filename: os.PathLike, 
    文件路径

    pointcloud: open3d.geometry.PointCloud, 
    要写入的 PointCloud 对象。

    format: str = 'auto', 
    输出文件的格式。当未指定或设置为 auto 时，格式将从文件扩展名推断。

    write_ascii: bool = False, 
    如果为 True，则以 ASCII 格式输出，否则使用二进制格式。

    compressed: bool = False, 
    如果为 True，则以压缩格式写入。

    print_progress: bool = False
    如果为 True，在控制台中显示进度条。

    ) -> bool

   ```

2. 读取点云文件
   ```
    open3d.io.read_point_cloud(
    filename: os.PathLike, 
    文件路径
    
    format: str = 'auto',
    输入文件的格式。当未指定或设置为 auto 时，格式将从文件扩展名推断。

    remove_nan_points: bool = False, 
    如果为 True，则移除包含 NaN 值的点。

    remove_infinite_points: bool = False, 
    果为 True，则移除包含无限值的点。

    print_progress: bool = False
    如果为 True，在控制台中显示进度条。

    ) -> open3d.geometry.PointCloud

   ```
3. 可视化点云
```
    open3d.visualization.draw_geometries(
    geometry_list: list[open3d.geometry.Geometry], 
    要可视化的几何对象列表。

    window_name: str = 'Open3D', 
    可视化窗口的标题。

    width: int = 1920, 
    可视化窗口的宽度。

    height: int = 1080, 
    可视化窗口的高度。

    left: int = 50, 
    可视化窗口的左边距。

    top: int = 50, 
    可视化窗口的上边距。

    point_show_normal: bool = False, 
    如果为 True，则显示点的法线。

    mesh_show_wireframe: bool = False, 
    如果为 True，则显示网格的线框

    mesh_show_back_face: bool = False, 
    如果为 True，则显示网格三角形的背面

    lookat: numpy.ndarray[numpy.float64[3, 1]] | None = None, 
    机的 lookat 向量。

    up: numpy.ndarray[numpy.float64[3, 1]] | None = None, 
    相机的 up 向量。

    front: numpy.ndarray[numpy.float64[3, 1]] | None = None, 
    机的 front 向量。

    zoom: float | None = None
    相机的缩放
    ) -> None

```
4. 创建空点云
```
pcd = o3d.geometry.PointCloud()

```
| 属性 / 方法 | 含义 |
| --- | --- |
| `pcd.points` | 点坐标列表（Nx3） |
| `pcd.colors` | 点的颜色（Nx3） |
| `pcd.normals` | 点的法向量（Nx3） |
| `pcd.has_points()` | 判断是否有点数据 |
| `pcd.has_colors()` | 判断是否有颜色 |
| `pcd.has_normals()` | 判断是否有法向量 |
| `pcd.translate([x,y,z])` | 平移点云 |
| `pcd.scale(s, center)` | 缩放点云 |
| `pcd.rotate(R, center)` | 旋转点云 |
| `pcd.paint_uniform_color([r,g,b])` | 给所有点赋同一颜色 |
| `pcd.voxel_down_sample(voxel_size)` | 体素下采样（减少点数） |
| `pcd.estimate_normals()` | 估计法向量 |
| `pcd.normalize_normals()` | 归一化法向量 |
| `pcd.remove_statistical_outlier(nb_neighbors, std_ratio)` | 去除噪声点 |
5. 点云数据转换
```
o3d.utility.Vector3dVector(points)
把 NumPy 的二维数组（N×3）转换为 Open3D 专用的三维向量列表对象
```
6. 
