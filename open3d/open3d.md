## 点云基础操作
1. RGBD转PCD
```

    import open3d as o3d
    import matplotlib.pyplot as plt

    color_raw = o3d.io.read_image("img/color_file_1.jpeg")
    depth_raw = o3d.io.read_image("img/depth_file_1.png")
    # 创建一个rgbd图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    print(rgbd_image)

    # 使用matplotlib显示图像
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    # rgbd转pcd并显示
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
    )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("pointcloud/test.pcd", pcd, format='auto', write_ascii=False, compressed=False, print_progress=True)

```
2. PLY转PCD
```
   import open3d as o3d

    def convert_ply_to_pcd(ply_file, pcd_file):
        point_cloud = o3d.io.read_point_cloud(ply_file) # 读取PLY文件
        o3d.io.write_point_cloud(pcd_file, point_cloud) # 保存为PCD文件

    ply_file_path = "pointcloud/test_file_1.ply"
    pcd_file_path = "pointcloud/test_file_1.pcd"
    convert_ply_to_pcd(ply_file_path, pcd_file_path)
```

## KD_Tree
1. 相关函数
   - `o3d.geometry.KDTreeFlann(pcd)`:创建KD-Tree
   - `search_knn_vector_3d(search_pt, k)`：K近邻搜索
   - `search_radius_vector_3d(search_pt，radius)`：半径R近邻搜索
   - `search_hybrid_vector_3d(search_pt, radius, max_nn)`：混合邻域搜索，返回半径radius内不超过max_nn个近邻点
2. 原理
   1. 确定 split域 判断在k维中哪个维度的方差最大 选择最大值作为域值a
   2. node-data 根据a维上的数据点进行划分选取中间值作为node-data域位数据点，该节点的分割超平面就是通过（x,y）并垂直于：split=a轴的直线a=x；
   3. 确定左子空间和右子空间，通过上述