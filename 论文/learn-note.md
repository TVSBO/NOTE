## sensors-19-00810-v2
1. Feature Extraction and Segmentation
   - Hough Transform, Random Sample Consensus(RANSAC), Principal Component Analysis (PCA), Fast Point Feature Histograms (FPFH), Region Growing and Connected Components, Graph-Cut, and Supervoxelization.
     - `RANSAC is another well-known algorithm that can be applied to detect pre-defined geometric primitives (Fischler and Bolles [17]). RANSAC commences with random sampling and determination of inliers and outliers for a target model. Constraints can be applied to the sampling procedure to improve the efficiency. Given the rules (e.g., number of inliers) for selecting the initial model from the previous step, the pre-defined primitives can be detected and further refined. Because RANSAC is generally robust to outliers and noise, there are many methods derived from its basic concept for feature extraction, segmentation, and modeling (Schnabel, et al. [18]).`随机抽样，对内点和外点进行检测，鲁棒性高。
    ```PCA is a data analysis technique that has been widely used for feature extraction from point
    cloud data (Jolliffe [19]). Different from the Hough Transform and RANSAC with one or multiple
    pre-defined models as input, PCA is a data-driven process to extract geometric information from an
    analysis of the local point distribution. Essentially, the results of PCA at a point are the eigenvalues
    and eigenvectors of the covariance matrix derived by this point and its neighbors. Further analysis can
    be completed to extract 1D, 2D, and 3D features from the point clouds by metrics derived from the
    eigenvalues and eigenvectors (Weinmann, et al. [20]).
    FPFH is proposed by (Rusu, et al. [21]) and has been widely used as a descriptor in various point
    cloud processing tasks (e.g., classification, registration). For each point, its k-nearest neighbors (kNN)
    within a given range are searched followed by an analysis of the variation of normals and distance
    between each pair of points within this neighborhood. Next, the neighbors and point pairs are further
    optimized to refine the descriptor of the geometric features of a local area. The outcome of Point
    Feature Histograms (PFH) at a point is a multi-dimensional histogram, which essentially describes
    and generalizes the local curvature at this point ```
2. Object Recognition
    1. 处理MSL数据的方法(1) rasterization, (2) 3D-point, and (3) scanline methods.