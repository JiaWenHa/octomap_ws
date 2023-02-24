**功能简介：**

1) 使用 激光雷达+IMU 进行定位，定位算法：NDT + EKF
2) 使用octomap将三维点云地图转为八叉树地图和二维栅格地图，二维栅格地图用于路径规划。

使用方法：

注意在使用前要运行对应的 gazebo 仿真环境

```shell
roslaunch pointcloud_publisher demo.launch
```

