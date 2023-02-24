# pointcloud_publisher

此代码的作用：

1. 将点云文件内容数据发布到话题 `/pointcloud/output`
2. 使用octomap_server_node接收数据并生成Octomap
3. 使用rviz查看点云及对应的Octomap

## 使用方法

### 安装Octomap

An Efficient Probabilistic 3D Mapping Framework Based on Octrees

源码地址：https://github.com/OctoMap/octomap

这里 `$ROS_DISTRO` 代表你电脑上的ros版本，可以通过 `echo $ROS_DISTRO` 查看具体代表的版本，如果命令没有输出内容，可以根据不同的ubuntu系统，替换成字符串即可：

- Ubuntu14.04 -> indigo
- Ubuntu16.04 -> kinetic
- Ubuntu18.04 -> melodic

```bash
sudo apt-get install ros-$ROS_DISTRO-octomap-ros 
sudo apt-get install ros-$ROS_DISTRO-octomap-msgs
sudo apt-get install ros-$ROS_DISTRO-octomap-server
```

给rviz安装插件用于显示octomap

```bash
sudo apt-get install ros-$ROS_DISTRO-octomap-rviz-plugins
```



### 下载并编译ROS节点

```bash
cd catkin_ws/src
git clone https://gitee.com/tangyang/pointcloud_publisher.git
cd ..
catkin_make
```

### 启动节点并预览

```bash
roslaunch pointcloud_publisher demo.launch
```

### 使用其他点云

我们可以将其他.pcd文件拷贝到 `pointcloud_publisher/data` 目录下，然后修改 `launch/demo.launch` 中的path参数即可：

```xml
<param name="path" value="$(find pointcloud_publisher)/data/room_scan1.pcd" type="str" />
```

