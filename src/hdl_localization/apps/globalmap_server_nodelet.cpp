/***
 * 这个节点就是用来发布全局地图的
*/

#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>

namespace hdl_localization {

class GlobalmapServerNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  GlobalmapServerNodelet() {
  }
  virtual ~GlobalmapServerNodelet() {
  }

  // 这里是在重写了初始化函数。同时利用计时器触发回调函数。
  void onInit() override {
    //定义三个节点
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    //获得全局地图，并对全局地图 降采样
    initialize_params();

    // publish globalmap with "latched" publisher
    //发布话题名为 /globalmap 的全局地图
    globalmap_pub = nh.advertise<sensor_msgs::PointCloud2>("/globalmap", 5, true);
    //订阅话题 /map_request/pcd ，回调函数：GlobalmapServerNodelet::map_update_callback -> 发布 /globalmap
    // map_update_sub = nh.subscribe("/map_request/pcd", 10,              &GlobalmapServerNodelet::map_update_callback, this);

    //1秒钟触发一次，回调函数： GlobalmapServerNodelet::pub_once_cb -> 发布 /globalmap
    globalmap_pub_timer = nh.createWallTimer(ros::WallDuration(1.0), &GlobalmapServerNodelet::pub_once_cb, this, true, true);
  }

private:
  // 完成读取地图pcd文件的功能，并对该地图进行下采样，最终的globalmap是下采样的地图。
  void initialize_params() {
    // read globalmap from a pcd file
    std::string globalmap_pcd = private_nh.param<std::string>("globalmap_pcd", "");
    globalmap.reset(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
    globalmap->header.frame_id = "map";

    //这个实际上是没有到这里来的，初步想法是没有 utm 文件。类似于经纬度的坐标文件。
    std::ifstream utm_file(globalmap_pcd + ".utm");
    if (utm_file.is_open() && private_nh.param<bool>("convert_utm_to_local", true)) {
      double utm_easting;
      double utm_northing;
      double altitude;
      utm_file >> utm_easting >> utm_northing >> altitude;
      for(auto& pt : globalmap->points) {
        pt.getVector3fMap() -= Eigen::Vector3f(utm_easting, utm_northing, altitude);
      }
      ROS_INFO_STREAM("Global map offset by UTM reference coordinates (x = "
                      << utm_easting << ", y = " << utm_northing << ") and altitude (z = " << altitude << ")");
    }
    //endTODO

    // downsample globalmap
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);

    globalmap = filtered;
  }

  void pub_once_cb(const ros::WallTimerEvent& event) {
    globalmap_pub.publish(globalmap);
  }

  void map_update_callback(const std_msgs::String &msg){
    // msg is /map_request/pcd
    ROS_INFO_STREAM("Received map request, map path : "<< msg.data);
    std::string globalmap_pcd = msg.data;
    globalmap.reset(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
    globalmap->header.frame_id = "map";

    // downsample globalmap
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);

    globalmap = filtered;
    globalmap_pub.publish(globalmap);
  }

private:
  // ROS
  // 声明了三个句柄，1个发布，1个订阅，1个计时器，1个globalmap的变量
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  ros::Publisher globalmap_pub;
  ros::Subscriber map_update_sub;

  ros::WallTimer globalmap_pub_timer;
  pcl::PointCloud<PointT>::Ptr globalmap;
};

}


PLUGINLIB_EXPORT_CLASS(hdl_localization::GlobalmapServerNodelet, nodelet::Nodelet)
