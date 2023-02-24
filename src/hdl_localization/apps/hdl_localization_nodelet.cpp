/***
 * hdl-localization 用的是ndt匹配加上 UKF 的数据融合。它的新颖之处在于后端数据融合是的UKF，
 * 与 EKF 相比，UKF更好地拟合非线性的概率分布，而不是强行进行线性化。
 * 
 * 在hdl中使用的ndt就像我们熟悉的 pcl 接口一样，只不过调用了多线程 omp 模式，它提供的是测量值。
 * imu 提供的是先验值，我们如果选择不用 imu 则先验值为原有的位姿。
*/

#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>

#include <std_srvs/Empty.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl/filters/voxel_grid.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <pclomp/gicp_omp.h> //MODIFY:自己添加的
#include <pcl/registration/gicp.h> //MODIFY:自己添加的

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/ScanMatchingStatus.h>
#include <hdl_global_localization/SetGlobalMap.h>
#include <hdl_global_localization/QueryGlobalLocalization.h>

namespace hdl_localization {

class HdlLocalizationNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() : tf_buffer(), tf_listener(tf_buffer) {
  }
  virtual ~HdlLocalizationNodelet() {
  }

  // ROS基础设施的所有初始化都必须放在这个函数中
  void onInit() override {
    // 创建3个句柄
    nh = getNodeHandle();//获取节点句柄，所有回调按顺序到达
    mt_nh = getMTNodeHandle();//使用多线程回调队列获取节点句柄，回调将分布在管理器的线程池中
    private_nh = getPrivateNodeHandle();//获取私有节点句柄，所有回调按顺序到达

    //初始化点云配准方法为NDT_OMP, 以前的版本有 GICP_OMP，这一版本没有，不知道为什么
    initialize_params();

    // 获取外部参数
    robot_odom_frame_id = private_nh.param<std::string>("robot_odom_frame_id", "robot_odom");
    // 这个默认的base_link, launch里覆盖了。实际上是velodyne。
    odom_child_frame_id = private_nh.param<std::string>("odom_child_frame_id", "base_link");
    // 是否使用imu
    use_imu = private_nh.param<bool>("use_imu", true);
    // imu是否倒置
    invert_acc = private_nh.param<bool>("invert_acc", false);
    invert_gyro = private_nh.param<bool>("invert_gyro", false);
    if (use_imu) { //如果使用imu，则定义订阅函数。
      NODELET_INFO("enable imu-based prediction");
      // "/gpsimu_driver/imu_data" 在launch 文件中被 remap 了
      imu_sub = mt_nh.subscribe("/gpsimu_driver/imu_data", 256, &HdlLocalizationNodelet::imu_callback, this);
    }

    //点云数据、全局地图、初始位姿的订阅。initialpose_sub只是用于rviz划点用的。
    //输入点云，输出经过 UKF 滤波 和 NDT 配准后的机器人位姿
    points_sub = mt_nh.subscribe("/velodyne_points", 5, &HdlLocalizationNodelet::points_callback, this);
    globalmap_sub = nh.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmap_callback, this); //获取全局地图
    initialpose_sub = nh.subscribe("/initialpose", 8, &HdlLocalizationNodelet::initialpose_callback, this);//用于在rviz中初始化位姿

    //发布里程计信息，以及对齐后的点云数据。
    pose_pub = nh.advertise<nav_msgs::Odometry>("/odom", 5, false); //初始化位姿话题发布对象发布的话题名为"/odom"
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 5, false); // "aligned_points" -> 当前点云配准到全局地图坐标系后的点云
    status_pub = nh.advertise<ScanMatchingStatus>("/status", 5, false);

    // global localization  --> 已经屏蔽了，当前版本未使用
    use_global_localization = private_nh.param<bool>("use_global_localization", true);
    if(use_global_localization) {
      NODELET_INFO_STREAM("wait for global localization services");
      ros::service::waitForService("/hdl_global_localization/set_global_map");
      ros::service::waitForService("/hdl_global_localization/query");

      set_global_map_service = nh.serviceClient<hdl_global_localization::SetGlobalMap>("/hdl_global_localization/set_global_map");
      query_global_localization_service = nh.serviceClient<hdl_global_localization::QueryGlobalLocalization>("/hdl_global_localization/query");

      relocalize_server = nh.advertiseService("/relocalize", &HdlLocalizationNodelet::relocalize, this);
    }
  }

private:
  pcl::Registration<PointT, PointT>::Ptr create_registration() const {
    // 点云配准方法参数：reg_method
    // NDT_OMP -- 多线程的ndt配准
    std::string reg_method = private_nh.param<std::string>("reg_method", "NDT_OMP");
    std::string ndt_neighbor_search_method = private_nh.param<std::string>("ndt_neighbor_search_method", "DIRECT7");
    double ndt_neighbor_search_radius = private_nh.param<double>("ndt_neighbor_search_radius", 2.0);
    double ndt_resolution = private_nh.param<double>("ndt_resolution", 1.0);

    if(reg_method == "NDT_OMP") {
      NODELET_INFO("NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);//为终止条件设置最小转换差异
      ndt->setResolution(ndt_resolution);//设置NDT网格结构的分辨率
      //ndt 搜索方法。默认是 DIRECT7，作者说效果不好可以尝试改为 DIRECT1
      /**
       * If you select pclomp::KDTREE, results will be completely same as the original pcl::NDT. 
       * We recommend to use pclomp::DIRECT7 which is faster and stable. 
       * If you need extremely fast registration, choose pclomp::DIRECT1, but it might be a bit unstable.
      */
      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          NODELET_INFO("search_method KDTREE is selected");
        } else if (ndt_neighbor_search_method == "GICP_OMP"){ //MODIFY:这个是我自己添加的,未成功
          NODELET_INFO("search_method GICP_OMP is selected");
          pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr gicp(new pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>());
          return gicp;
        }else if (ndt_neighbor_search_method == "GICP") { //MODIFY:这个是我自己添加的,已成功
          NODELET_INFO("search_method PCL GICP is selected");
          pcl::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr gicp(new pcl::GeneralizedIterativeClosestPoint<PointT, PointT>());
          return gicp;
        }else {
          NODELET_WARN("invalid search method was given");
          NODELET_WARN("default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    } else if(reg_method.find("NDT_CUDA") != std::string::npos) {
      NODELET_INFO("NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if(reg_method.find("D2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      } else if (reg_method.find("P2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
        NODELET_INFO_STREAM("search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      } else {
        NODELET_WARN("invalid search method was given");
      }
      return ndt;
    }

    NODELET_ERROR_STREAM("unknown registration method:" << reg_method);
    return nullptr;
  }

  void initialize_params() {
    // intialize scan matching method
    // 获取 downsample_resolution 的值
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    // 用于点云体素化
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    // 设置每个体素的大小, leaf_size分别为lx ly lz的长 宽 高
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    // 用于对点云进行降采样
    downsample_filter = voxelgrid;

    NODELET_INFO("create registration method for localization");
    registration = create_registration();

    // global localization 
    NODELET_INFO("create registration method for fallback during relocalization");
    relocalizing = false;
    delta_estimater.reset(new DeltaEstimater(create_registration()));

    // initialize pose estimator
    //设置起点用
    if(private_nh.param<bool>("specify_init_pose", true)) {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
        ros::Time::now(),
        Eigen::Vector3f(private_nh.param<double>("init_pos_x", 0.0), private_nh.param<double>("init_pos_y", 0.0), private_nh.param<double>("init_pos_z", 0.0)),
        Eigen::Quaternionf(private_nh.param<double>("init_ori_w", 1.0), private_nh.param<double>("init_ori_x", 0.0), private_nh.param<double>("init_ori_y", 0.0), private_nh.param<double>("init_ori_z", 0.0)),
        private_nh.param<double>("cool_time_duration", 0.5)
      ));
    }
  }

private:
  /***
   * 实际使用的回调函数就是 imu_callback(), points_callback(), globalmap_callback()
   * 分别订阅了 "/gpsimu_driver/imu_data", "/velodyne_points", "/globalmap"
  */
  /**
   * @brief callback for imu data.
   *        接收imu并扔到imu_data里去。会在HdlLocalizationNodelet::points_callback里用到。
   * @param imu_msg
   */
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex);
    imu_data.push_back(imu_msg);
  }

  /**
   * @brief callback for point cloud data.输入点云，输出位姿。
   *        
   * @param points_msg
   */
  void points_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    // 加锁
    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    //等待位姿估计器初始化
    if(!pose_estimator) {
      NODELET_ERROR("waiting for initial pose input!!");
      return;
    }

    //等待全局地图
    if(!globalmap) {
      NODELET_ERROR("globalmap has not been received!!");
      return;
    }

    //将ros消息转化为点云
    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud); 
    if(pcl_cloud->empty()) {
      NODELET_ERROR("cloud is empty!!");
      return;
    }

    //将点云转换到 odom_child_frame_id 坐标系 -> velodyne 坐标系。
    // transform pointcloud into odom_child_frame_id
    std::string tfError;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if(this->tf_buffer.canTransform(odom_child_frame_id, pcl_cloud->header.frame_id, stamp, ros::Duration(0.1), &tfError))
    {
        if(!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, this->tf_buffer)) {
            NODELET_ERROR("point cloud cannot be transformed into target frame!!");
            return;
        }
    }else
    {
        NODELET_ERROR(tfError.c_str());
        return;
    }

    //对点云降采样。这里用到的是同一文件下的函数。
    auto filtered = downsample(cloud);
    last_scan = filtered;

    if(relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    Eigen::Matrix4f before = pose_estimator->matrix();

    // predict 利用两帧点云之间的所有 imu 的读数
    if(!use_imu) {
      pose_estimator->predict(stamp);
    } else {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      //利用imu数据迭代。
      for(imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        // 若当前点云时间 小于 imu 雷达，则跳出。imu做预测，点云观测。
        if(stamp < (*imu_iter)->header.stamp) {
          break;
        }

        // 读取线加速度和角速度。判断是否倒置imu
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        double acc_sign = invert_acc ? -1.0 : 1.0;
        double gyro_sign = invert_gyro ? -1.0 : 1.0;
        //利用imu数据做位姿的预测。这里用了pose_estimator→predict,进一步调用了ukf进行估计。
        pose_estimator->predict((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      //删除用过的imu数据
      imu_data.erase(imu_data.begin(), imu_iter);
    }

    // odometry-based prediction --> 未使用
    ros::Time last_correction_time = pose_estimator->last_correction_time();
    if(private_nh.param<bool>("enable_robot_odometry_prediction", false) && !last_correction_time.isZero()) {
      geometry_msgs::TransformStamped odom_delta;
      if(tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0.1))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0));
      } else if(tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0));
      }

      if(odom_delta.header.stamp.isZero()) {
        NODELET_WARN_STREAM("failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
      } else {
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        pose_estimator->predict_odom(delta.cast<float>().matrix());
      }
    }

    // correct
    //用pose_estimator 来矫正点云。pcl库配准。获取到结果后利用ukf矫正位姿。
    // aligned 是 当前点云配准到点云地图坐标系后的点云
    auto aligned = pose_estimator->correct(stamp, filtered);
    //如果有订阅才发布
    if(aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      aligned_pub.publish(aligned);
    }

    if(status_pub.getNumSubscribers()) {
      publish_scan_matching_status(points_msg->header, aligned);
    }

    //发布里程计。时间戳为当前帧雷达时间，里程计位姿为ukf校正后位姿。同时也会发布从map到odom_child_frame_id的tf
    publish_odometry(points_msg->header.stamp, pose_estimator->matrix());
  }

  /**
   * @brief callback for globalmap input.
   *        完成了对全局地图的订阅以及从ros消息到点云地图的转化。并将其设置为点云配准过程中的目标点云
   *        目标点云可以理解为起始点云，后面采集到的所有的点云都与该目标点云进行配准
   * @param points_msg
   */
  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    //pcl::fromROSMsg(sensor_msgs::PointCloud2 in, pcl::PointCloud<PointType> out)
    //输入是 ROS 的点云类型，输出是 PCL的点云类型
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;

    // 发布出来的全局地图用作配准用的目标点云。这里就是globalmap_server_nodelet发出来的
    registration->setInputTarget(globalmap);

    //hdl_global_localization时使用，目前未使用，可以不管
    if(use_global_localization) {
      NODELET_INFO("set globalmap for global localization!");
      hdl_global_localization::SetGlobalMap srv;
      pcl::toROSMsg(*globalmap, srv.request.global_map);

      if(!set_global_map_service.call(srv)) {
        NODELET_INFO("failed to set global map");
      } else {
        NODELET_INFO("done");
      }
    }
  }

  /**
   * @brief perform global localization to relocalize the sensor position
   *        这个功能的用处是什么？使用这个功能后定位更加错乱
   * @param
   */
  bool relocalize(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res) {
    if(last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;

    if(!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
      relocalizing = false;
      NODELET_INFO_STREAM("global localization failed");
      return false;
    }

    const auto& result = srv.response.poses[0];

    NODELET_INFO_STREAM("--- Global localization result ---");
    NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
    NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
    pose = pose * delta_estimater->estimated_delta();

    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      ros::Time::now(),
      pose.translation(),
      Eigen::Quaternionf(pose.linear()),
      private_nh.param<double>("cool_time_duration", 0.5)));

    relocalizing = false;

    return true;
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    NODELET_INFO("initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator.reset(
          new hdl_localization::PoseEstimator(
            registration,
            ros::Time::now(),
            Eigen::Vector3f(p.x, p.y, p.z),
            Eigen::Quaternionf(q.w, q.x, q.y, q.z),
            private_nh.param<double>("cool_time_duration", 0.5))
    );
  }

  /**
   * @brief downsampling。当前帧点云数据降采样
   * @param cloud   input cloud
   * @return downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    //在函数initialize_params()里声明了。0.1，0.1，0.1网格
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  /**
   * @brief publish odometry。发布里程计的tf和msg。输入为当前帧点云时间戳与pose_estimator的结果矩阵。
   *        这里还用到了matrix2transform这个函数，用于做eigen矩阵到tf的转化（取值）。
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose) {
    // broadcast the transform over tf
    // robot_odom_frame_id -> odom, odom_child_frame_id -> velodyne
    // 测试 velodyne 坐标系 能不能坐标转换到 odom 坐标系，ros::Time(0)表示使用缓冲中最新的tf数据
    //TODO:下面这个 if 语句是用来干什么的？
    if(tf_buffer.canTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0))) {
      //Eigen::Isometry3d 是一个 4*4 的矩阵 ，eigenToTransform -> Convert an Eigen Isometry3d transform to the equivalent geometry_msgs message type.
      geometry_msgs::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
      map_wrt_frame.header.stamp = stamp;
      map_wrt_frame.header.frame_id = odom_child_frame_id;
      map_wrt_frame.child_frame_id = "map";

      // 查找 velodyne 坐标系 转到 odom 坐标系 的坐标变换
      geometry_msgs::TransformStamped frame_wrt_odom = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0), ros::Duration(0.1));
      // tf2::transformToEigen -> Convert a timestamped transform to the equivalent Eigen data type.
      Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

      geometry_msgs::TransformStamped map_wrt_odom;
      tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom); //把机器人的位姿转换到 以odom 为坐标系下

      tf2::Transform odom_wrt_map;
      tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
      odom_wrt_map = odom_wrt_map.inverse();

      geometry_msgs::TransformStamped odom_trans;
      odom_trans.transform = tf2::toMsg(odom_wrt_map);
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = robot_odom_frame_id;

      tf_broadcaster.sendTransform(odom_trans);
    } else {
      geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = odom_child_frame_id;
      tf_broadcaster.sendTransform(odom_trans);
    }

    // publish the transform。把机器人位姿信息 变成里程计信息 发布出去
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";

    tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    odom.child_frame_id = odom_child_frame_id;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    pose_pub.publish(odom);
  }

  /**
   * @brief publish scan matching status information
   */
  void publish_scan_matching_status(const std_msgs::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
    ScanMatchingStatus status;
    status.header = header;

    status.has_converged = registration->hasConverged();
    status.matching_error = registration->getFitnessScore();

    const double max_correspondence_dist = 0.5;

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(int i = 0; i < aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();
    status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;

    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);

    std::vector<double> errors(6, 0.0);

    if(pose_estimator->wo_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "without_pred";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
    }

    if(pose_estimator->imu_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
    }

    if(pose_estimator->odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "odom";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
    }

    status_pub.publish(status);
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  std::string robot_odom_frame_id;
  std::string odom_child_frame_id;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;
  ros::Subscriber imu_sub;
  ros::Subscriber points_sub;
  ros::Subscriber globalmap_sub;
  ros::Subscriber initialpose_sub;

  ros::Publisher pose_pub; // 位姿话题发布对象
  ros::Publisher aligned_pub;
  ros::Publisher status_pub;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
  tf2_ros::TransformBroadcaster tf_broadcaster;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::ImuConstPtr> imu_data;

  // globalmap and registration method
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

  // global localization
  bool use_global_localization;
  std::atomic_bool relocalizing;
  std::unique_ptr<DeltaEstimater> delta_estimater;

  pcl::PointCloud<PointT>::ConstPtr last_scan;
  ros::ServiceServer relocalize_server;
  ros::ServiceClient set_global_map_service;
  ros::ServiceClient query_global_localization_service;
};
}


PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
