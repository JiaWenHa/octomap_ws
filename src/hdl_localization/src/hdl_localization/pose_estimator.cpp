#include <hdl_localization/pose_estimator.hpp>

#include <pcl/filters/voxel_grid.h>
#include <hdl_localization/pose_system.hpp>
#include <hdl_localization/odom_system.hpp>
#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

/**
 * @brief constructor  主要是进行一些初始化工作
 * @param registration        registration method
 * @param stamp               timestamp
 * @param pos                 initial position
 * @param quat                initial orientation
 * @param cool_time_duration  during "cool time", prediction is not performed
 */
PoseEstimator::PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const ros::Time& stamp, 
                            const Eigen::Vector3f& pos, const Eigen::Quaternionf& quat, double cool_time_duration)
    : init_stamp(stamp), registration(registration), cool_time_duration(cool_time_duration) {
  last_observation = Eigen::Matrix4f::Identity();
  /***
   * .block<p,q>(i,j)函数
   * p,q表示block的大小。
   * i,j表示从矩阵中的第几个元素开始向右向下开始算起
  */
  last_observation.block<3, 3>(0, 0) = quat.toRotationMatrix();
  last_observation.block<3, 1>(0, 3) = pos;

  //单位阵初始化，随后给过程噪声。
  process_noise = Eigen::MatrixXf::Identity(16, 16);
  //middleRows(行起始，行数)
  process_noise.middleRows(0, 3) *= 1.0;
  process_noise.middleRows(3, 3) *= 1.0;
  process_noise.middleRows(6, 4) *= 0.5;
  process_noise.middleRows(10, 3) *= 1e-6;
  process_noise.middleRows(13, 3) *= 1e-6;

  //测量噪声，随后给测量噪声
  Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
  measurement_noise.middleRows(0, 3) *= 0.01;
  measurement_noise.middleRows(3, 4) *= 0.001;

  //加权平均的位姿。
  Eigen::VectorXf mean(16);
  mean.middleRows(0, 3) = pos;
  mean.middleRows(3, 3).setZero(); //速度设置为0
  mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());
  mean.middleRows(10, 3).setZero(); //加速度计的误差暂时设置为0
  mean.middleRows(13, 3).setZero(); //陀螺仪的误差暂时设置为0

  //初始化协方差
  Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;

  //声明posesystem。
  PoseSystem system;
  //初始化ukf
  ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>(system, 16, 6, 7, process_noise, measurement_noise, mean, cov));
}

PoseEstimator::~PoseEstimator() {}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const ros::Time& stamp) {
  //当前与初始化的时间间隔小于设置的时间，或prev_stamp（上次更新时间）为0（未更新），或prev_stamp等于当前时间。则更新prev_stamp并跳出。
  if ((stamp - init_stamp).toSec() < cool_time_duration || prev_stamp.is_zero() || prev_stamp == stamp) {
    prev_stamp = stamp;
    return;
  }

  //正常处理，首先计算dt，更新prev_stamp。
  double dt = (stamp - prev_stamp).toSec();
  prev_stamp = stamp;

  //设置 ukf 的过程噪声
  ukf->setProcessNoiseCov(process_noise * dt);
  //设置 UKF 的处理间隔
  ukf->system.dt = dt;

  //利用ukf预测。
  ukf->predict();
}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const ros::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
  //当前与初始化的时间间隔小于设置的时间，或prev_stamp（上次更新时间）为0（未更新），或prev_stamp等于当前时间。则更新prev_stamp并跳出。
  if ((stamp - init_stamp).toSec() < cool_time_duration || prev_stamp.is_zero() || prev_stamp == stamp) {
    prev_stamp = stamp;
    return;
  }

  //正常处理，首先计算dt，更新prev_stamp。
  double dt = (stamp - prev_stamp).toSec();
  prev_stamp = stamp;

  //设置 ukf 的过程噪声 和 UKF的处理间隔
  ukf->setProcessNoiseCov(process_noise * dt);
  ukf->system.dt = dt;

  //获取控制量
  Eigen::VectorXf control(6);
  control.head<3>() = acc;
  control.tail<3>() = gyro;

  //利用ukf预测 机器人 下一时刻的状态。
  ukf->predict(control);
}

/**
 * @brief update the state of the odomety-based pose estimation
 */
void PoseEstimator::predict_odom(const Eigen::Matrix4f& odom_delta) {
  if(!odom_ukf) {
    Eigen::MatrixXf odom_process_noise = Eigen::MatrixXf::Identity(7, 7);
    Eigen::MatrixXf odom_measurement_noise = Eigen::MatrixXf::Identity(7, 7) * 1e-3;

    Eigen::VectorXf odom_mean(7);
    odom_mean.block<3, 1>(0, 0) = Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
    odom_mean.block<4, 1>(3, 0) = Eigen::Vector4f(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]);
    Eigen::MatrixXf odom_cov = Eigen::MatrixXf::Identity(7, 7) * 1e-2;

    OdomSystem odom_system;
    odom_ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, OdomSystem>(odom_system, 7, 7, 7, odom_process_noise, odom_measurement_noise, odom_mean, odom_cov));
  }

  // invert quaternion if the rotation axis is flipped
  Eigen::Quaternionf quat(odom_delta.block<3, 3>(0, 0));
  if(odom_quat().coeffs().dot(quat.coeffs()) < 0.0) {
    quat.coeffs() *= -1.0f;
  }

  Eigen::VectorXf control(7);
  control.middleRows(0, 3) = odom_delta.block<3, 1>(0, 3);
  control.middleRows(3, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());

  Eigen::MatrixXf process_noise = Eigen::MatrixXf::Identity(7, 7);
  process_noise.topLeftCorner(3, 3) = Eigen::Matrix3f::Identity() * odom_delta.block<3, 1>(0, 3).norm() + Eigen::Matrix3f::Identity() * 1e-3;
  process_noise.bottomRightCorner(4, 4) = Eigen::Matrix4f::Identity() * (1 - std::abs(quat.w())) + Eigen::Matrix4f::Identity() * 1e-3;

  odom_ukf->setProcessNoiseCov(process_noise);
  odom_ukf->predict(control);
}

/**
 * @brief correct
 * @param cloud   input cloud
 * @return cloud aligned to the globalmap
 */
pcl::PointCloud<PoseEstimator::PointT>::Ptr PoseEstimator::correct(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
  last_correction_stamp = stamp;

  //单位阵来初始化
  Eigen::Matrix4f no_guess = last_observation;
  Eigen::Matrix4f imu_guess;
  Eigen::Matrix4f odom_guess;
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

  if(!odom_ukf) {
    //matrix() 得到的是机器人当前的位姿的 齐次矩阵
    init_guess = imu_guess = matrix();
  } else {
    imu_guess = matrix();
    odom_guess = odom_matrix();

    Eigen::VectorXf imu_mean(7);
    Eigen::MatrixXf imu_cov = Eigen::MatrixXf::Identity(7, 7);
    imu_mean.block<3, 1>(0, 0) = ukf->mean.block<3, 1>(0, 0);
    imu_mean.block<4, 1>(3, 0) = ukf->mean.block<4, 1>(6, 0);

    imu_cov.block<3, 3>(0, 0) = ukf->cov.block<3, 3>(0, 0);
    imu_cov.block<3, 4>(0, 3) = ukf->cov.block<3, 4>(0, 6);
    imu_cov.block<4, 3>(3, 0) = ukf->cov.block<4, 3>(6, 0);
    imu_cov.block<4, 4>(3, 3) = ukf->cov.block<4, 4>(6, 6);

    Eigen::VectorXf odom_mean = odom_ukf->mean;
    Eigen::MatrixXf odom_cov = odom_ukf->cov;

    if (imu_mean.tail<4>().dot(odom_mean.tail<4>()) < 0.0) {
      odom_mean.tail<4>() *= -1.0;
    }

    Eigen::MatrixXf inv_imu_cov = imu_cov.inverse();
    Eigen::MatrixXf inv_odom_cov = odom_cov.inverse();

    Eigen::MatrixXf fused_cov = (inv_imu_cov + inv_odom_cov).inverse();
    Eigen::VectorXf fused_mean = fused_cov * inv_imu_cov * imu_mean + fused_cov * inv_odom_cov * odom_mean;

    init_guess.block<3, 1>(0, 3) = Eigen::Vector3f(fused_mean[0], fused_mean[1], fused_mean[2]);
    init_guess.block<3, 3>(0, 0) = Eigen::Quaternionf(fused_mean[3], fused_mean[4], fused_mean[5], fused_mean[6]).normalized().toRotationMatrix();
  }

  //点云的配准。ndt
  pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
  registration->setInputSource(cloud); //设置输入的点云
  /**
   * 调用配准算法，该算法输入为 UKF预测的机器人状态估计 并返回转换后的源(输入)作为输出。
   * 进行ndt配准,计算变换矩阵。 aligned: 存储input_cloud经过配准后的点云
   * (由于input_cloud被极大的降采样了,因此这个数据没什么用) 。此处 aligned 不能作为最终的源点云变换，因为对源点云进行了滤波处理
  */
  registration->align(*aligned, init_guess);

  //读取数据
  Eigen::Matrix4f trans = registration->getFinalTransformation(); //估计的变换矩阵
  Eigen::Vector3f p = trans.block<3, 1>(0, 3); //读出平移量
  Eigen::Quaternionf q(trans.block<3, 3>(0, 0)); //读出旋转量

  if(quat().coeffs().dot(q.coeffs()) < 0.0f) {
    q.coeffs() *= -1.0f;
  }

  //填充至观测矩阵observation --> 观测矩阵中保存的是进行点云配准后得到的位姿变换矩阵
  Eigen::VectorXf observation(7);
  observation.middleRows(0, 3) = p;
  observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());
  last_observation = trans; //保存一下的观测到的位姿变换矩阵

  //TODO:这句话用来干什么的？
  wo_pred_error = no_guess.inverse() * registration->getFinalTransformation();

  //使用ukf更新
  ukf->correct(observation);
  imu_pred_error = imu_guess.inverse() * registration->getFinalTransformation();

  //下面的语句只有当使用轮式里程计的时候才会用到
  if(odom_ukf) {
    if (observation.tail<4>().dot(odom_ukf->mean.tail<4>()) < 0.0) {
      odom_ukf->mean.tail<4>() *= -1.0;
    }

    odom_ukf->correct(observation);
    odom_pred_error = odom_guess.inverse() * registration->getFinalTransformation();
  }

  return aligned;
}

/* getters */
ros::Time PoseEstimator::last_correction_time() const {
  return last_correction_stamp;
}

// 从 UKF 的 mean 中获取 机器人的位置
// 函数后面加 const 表示函数不可以修改 class 的成员
Eigen::Vector3f PoseEstimator::pos() const {
  return Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
}

Eigen::Vector3f PoseEstimator::vel() const {
  return Eigen::Vector3f(ukf->mean[3], ukf->mean[4], ukf->mean[5]);
}

// 从 UKF 的 mean 中获取 机器人的姿态
Eigen::Quaternionf PoseEstimator::quat() const {
  return Eigen::Quaternionf(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]).normalized();
}

//matrix() 函数 用于获取 经过UKF滤波后机器人位姿的齐次矩阵
Eigen::Matrix4f PoseEstimator::matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = pos();
  return m;
}

Eigen::Vector3f PoseEstimator::odom_pos() const {
  return Eigen::Vector3f(odom_ukf->mean[0], odom_ukf->mean[1], odom_ukf->mean[2]);
}

Eigen::Quaternionf PoseEstimator::odom_quat() const {
  return Eigen::Quaternionf(odom_ukf->mean[3], odom_ukf->mean[4], odom_ukf->mean[5], odom_ukf->mean[6]).normalized();
}

Eigen::Matrix4f PoseEstimator::odom_matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = odom_quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = odom_pos();
  return m;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::wo_prediction_error() const {
  return wo_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::imu_prediction_error() const {
  return imu_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::odom_prediction_error() const {
  return odom_pred_error;
}
}