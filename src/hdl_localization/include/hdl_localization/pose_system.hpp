/*
本文件定义了完成了类PoseSystem的实现。主要是实现了ukf里 矩阵f（定义了系统）和h（观测）代码实现。这是要扔到ukf中去的。
系统状态量16位，分别是位姿（3）、速度（3）、四元数（4）、加速度偏差（3）、陀螺仪偏差（3）。另还有6位控制量，加速度（3）和陀螺仪（3）。
*/
#ifndef POSE_SYSTEM_HPP
#define POSE_SYSTEM_HPP

#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

/**
 * @brief Definition of system to be estimated by ukf
 * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
 */
class PoseSystem {
public:
  //定义变量类型方便后续使用
  typedef float T;
  typedef Eigen::Matrix<T, 3, 1> Vector3t;
  typedef Eigen::Matrix<T, 4, 4> Matrix4t;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Quaternion<T> Quaterniont;
public:
  PoseSystem() {
    dt = 0.01; //TODO：时间间隔为什么是0.01?
  }

  // system equation (without input)
  /**
   * 没有控制量的机器人的运动方程
  */
  VectorXt f(const VectorXt& state) const {
    // 下一时刻的状态：px, py, pz, vx, vy, vz, qx, qy, qz, qw, bax, bay, baz, bwx, bwy, bwz
    VectorXt next_state(16);//机器人下一时刻的状态 -->　一个16维的向量

    //state.middleRows(0, 3)是向量的第0,1,2元素
    Vector3t pt = state.middleRows(0, 3);//位置 px, py, pz
    Vector3t vt = state.middleRows(3, 3);//速度 vx, vy, vz
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    /**
     * 四元数归一化后，其逆等于其共轭。四元数不进行归一化还会导致 旋转矩阵R 非正交的情况，
     * 这样的话，R的转置就不等于R的逆了
    */
    qt.normalize();// 归一化四元数

    Vector3t acc_bias = state.middleRows(10, 3);//加速度偏差
    Vector3t gyro_bias = state.middleRows(13, 3);//陀螺仪偏差

    //下一时刻状态  
    // position.首先更新位置
    next_state.middleRows(0, 3) = pt + vt * dt;  //

    // velocity.
    next_state.middleRows(3, 3) = vt;

    // orientation.
    Quaterniont qt_ = qt;

    next_state.middleRows(6, 4) << qt_.w(), qt_.x(), qt_.y(), qt_.z();
    
    next_state.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
    next_state.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity

    return next_state;
  }

  // system equation
  /***
   * 有控制量的机器人的运动方程。根据机器人当前时刻的状态 和 控制量（imu的读数）计算下一时刻机器人的状态
   * 输入参数1：state --> 当前时刻的状态
   * 输入参数2：control --> 控制量
   * 输出：next_state --> 下一时刻的状态
  */
  VectorXt f(const VectorXt& state, const VectorXt& control) const {
    // 下一时刻的状态：px, py, pz, vx, vy, vz, qx, qy, qz, qw, bax, bay, baz, bwx, bwy, bwz
    VectorXt next_state(16);

    Vector3t pt = state.middleRows(0, 3);
    Vector3t vt = state.middleRows(3, 3);
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    qt.normalize();

    Vector3t acc_bias = state.middleRows(10, 3);//加速度偏差
    Vector3t gyro_bias = state.middleRows(13, 3);//陀螺仪偏差

    //控制量 control 是一个6维向量 ax, ay, az, wx, wy, wz
    Vector3t raw_acc = control.middleRows(0, 3);
    Vector3t raw_gyro = control.middleRows(3, 3);

    //下一时刻状态  
    // position。首先更新位置
    next_state.middleRows(0, 3) = pt + vt * dt;  //

    // velocity。更新速度
    Vector3t g(0.0f, 0.0f, 9.80665f); //重力加速度
    Vector3t acc_ = raw_acc - acc_bias;//获取实际的加速度值 = imu测得的加速度值 - 加速度偏差
    /**
     * 对于加速度，因为imu的加速度数据是在Body坐标系下表示的，所以要利用对应时刻的姿态将其转换到世界坐标系下，
     * 转换之前要减去bias，转化之后要减去重力加速度（世界坐标系下的重力加速度恒等于9.8）
    */
    Vector3t acc = qt * acc_;
    next_state.middleRows(3, 3) = vt + (acc - g) * dt;
    // next_state.middleRows(3, 3) = vt; // + (acc - g) * dt;		// acceleration didn't contribute to accuracy due to large noise

    // orientation.首先完成了陀螺仪的增量计算并归一化（直接转化为四元数形式），将其转换为下一时刻的四元数。
    Vector3t gyro = raw_gyro - gyro_bias;
    Quaterniont dq(1, gyro[0] * dt / 2, gyro[1] * dt / 2, gyro[2] * dt / 2);
    dq.normalize();
    Quaterniont qt_ = (qt * dq).normalized();
    next_state.middleRows(6, 4) << qt_.w(), qt_.x(), qt_.y(), qt_.z();
    //将当前控制量传入下一时刻的状态向量。认为加速度和角速度上的偏差不变
    next_state.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
    next_state.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity

    return next_state;
  }

  // observation equation
  /***
   * 输入参数：在更新阶段（correct）生成的带误差方差的（error variances）的扩展状态空间下的（extended state space）状态量，也就是ext_sigma_points
  */
  VectorXt h(const VectorXt& state) const {
    VectorXt observation(7);
    observation.middleRows(0, 3) = state.middleRows(0, 3);
    observation.middleRows(3, 4) = state.middleRows(6, 4).normalized();

    return observation;
  }

  double dt;
};

}

#endif // POSE_SYSTEM_HPP
