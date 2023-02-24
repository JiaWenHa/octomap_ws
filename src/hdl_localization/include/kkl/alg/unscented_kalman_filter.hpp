/**
 * 本文件中主要的函数也就构造函数、预测、矫正、计算sigma点、使协方差矩阵正有限（不太清楚）五个。
*/
/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>

namespace kkl {
  namespace alg {

/**
 * @brief Unscented Kalman Filter class，初始化 UKF 的权重
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
class UnscentedKalmanFilterX {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
public:
  /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param input_dim            input vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param process_noise        process noise covariance (state_dim x state_dim)
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)
   * @param mean                 initial mean
   * @param cov                  initial covariance
   */
  /**
  * 首先，构造函数。可以看到输入了一系列包括待估计系统、状态向量维度、输入维度、观测维度、两个噪声、参数等等。完成了初始化操作。
  */
  UnscentedKalmanFilterX(const System& system, int state_dim, int input_dim, int measurement_dim, const MatrixXt& process_noise,
                         const MatrixXt& measurement_noise, const VectorXt& mean, const MatrixXt& cov)
    : state_dim(state_dim),
    input_dim(input_dim),
    measurement_dim(measurement_dim),
    N(state_dim),
    M(input_dim),
    K(measurement_dim),
    S(2 * state_dim + 1),
    mean(mean),
    cov(cov),
    system(system),
    process_noise(process_noise),
    measurement_noise(measurement_noise),
    lambda(1),
    normal_dist(0.0, 1.0)
  {
    //设置长度。
    weights.resize(S, 1);
    sigma_points.resize(S, N);
    ext_weights.resize(2 * (N + K) + 1, 1);
    ext_sigma_points.resize(2 * (N + K) + 1, N + K);
    expected_measurements.resize(2 * (N + K) + 1, K);

    // initialize weights for unscented filter
    /***
     * 在计算到所有的 sigma 点之后，用这些点重新拟合一个正态分布，在这个过程中引入了权重 weights, 它的初始值在构造函数中
    */
    weights[0] = lambda / (N + lambda);
    for (int i = 1; i < 2 * N + 1; i++) {
      weights[i] = 1 / (2 * (N + lambda));
    }

    // weights for extended state space which includes error variances
    ext_weights[0] = lambda / (N + K + lambda);
    for (int i = 1; i < 2 * (N + K) + 1; i++) {
      ext_weights[i] = 1 / (2 * (N + K + lambda));
    }
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  // 通过pose_estimator->predict调用。
  void predict() {
    // calculate sigma points.ukf的sigma点
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);
    //sigma_points更新。用在posesystem中定义的f函数来进行。
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i));
    }

    /*----至此，sigma_points里存储的就是当前时刻的由ukf输出的系统状态。-----*/

    //过程噪声，即ukf中的矩阵R
    const auto& R = process_noise;

    // unscented transform.定义当前的平均状态和协方差矩阵，并设置为0矩阵。
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    //加权平均，预测状态
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    //根据状态预测协方差。
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    //附加过程噪声R，在pose_estimator中给出初值
    cov_pred += R;

    //更新mean和cov
    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief predict, 带真实 imu 数据的预测，在每帧 imu 传入后都进行一次预测更新，在观测矫正之前我们得到了 sigma 点拟合的状态分布，而非状态转移矩阵。
   *        这个函数会更新 均值 和 协方差 为预测的均值和协方差。这里的均值和协方差 描述的是 预测的下一时刻机器人状态的高斯分布
   * @param control  input vector
   */
  void predict(const VectorXt& control) {
    // calculate sigma points，ukf的sigma点
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);//根据均值，协方差计算 sigma 点，并保存在  sigma_points 中
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i), control);
    }

    const auto& R = process_noise;

    // unscented transform
    //定义预测后的 均值 和 协方差，并初始化为 0
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());
    mean_pred.setZero();
    cov_pred.setZero();

    // 计算预测的均值 和 协方差
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += R;

    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief correct。用 NDT 配准后的位姿变换矩阵 来修正 机器人的状态
   * @param measurement  measurement vector
   */
  void correct(const VectorXt& measurement) {
    // create extended state space which includes error variances
    //N-状态方程维度。K-观测维度
    //状态扩增，即先更新预测值的协方差矩阵，将上一部分用sigma点拟合的协方差加上测量噪声
    VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);
    //左上角N行1列 --> 状态值
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);
    //左上角N行N列 --> 状态的协方差
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);
    //右下角K行K列。初始化为在pose_estimator输入的噪声。位置噪声0.01，四元数0.001
    ext_cov_pred.bottomRightCorner(K, K) = measurement_noise;
    /*---------------- 经过以上操作，现在扩展状态变量前N项为mean，扩展协方差左上角为N*N的cov，右下角为K*K的观测噪声--------------*/

    //验证并计算
    //拟合状态扩增后的均值与协方差，也就是再算一遍sigma点
    ensurePositiveFinite(ext_cov_pred);
    //利用扩展状态空间的参数计算sigma点。加入观测后的。
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points);

    // unscented transform
    // UT 变换，即拟合测量的均值和协方差
    // 这里使用了 ukf 的h 函数来计算观测。
    // ext_sigma_points、expected_measurements是（2 * (N + K) + 1, K)的矩阵
    //取左上角前N个量，加上右下角K个量。
    expected_measurements.setZero();
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurements.row(i) = system.h(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1));
      expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(K, 1));
    }

    //加权平均。同predict函数相似。
    //虽然叫 expected ，但这是7维的测量值，也就是用 sigma 点拟合的测量分布
    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
    }

    //测量的协方差矩阵
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance
    //转换方差。用于计算sigama，进而计算卡尔曼增益
    MatrixXt sigma = MatrixXt::Zero(N + K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred);
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }

    kalman_gain = sigma * expected_measurement_cov.inverse();
    const auto& K = kalman_gain;

    //更新观测后的真值估计。
    VectorXt ext_mean = ext_mean_pred + K * (measurement - expected_measurement_mean);
    MatrixXt ext_cov = ext_cov_pred - K * expected_measurement_cov * K.transpose();

    mean = ext_mean.topLeftCorner(N, 1);
    cov = ext_cov.topLeftCorner(N, N);
  }

  /*			getter			*/
  const VectorXt& getMean() const { return mean; }
  const MatrixXt& getCov() const { return cov; }
  const MatrixXt& getSigmaPoints() const { return sigma_points; }

  System& getSystem() { return system; }
  const System& getSystem() const { return system; }
  const MatrixXt& getProcessNoiseCov() const { return process_noise; }
  const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise; }

  const MatrixXt& getKalmanGain() const { return kalman_gain; }

  /*			setter			*/
  UnscentedKalmanFilterX& setMean(const VectorXt& m) { mean = m;			return *this; }
  UnscentedKalmanFilterX& setCov(const MatrixXt& s) { cov = s;			return *this; }

  UnscentedKalmanFilterX& setProcessNoiseCov(const MatrixXt& p) { process_noise = p;			return *this; }
  UnscentedKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& m) { measurement_noise = m;	return *this; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int state_dim;
  const int input_dim;
  const int measurement_dim;

  const int N;
  const int M;
  const int K;
  const int S;

public:
  VectorXt mean;
  MatrixXt cov;

  System system;
  MatrixXt process_noise;		//
  MatrixXt measurement_noise;	//

  T lambda;
  VectorXt weights;

  MatrixXt sigma_points;

  VectorXt ext_weights;
  MatrixXt ext_sigma_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points  calculated sigma points
   */
  /**
   * 更新预测。control 是传入的一帧imu数据，我们姑且看作是控制量，预测时首先判断协方差矩阵是否是方阵并且是否正定，
   * 因为我们在求 sigma 点的过程中要求协方差矩阵的逆，因此提取出它们的特征值和特征向量并再次相乘。接下来就是求解sigma点了
   * 
   * 无迹变换是计算一系列点（关键点，sigma points），然后通过非线性函数进行变换，通过变换结果和对应的权重来计算高斯分布。
   * 
   * 通过mean和cov计算sigma点。思路是将cov做Cholesky分解，用下三角矩阵L对mean做处理。得到一系列sigma_points.
  */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points) {
    const int n = mean.size();
    assert(cov.rows() == n && cov.cols() == n);

    /**
     * Cholesky 分解是把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解。Eigen的LLT分解实现了Cholesky 分解。
    */
    //llt分解。求Cholesky分解A=LL^*=U^*U。L是下三角矩阵
    Eigen::LLT<MatrixXt> llt;
    llt.compute((n + lambda) * cov);
    MatrixXt l = llt.matrixL();

    //mean是列向量。这里会自动转置处理
    sigma_points.row(0) = mean;
    for (int i = 0; i < n; i++) {
      sigma_points.row(1 + i * 2) = mean + l.col(i); //奇数1357
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);//偶数2468
    }
  }

  /**
   * @brief make covariance matrix positive finite。保证协方差的正有限。未实际应用。
   * @param cov  covariance matrix
   */
  void ensurePositiveFinite(MatrixXt& cov) {
    return;
    //就到这里了，在上面就return掉了。
    const double eps = 1e-9;

    Eigen::EigenSolver<MatrixXt> solver(cov);
    MatrixXt D = solver.pseudoEigenvalueMatrix();//特征值
    MatrixXt V = solver.pseudoEigenvectors();//特征向量
    for (int i = 0; i < D.rows(); i++) {
      if (D(i, i) < eps) {
        D(i, i) = eps;
      }
    }

    cov = V * D * V.inverse();
  }

public:
  MatrixXt kalman_gain;

  std::mt19937 mt;
  std::normal_distribution<T> normal_dist;
};

  }
}


#endif
