#include "MPC.h"
#include <math.h>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using CppAD::AD;

// 设置预测状态点的个数和两个相邻状态点之间的时间间隔
size_t N = 25;
double dt = 0.05;

// 设置汽车头到车辆重心之间的距离
const double Lf = 2.67;

// 参考速度，为了避免车辆在行驶中停止
double ref_v = 40;

// solver使用的是一个向量存储所有的状态值，所以需要确定每种状态在向量中的开始位置
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

class FG_eval {
 public:

  // 参考路径方程的系数
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  // 该函数的目的是定义约束，fg向量包含的是总损失和约束，vars向量包含的是状态值和驱动器的输入
  void operator()(ADvector& fg, const ADvector& vars) {
	// 任何cost都会加到fg[0]
	fg[0] = 0;
    // 使车辆轨迹和参考路径的误差最小，且使车辆的速度尽量接近参考速度
	for (int t = 0; t < N; t++) {
		fg[0] += CppAD::pow(vars[cte_start + t], 2);
		fg[0] += CppAD::pow(vars[epsi_start + t], 2);
		fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
	}
	// 使车辆行驶更平稳，尽量减少每一次驱动器的输入大小
	// 注意：驱动器的输入是N-1个.最后一个点没有驱动器输入
	for (int t = 0; t < N - 1; t++) {
		fg[0] += CppAD::pow(vars[delta_start + t], 2);
		fg[0] += CppAD::pow(vars[a_start + t], 2);
	}
	// 为了使车辆运动更平滑，尽量减少相邻两次驱动器输入的差距
	// 注意：这个地方是N-2个
	for (int t = 0; t < N - 2; t++) {
		fg[0] += CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
		fg[0] += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
	}

    // 设置fg的初始值为状态的初始值，这个地方为初始条件约束
	fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // 因为t=0初始条件约束已经有了，计算其他约束条件
    for (int t = 1; t < N; t++) {
      // t+1时刻的状态
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1 = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      // t时刻的状态
      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0 = vars[v_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];
      // t时刻的驱动器输入
      AD<double> delta0 = vars[delta_start + t - 1];
      AD<double> a0 = vars[a_start + t - 1];

      // t时刻参考路径的距离和角度值
      AD<double> f0 = coeffs[0] + coeffs[1] * x0;
      AD<double> psides0 = CppAD::atan(coeffs[1]);

      // 根据如上的状态值计算约束条件
      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t] = epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
    }
  }
};

// MPC class definition
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd x0, Eigen::VectorXd coeffs) {
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  double x = x0[0];
  double y = x0[1];
  double psi = x0[2];
  double v = x0[3];
  double cte = x0[4];
  double epsi = x0[5];

  // 独立状态的个数，注意：驱动器的输入（N - 1）* 2
  size_t n_vars = N * 6 + (N - 1) * 2;
  // 约束条件的个数
  size_t n_constraints = N * 6;

  // 除了初始值，初始化每一个状态为0
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  // 设置每一个状态变量的最大和最小值
  // 【1】设置非驱动输入的最大和最小值
  // 【2】设置方向盘转动角度范围-25—25度
  // 【3】加速度的范围-1—1
  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  for (int i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }
  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }
  for (int i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // 设置约束条件的的最大和最小值，除了初始状态其他约束都为0
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // Object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  // options
  std::string options;
  options += "Integer print_level  0\n";
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // 计算这个问题
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  //
  // Check some of the solution values
  //
  bool ok = true;
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  auto cost = solution.obj_value;
  // 返回计算出的下一时刻预测出的路径的点状态
  std::cout << "Cost " << cost << std::endl;
  return {solution.x[x_start + 1],   solution.x[y_start + 1],
          solution.x[psi_start + 1], solution.x[v_start + 1],
          solution.x[cte_start + 1], solution.x[epsi_start + 1],
          solution.x[delta_start],   solution.x[a_start]};
}

// Helper functions to fit and evaluate polynomials.

// 根据系数和输入值确定多项式的输出值
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// 使用Qr分解计算多项式的系数
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  MPC mpc;
  int iters = 500;// 迭代50次

  // 生成两个点
  Eigen::VectorXd ptsx(2);
  Eigen::VectorXd ptsy(2);
  ptsx << -100, 100;
  ptsy << -1, -1;

  // 根据如上的点，求得方程系数，作为参考路线
  auto coeffs = polyfit(ptsx, ptsy, 1);

  // 初始状态
  double x = -1;
  double y = 10;
  double psi = 0;
  double v = 10;
  double cte = polyeval(coeffs, x) - y;
  double epsi = psi - atan(coeffs[1]);
  Eigen::VectorXd state(6);
  state << x, y, psi, v, cte, epsi;

  // 记录每一次拟合的状态
  std::vector<double> x_vals = {state[0]};
  std::vector<double> y_vals = {state[1]};
  std::vector<double> psi_vals = {state[2]};
  std::vector<double> v_vals = {state[3]};
  std::vector<double> cte_vals = {state[4]};
  std::vector<double> epsi_vals = {state[5]};
  std::vector<double> delta_vals = {};
  std::vector<double> a_vals = {};

  for (size_t i = 0; i < iters; i++) {
    std::cout << "Iteration " << i << std::endl;
    // 计算出状态值
    auto vars = mpc.Solve(state, coeffs);

    x_vals.push_back(vars[0]);
    y_vals.push_back(vars[1]);
    psi_vals.push_back(vars[2]);
    v_vals.push_back(vars[3]);
    cte_vals.push_back(vars[4]);
    epsi_vals.push_back(vars[5]);
    delta_vals.push_back(vars[6]);
    a_vals.push_back(vars[7]);

    // 把计算出的状态作为初始状态传给下一次拟合
    state << vars[0], vars[1], vars[2], vars[3], vars[4], vars[5];

    // 输出当前拟合的状态值
    std::cout << "x = " << vars[0] << std::endl;
    std::cout << "y = " << vars[1] << std::endl;
    std::cout << "psi = " << vars[2] << std::endl;
    std::cout << "v = " << vars[3] << std::endl;
    std::cout << "cte = " << vars[4] << std::endl;
    std::cout << "epsi = " << vars[5] << std::endl;
    std::cout << "delta = " << vars[6] << std::endl;
    std::cout << "a = " << vars[7] << std::endl;
    std::cout << std::endl;
  }

  // 展示结果
  plt::subplot(3, 1, 1);
  plt::title("CTE");
  plt::plot(cte_vals);
  plt::subplot(3, 1, 2);
  plt::title("Delta (Radians)");
  plt::plot(delta_vals);
  plt::subplot(3, 1, 3);
  plt::title("Velocity");
  plt::plot(v_vals);

  plt::show();
}
