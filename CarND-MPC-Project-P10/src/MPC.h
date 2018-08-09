#ifndef MPC_H
#define MPC_H

using namespace std;

#include <vector>
#include "Eigen-3.3/Eigen/Core"

class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
 vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
};

#endif /* MPC_H */
