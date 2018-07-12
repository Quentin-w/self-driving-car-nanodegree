#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
		const vector<VectorXd> &ground_truth) {
	/**
	 TODO:
	 * Calculate the RMSE here.
	 */

	//Lets create our rmse vector
	VectorXd rmse = VectorXd(4);
	//Lets initialize our rmse vector
	rmse << 0, 0, 0, 0;

	// Make sure our estimation vector is not empty and as the correct size
	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		std::cout << "[SIZE ERROR] Estimation or Ground truth size" << std::endl;
		return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i) {

		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array() * residual.array();
		rmse += residual;
	}

	//mean and squared root calculation
	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();

	//return rmse
	return rmse;
}
