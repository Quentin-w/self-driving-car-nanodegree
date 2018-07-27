#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {

	//Init the PID coefficient
	PID::Kp = Kp;
	PID::Ki = Ki;
	PID::Kd = Kd;

	//Init the PID errors
	p_error = 0.0;
	d_error = 0.0;
	i_error = 0.0;
}

void PID::UpdateError(double cte) {

	    //update the PID errors
	d_error = cte - p_error;
	p_error = cte;
	i_error += cte;


}

double PID::TotalError() {
	return (-Kp*d_error - Ki*i_error - Kd*d_error);
}



