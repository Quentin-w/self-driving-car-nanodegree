/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

//define gen , using the random number engine class that generates pseudo-random numbers
//that we can use for normal distribution.
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	if(is_initialized){
		return;
	}

	//Set the number of particles we want to use in the filter
	num_particles = 100;

	//creates a normal (Gaussian) distribution for the sensor noize
	normal_distribution<double> Init_noize_x(x, std[0]);
	//Create the noize normal distributions for y
	normal_distribution<double> Init_noize_y(y, std[1]);
	//Create the noize normal distributions for theta
	normal_distribution<double> Init_noize_theta(theta, std[2]);

	//Loop over the particle to initialize each of them
	for (int i = 0; i < num_particles; i++) {
		//create a particle object
		Particle p;
		//Create a unique particle id
		p.id = i;
		//assign particle init weight
		p.weight = 1.0;
		//assign particle x dstribution and noize
		p.x = Init_noize_x(gen);
		//assign particle y noize
		p.y = Init_noize_y(gen);
		//assign particle theta noize
		p.theta = Init_noize_theta(gen);

		//Now that the particle is created , add it to the list of particles
		particles.push_back(p);
	}
	//We've now initialize our particle filter
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
		double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//creates a normal (Gaussian) distribution for the sensor noize
	normal_distribution<double> noize_x(0, std_pos[0]);
	//Create the noize normal distributions for y
	normal_distribution<double> noize_y(0, std_pos[1]);
	//Create the noize normal distributions for theta
	normal_distribution<double> noize_theta(0, std_pos[2]);

	//loop over our particles
	for (int i = 0; i < num_particles; i++) {

		//Avoid 0 division
		if (fabs(yaw_rate) < 0.00001) {
			//equations for updating x, y when the yaw rate is ~ equal to zero
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}

		else {
			//equations for updating x, y and theta when the yaw rate is different to zero
			particles[i].x += velocity / yaw_rate
					* (sin(particles[i].theta + yaw_rate * delta_t)
							- sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate
					* (cos(particles[i].theta)
							- cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// add noise to x,y and thetha
		particles[i].x += noize_x(gen);
		particles[i].y += noize_y(gen);
		particles[i].theta += noize_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
		std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	//lets loop over all the observations
	for (unsigned int i = 0; i < observations.size(); i++) {

		//get the current observation
		LandmarkObs observation = observations[i];

		// set the min distance to the the max possible number
		double min_dist = numeric_limits<double>::max();

		// init id of landmark
		int map_id = -1;

		//loop over the predicitons
		for (unsigned int j = 0; j < predicted.size(); j++) {
			// get the current prediction
			LandmarkObs p = predicted[j];

			// get the distance between current/predicted landmarks
			double dist = sqrt((p.x - observation.x) * (p.x - observation.x) + (p.y - observation.y) * (p.y - observation.y));

			// find the nearest landmark from the current observed landmark
			if (dist < min_dist) {
				min_dist = dist;
				map_id = p.id;
			}
		}

		// set the observation's id to the nearest predicted landmark's id
		observations[i].id = map_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations,
		const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//Loop over each particles
	for (int i = 0; i < num_particles; i++) {

		// get particle x, y and theta
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// create a vector to store the predictions
		vector<LandmarkObs> predictions;

		//loop over each landmark
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			// get x, y and id from landmark
			int landmark_id = map_landmarks.landmark_list[j].id_i;
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;

			//not consider the predicton that are futher than our particle sensor range (thanks you to Jeremy for the computation efficiency trick)
			if (fabs(landmark_x - p_x) <= sensor_range
					&& fabs(landmark_y - p_y) <= sensor_range) {

				// save prediction in our vector
				predictions.push_back(LandmarkObs { landmark_id, landmark_x,
						landmark_y });
			}
		}

		//transform landmark from vehicle coordinates to map coordinates (in 2D)
		vector<LandmarkObs> transformed_observations;
		for (unsigned int j = 0; j < observations.size(); j++) {
			//transform for x
			double transformed_x = cos(p_theta) * observations[j].x
					- sin(p_theta) * observations[j].y + p_x;
			//transform for y
			double transformed_y = sin(p_theta) * observations[j].x
					+ cos(p_theta) * observations[j].y + p_y;

			//push the transformed coordinate
			transformed_observations.push_back(LandmarkObs { observations[j].id,
					transformed_x, transformed_y });
		}

		// Helper step
		//run data Association on our predictions and our transformed observations for each particle
		dataAssociation(predictions, transformed_observations);

		//Reset the particle weight
		particles[i].weight = 1.0;

		//lets now loop over the observations
		for (unsigned int j = 0; j < transformed_observations.size(); j++) {

			//define variables for storing the different values
			double observation_x;
			double observation_y;
			double prediction_x;
			double prediction_y;

			//get the observation values
			observation_x = transformed_observations[j].x;
			observation_y = transformed_observations[j].y;

			//get the id
			int associated_prediction = transformed_observations[j].id;

			// get the x,y coordinates for the prediction of the observation
			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == associated_prediction) {
					prediction_x = predictions[k].x;
					prediction_y = predictions[k].y;
				}
			}

			//create a gaussian distribution for x
			double gaussian_x = std_landmark[0];
			//create a gaussian distribution for y
			double gaussian_y = std_landmark[1];

			//calculate the observation weight using the gaussian distribution
			double observation_weight = (1
					/ (2 * M_PI * gaussian_x * gaussian_y))
					* exp(
							-(pow(prediction_x - observation_x, 2)
									/ (2 * pow(gaussian_x, 2))
									+ (pow(prediction_y - observation_y, 2)
											/ (2 * pow(gaussian_y, 2)))));

			// calculate the total observations weight
			particles[i].weight = particles[i].weight * observation_weight;

		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	 vector<Particle> new_particles;

	  // get all of the current weights
	  vector<double> weights;
	  for (int i = 0; i < num_particles; i++) {
	    weights.push_back(particles[i].weight);
	  }

	  // generate random starting index for resampling wheel
	  uniform_int_distribution<int> uniintdist(0, num_particles-1);
	  auto index = uniintdist(gen);

	  // get max weight
	  double max_weight = *max_element(weights.begin(), weights.end());

	  // uniform random distribution [0.0, max_weight)
	  uniform_real_distribution<double> unirealdist(0.0, max_weight);

	  double beta = 0.0;

	  // spin the resample wheel!
	  for (int i = 0; i < num_particles; i++) {
	    beta += unirealdist(gen) * 2.0;
	    while (beta > weights[index]) {
	      beta -= weights[index];
	      index = (index + 1) % num_particles;
	    }
	    new_particles.push_back(particles[index]);
	  }

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle,
		const std::vector<int>& associations,
		const std::vector<double>& sense_x,
		const std::vector<double>& sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
