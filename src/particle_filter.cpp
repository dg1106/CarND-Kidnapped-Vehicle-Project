/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

#define EPS (0.00001)

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  
  if (is_initialized) {
    return;
  }
  
  num_particles = 50;  // TODO: Set the number of particles

  Particle new_particle;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  
  // Creating normal distributions
  std::normal_distribution<double> dist_x(x, std_x);
  std::normal_distribution<double> dist_y(y, std_y);
  std::normal_distribution<double> dist_theta(theta, std_theta);
  
  particles.resize(num_particles);
  
  // Generate particles with normal distribution with mean on GPS values.
  int cnt = 0;
  
  for (auto& p: particles) {
    p.id = cnt + 1;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0; 
    
    cnt++;
    //std::cout << "it's ok " << p.id << std::endl;
  }
  
  // This filter is now initialized
  is_initialized = true;
  
  //std::cout << "Particle init done" << std::endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  
  // Extracting standard deviations
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
  
  // Creating normal distributions
  std::normal_distribution<double> dist_x(0, std_x);
  std::normal_distribution<double> dist_y(0, std_y);
  std::normal_distribution<double> dist_theta(0, std_theta);
  
  // Calculate new state
  for (int i = 0; i < num_particles; i++){
   
    double theta = particles[i].theta;
    
    if (fabs(yaw_rate < EPS)) {
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta); 
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    
    // Adding noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
 
  //std::cout << "Prediction step done" << std::endl;
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  //std::cout << "map obseravtions : " << observations.size() << std::endl;
  //std::cout << "valid landmarks : " << predicted.size() << std::endl;
  
  for (auto& obs: observations) {
    // Initialize min distance as a big amount.
    double minD = std::numeric_limits<float>::max();
    
    for (const auto& pred: predicted) {
     double distance = dist(obs.x, obs.y, pred.x, pred.y);
      if (minD > distance) {
        minD = distance;
        obs.id = pred.id;
      }
    }
    //std::cout << "catch obs with " << minD << "[m]" << std::endl;
  }
  
  //std::cout << "Association step done" << std::endl;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  for (auto& p: particles) {
    std::cout << "ID: " << p.id << std::endl;
    std::cout << "X: " << p.x << std::endl;
    std::cout << "Y: " << p.y << std::endl;
    p.weight = 1.0;
    
    // step 1. Find landmarks in range of particles
    vector<LandmarkObs> inRangeLandmarks;
    
    double particle_x = p.x;
    double particle_y = p.y;
    //double particle_theta = p.theta;
    
    for (const auto& lm: map_landmarks.landmark_list){
      double sqrt_snsr_range = dist(particle_x, particle_y, lm.x_f, lm.y_f);
      if (sqrt_snsr_range < sensor_range) {
        // if the landmark is within the sensor range, save it to predictions
        inRangeLandmarks.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }
    
    // step 2. Transform observation coordinates from vehicle to map
    vector<LandmarkObs> mappedObservations;
    double cos_theta = cos(p.theta);
    double sin_theta = sin(p.theta);
    
    for (const auto& obs: observations){
      LandmarkObs tmp;
      tmp.x = obs.x * cos_theta - obs.y * sin_theta + p.x;
      tmp.y = obs.x * sin_theta + obs.y * cos_theta + p.y;
      
      mappedObservations.push_back(tmp);
    }
    
    // step 3. find landmark index for each observation
    dataAssociation(inRangeLandmarks, mappedObservations);
    
    // step 4. compute the particle's weight:
    for (const auto& obs_m: mappedObservations){
      int target_m = obs_m.id - 1;
      //std::cout << "target map id : " << target_m << std::endl;
      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(target_m);
      
      //std::cout << "map.x : " << landmark.x_f << std::endl;
      //std::cout << "map.y : " << landmark.y_f << std::endl;
      
      // define inputs
      double sig_x, sig_y, x_obs, y_obs, mu_x, mu_y;
      double weight;
      
      sig_x = std_landmark[0];
      sig_y = std_landmark[1];
      x_obs = obs_m.x;
      y_obs = obs_m.y;
      mu_x  = landmark.x_f;
      mu_y  = landmark.y_f;
      
      // Calculate Obs weight
      weight = multiv_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
      //std::cout << "Weight: " << weight << std::endl;
      
      p.weight *= weight;
    }
    
    weights.push_back(p.weight);
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
/*
  // Get weights and max weight
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  // create resampled particles
  vector<Particle> resampled_particles;
  resampled_particles.resize(num_particles);

  // resample the particles according to weights
  for(int i=0; i<num_particles; i++){
    int idx = dist(gen);
    resampled_particles[i] = particles[idx];
  }

  // assign the resampled_particles to the previous particles
  particles = resampled_particles;

  // clear the weight vector for the next round
  weights.clear();
*/
  
  std::default_random_engine gen;
  
  vector<double> weights;
  double maxWeight = std::numeric_limits<double>::min();
  
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    
    if (particles[i].weight > maxWeight) {
      maxWeight = particles[i].weight;
    }
  }
  
  // Creating distributions
  std::uniform_real_distribution<double> distDouble(0.0, maxWeight);
  std::uniform_int_distribution<int> distInt(0, num_particles - 1);
  
  // Generating index
  int index = distInt(gen);
  
  double beta = 0.0;
  
  // the wheel
  vector<Particle> resampledParticles;
  //resampledParticles.resize(num_particles);
  
  for (int i = 0; i < num_particles; i++) {
    beta += distDouble(gen) * 2.0 * maxWeight;
    //std::cout << "beta : " << beta << std::endl;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }

  // assign the resampled_particles to the previous particles
  particles = resampledParticles;
  
  //weights.clear();
  
  //std::cout << "Resample step done (" << particles.size() << ")" << std::endl;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  //particle.associations.clear();
  //particle.sense_x.clear();
  //particle.sense_y.clear();
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}