
#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <vector>
#include <eigen3/Eigen/Dense>

float CosineDistance(float t1, float t2);


Eigen::Vector3f RMSE(std::vector<Eigen::Vector3f> predictions, std::vector<Eigen::Vector3f> groundTruth);

#endif 