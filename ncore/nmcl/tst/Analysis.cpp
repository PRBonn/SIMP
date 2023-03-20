#include "Analysis.h"
#include <iostream>


float CosineDistance(float t1, float t2)
{
	Eigen::Vector2f p1 = Eigen::Vector2f(cos(t1), sin(t1));
	Eigen::Vector2f p2 = Eigen::Vector2f(cos(t2), sin(t2));
	float dist = 1 - ( (p1.dot(p2))/ (p1.norm() * p2.norm()) );

	return dist;
}


Eigen::Vector3f RMSE(std::vector<Eigen::Vector3f> predictions, std::vector<Eigen::Vector3f> groundTruth)
{
	double x = 0.0;
	double y = 0.0;
	double theta = 0.0;

	int cnt = 0;

	for(int i = 10; i < predictions.size(); ++i)
	{
		Eigen::Vector3f p = predictions[i];
		Eigen::Vector3f g = groundTruth[i];
		Eigen::Vector3f delta = p - g;
		x += delta(0) * delta(0);
		y += delta(1) * delta(1);
		theta += CosineDistance(p(2), g(2));
		++cnt;
	}

	x = sqrt(x / cnt);
	y = sqrt(y / cnt);
	theta = theta / cnt;

	//std::cout << "rmse: " << x << ", " << y << ", " << theta << std::endl;

	return Eigen::Vector3f(x, y, theta);
}