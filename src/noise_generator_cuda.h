#ifndef NOISE_GENERATOR_H
#define NOISE_GENERATOR_H

#include <vector>

void generateNoiseCUDA(std::vector<double>& noise, double mean, double stddev);

#endif // NOISE_GENERATOR_H