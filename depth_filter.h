#pragma once
#ifndef DEPTH_FILTER_H
#define DEPTH_FILTER_H

#include "seed.h"
#include <opencv2/opencv.hpp>

class Depth_Filter
{
public:
    //compute observed depth std
    static void ComputeTau(const cv::Mat& T12, const cv::Point3f& P, const float& px_error_angle, float& tau);
    //gaussian&beta distribution fusion,  (mu, sigma2) combine with (x, tau2)(observation)
    static void UpdateFilter(const float& x, const float& tau2, Seed* seed);   
};
#endif //DEPTH_FILTER_H