#pragma once
#ifndef SEED_H
#define SEED_H

#include "frame.h"
#include <opencv2/opencv.hpp>

class Seed
{
public:
	Seed(Frame* frame, cv::Point2f& pt, float depth_mean, float depth_min);
    cv::Point3f GetWorldPose();
    cv::Point3f GetWorldPoseMax();    //!< mean + 3*sigma2
    cv::Point3f GetWorldPoseMin();    //!< mean - 3*sigma2

public:
	static int counter;
    Frame* mframe;                 //!< the frame which the seed was created from.
    int mId;                       //!< Seed ID
    cv::Point2f mpt;               //!< feature's pixel location.
    cv::Point3f mf;                //!< unit vector of feature in camera frame
    float a;                      //!< a of Beta distribution: When high, probability of inlier is large.
    float b;                      //!< b of Beta distribution: When high, probability of outlier is large.
    float mu;                     //!< Mean of normal distribution.
    float z_range;                //!< Max range of the possible depth.
    float sigma2;                 //!< Variance of normal distribution.
};

#endif //SEED_H