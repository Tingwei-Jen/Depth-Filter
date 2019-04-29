#pragma once
#ifndef MATCHER_H
#define MATCHER_H

#include "frame.h"
#include "seed.h"
#include <opencv2/opencv.hpp>

class Matcher
{
public:
	static void FindCorresponding(Frame* frame1, Frame* frame2, std::vector<cv::DMatch>& good_matches);
	static cv::Point2f GetEpipole(Frame* frame1, Frame* frame2);
	static cv::Point3f GetEpipolarLine(Frame* frame1, Frame* frame2, const cv::Point2f& pt1);
	static std::vector<cv::Point2f> GetEpipolarSegment(Seed* seed, Frame* frame);

};
#endif //MATCHER_H