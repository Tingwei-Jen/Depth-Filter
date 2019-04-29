#pragma once
#ifndef TEST_H
#define TEST_H

#include "pcl.h"
#include "depth_filter.h"
#include "frame.h"
#include "seed.h"
//#include "mapping.h"
#include "utils.h"
#include "optimizer.h"


class Test
{
public:
	Test();
	void testReconstructMonitor();
	void testReconstructMonitorHomography();
	void testReconstructLandmark();
	void testBundleAdjustment();

private:
	void getMyntParameter(cv::Mat& K, cv::Mat& distort, int& width, int& height);
	void getAll(std::vector<cv::Mat>& imgsUndistorted, std::vector<cv::Mat>& T_w_cs, std::vector<std::vector<cv::Point2d>>& vertexss);
};
#endif //TEST_H