#pragma once
#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "map.h"
#include <opencv2/opencv.hpp>

class LocalMapping
{
public:
	LocalMapping(Map* map);
	void Run();
	Map* GetMap(){ return mMap; }


private:
	std::vector<Frame*> CreateFrames1();
	std::vector<Frame*> CreateFrames2();

private:
	void GTerror(Sign* sign, const float& width_gt, const float& height_gt, const float& LT_gt);
	float AvgReprojectionError(const std::vector<cv::Point3f>& pts3D, Frame* frame, const std::vector<cv::Point2f>& pts);

private:
	void PlotPts(Frame* frame, const std::vector<cv::Point2f>& pts);
	void PlotReprojections(const std::vector<cv::Point3f>& pts3D, Frame* frame, const std::vector<cv::Point2f>& pts);

private:
	Map* mMap;







};
#endif //LOCALMAPPING_H