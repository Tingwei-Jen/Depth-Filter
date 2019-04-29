#pragma once
#ifndef MAPPING_H
#define MAPPING_H

#include "map.h"
#include <opencv2/opencv.hpp>


class Mapping
{
public:
	Mapping();
	void CreateMap();
	void CreateMap2();   //with other mappoints

	void CreateMap3();
	
	Map* GetMap(){ return map; }


private:
	std::vector<Frame*> CreateFrames();

	Sign* CreateSign(Frame* frame, Frame::SignLabel* label);
	bool UpdateSign(Sign* sign, Frame* frame, Frame::SignLabel* label);
	float AvgReprojectionError(Sign* sign);

private:
	std::vector<MapPoint*> CreateMapPoints(Frame* frame);
	bool Generate3Dpoints(Frame* frame1, Frame* frame2, std::vector<cv::Point3f>& pts3D, std::vector<int>& queryIdxs, std::vector<int>& trainIdxs);
	void UpdateMapPoints(std::vector<MapPoint*>& MapPoints, Frame* frame_new, const std::vector<cv::Point3f>& pts3D, const std::vector<int>& queryIdxs);

private:
	float AvgReprojectionError(const std::vector<cv::Point3f>& pts3D, Frame* frame, const std::vector<cv::Point2f>& pts);
	float ReprojectionError(std::vector<cv::Point3f> pts3D, Sign* sign);
	float ReprojectionError(std::vector<cv::Point3f> pts3D, Frame* frame, Frame::SignLabel* label);
	float ReprojectionError(const cv::Point3f& pt3D, Frame* frame, const cv::Point2f& pt);

private:
	void PlotLabel(Frame* frame, Frame::SignLabel* label);	
	void PlotSign(Sign* sign);
	void PlotSignObs(Sign* sign);
	void PlotMappoints(std::vector<MapPoint*> MapPoints);
	void PlotMappointObs(MapPoint* mappoint);

	void PlotReprojections(std::vector<cv::Point3f> pts3D, Sign* sign);
	void PlotReprojections(std::vector<cv::Point3f> pts3D, Frame* frame, Frame::SignLabel* label);
	void PlotReprojections(const std::vector<cv::Point3f> pts3D, Frame* frame, const std::vector<cv::Point2f> pts);

private:
	Map* map;


};
#endif //MAPPING_H