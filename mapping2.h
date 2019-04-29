#pragma once
#ifndef MAPPING_H
#define MAPPING_H

#include "frame.h"
#include "seed.h"
#include <opencv2/opencv.hpp>

class Mapping
{
public:
	Mapping();

	void CreateMapPoints();
	void MovingObjectTest();
	void TestBA();


private:
	//update depth filter
	bool UpdateSeed(Seed* seed, Frame* new_frame, const cv::Point2f& new_pt);
	void UpdateSeedBA(Seed* seed, cv::Point3f x3Dp);

	//compute 3D point based on world frame
    bool Triangulation(const cv::Mat& Tcw1, const cv::Mat& Tcw2, const cv::Point3f& pt1_cam, const cv::Point3f& pt2_cam, cv::Point3f& x3Dp);
    cv::Mat ComputeF21(Frame* frame1, Frame* frame2);
    cv::Point2f GetEpipole(Frame* frame1, Frame* frame2);
    cv::Point3f GetEpipolarLine(Frame* frame1, Frame* frame2, const cv::Point2f& pt1);
    float DistPt2Line(const cv::Point2f& pt, const cv::Point3f& line);
    cv::Point2f Reprojection(Seed* seed, Frame* frame);

    void PlanarConstraint(const std::vector<Seed*>& Seeds, std::vector<cv::Point3f>& pts3D_afterPCA);
    void PCAPlane(const std::vector<Seed*>& Seeds, cv::Point3f& normal_vector, float& d);
    void PlaneLineIntersection(const cv::Point3f& camera_center, const cv::Point3f& unit_vector, 
                                const cv::Point3f& normal_vector, const float& d, cv::Point3f& intersect_point);
    
    void TrafficSignCoordinate(const std::vector<cv::Point3f>& SignVertices, cv::Mat& Rsw, cv::Mat& tsw);

    cv::Mat ComputeH21(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2);
    void PerspectiveTransform(const cv::Mat& H21, const std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst);


private:
	void GetParameter(cv::Mat& K, cv::Mat& DistCoef, int& width, int& height);
	bool findCorresponding( const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2 );
};
#endif //MAPPING_H