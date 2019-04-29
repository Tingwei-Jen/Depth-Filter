#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>  

class Utils
{
public:
	static cv::Mat Quaternion2RotM(float x, float y, float z, float w);
    static void ReadVertices(const std::string csv_path, std::vector<std::string>& Imagenames, std::vector<std::vector<cv::Point2f>>& Verticess);
    static std::vector<cv::Mat> ReadCameraPoses(const std::string csv_path, const std::vector<std::string> Imagenames);

    static cv::Point3f Px2Cam(cv::Point2f px, float fx, float fy, float cx, float cy);
    static cv::Point2f Cam2Px(cv::Point3f p_cam, float fx, float fy, float cx, float cy);

    static cv::Mat ComputeF21(const cv::Mat& Tcw1, const cv::Mat& Tcw2, const cv::Mat& K);
    static bool Triangulation(const cv::Mat& Tcw1, const cv::Mat& Tcw2, const cv::Point3f& pt1_cam, const cv::Point3f& pt2_cam, cv::Point3f& x3Dp);


};
#endif //MAPPING_H