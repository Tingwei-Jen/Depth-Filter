#pragma once
#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

class Frame
{
public:
    struct SignLabel //rectengle
    {
        std::vector<cv::Point2f> Vertices;
    };   

	Frame(const cv::Mat& img, const cv::Mat &K, std::vector<SignLabel*> Labels, cv::Ptr<cv::xfeatures2d::SURF> detector);
	void SetPose(const cv::Mat& Tcw);

    cv::Mat GetImg(){ return mImg.clone(); }
	cv::Mat GetPose(){ return mTcw.clone(); }
    cv::Mat GetCameraCenter(){ return mOw.clone(); }
    cv::Mat GetRotation(){ return mRcw.clone(); }
    cv::Mat GetTranslation(){ return mtcw.clone(); }
    cv::Mat GetRotationInverse(){ return mRwc.clone(); }
    std::vector<SignLabel*> GetSignLabels(){ return mLabels; }
    std::vector<cv::KeyPoint> GetKeyPoints(){ return mKeyPoints; }
    cv::Mat GetDescriptor(){ return mDescriptors; }

public:
    static int frame_counter;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    int mId;
    cv::Mat mK;

private:
	cv::Mat mImg;
	cv::Mat mTcw;                                          ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵
    cv::Mat mRcw;                                          ///< Rotation from world to camera
    cv::Mat mRwc;                                          ///< Rotation from camera to world
    cv::Mat mtcw;                                          ///< Translation from world to camera   
    cv::Mat mOw;                                           ///< mtwc,Translation from camera to world
    std::vector<SignLabel*> mLabels;                       ///< the signs label in this frame
    std::vector<cv::KeyPoint> mKeyPoints;
    cv::Mat mDescriptors;

};
#endif //FRAME_H
