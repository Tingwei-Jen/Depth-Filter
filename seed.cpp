#include "seed.h"
#include "converter.h"

int Seed::counter = 0;

Seed::Seed(Frame* frame, cv::Point2f& pt, float depth_mean, float depth_min)
:mId(counter++), mpt(pt), a(10), b(10), mu(1.0/depth_mean), z_range(1.0/depth_min)
{
	mframe = frame;
    sigma2 = z_range*z_range/36;
    cv::Point3f pt_cam = cv::Point3f((pt.x-mframe->cx)/mframe->fx,(pt.y-mframe->cy)/mframe->fy,1);
    mf = pt_cam/norm(pt_cam);
}

cv::Point3f Seed::GetWorldPose()
{
	cv::Point3f p = mf * (1/mu);
	cv::Mat p_ = Converter::toCvMat(p);
	cv::Mat Rwc = mframe->GetRotationInverse();
	cv::Mat twc = mframe->GetCameraCenter();
    cv::Mat pw_ = Rwc * p_ + twc;
    cv::Point3f pw = Converter::toCvPoint3f(pw_);

    return pw; 
}

cv::Point3f Seed::GetWorldPoseMax()
{
    float range = 3*sigma2;
    float depth_max = (1/mu) + range;
    cv::Point3f p = mf * depth_max;
    cv::Mat p_ = Converter::toCvMat(p);
    cv::Mat Rwc = mframe->GetRotationInverse();
    cv::Mat twc = mframe->GetCameraCenter();
    cv::Mat pw_ = Rwc * p_ + twc;
    cv::Point3f pw = Converter::toCvPoint3f(pw_);

    return pw; 
}

cv::Point3f Seed::GetWorldPoseMin()
{
    float range = 3*sigma2;
    float depth_min = (1/mu) - range;
    if(depth_min<0.1)
        depth_min = 0.1;

    cv::Point3f p = mf * depth_min;
    cv::Mat p_ = Converter::toCvMat(p);
    cv::Mat Rwc = mframe->GetRotationInverse();
    cv::Mat twc = mframe->GetCameraCenter();
    cv::Mat pw_ = Rwc * p_ + twc;
    cv::Point3f pw = Converter::toCvPoint3f(pw_);

    return pw; 
}