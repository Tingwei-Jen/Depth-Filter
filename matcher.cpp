#include "matcher.h"

#include "converter.h"
#include "utils.h"

using namespace std;
using namespace cv;

void Matcher::FindCorresponding(Frame* frame1, Frame* frame2, vector<DMatch>& good_matches)
{
    good_matches.clear();

	Mat descriptors1 = frame1->GetDescriptor();
	Mat descriptors2 = frame2->GetDescriptor();

	FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( descriptors1, descriptors2, matches );

    double max_dist = 0; 
    double min_dist = 100;
    for( int i = 0; i < descriptors1.rows; i++ )
    { 
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    for( int i = 0; i < descriptors1.rows; i++ )
    {
        if( matches[i].distance <= max(2*min_dist, 0.02) )
        { 
            good_matches.push_back( matches[i]); 
        }
    }
}

Point2f Matcher::GetEpipole(Frame* frame1, Frame* frame2)
{
    Mat Rcw2 = frame2->GetRotation();
    Mat tcw2 = frame2->GetTranslation();

    Mat Ow1 = frame1->GetCameraCenter();
    Mat Oc2w1 = Rcw2 * Ow1 + tcw2;

    float fx = frame1->fx;
    float fy = frame1->fy;
    float cx = frame1->cx;
    float cy = frame1->cy;

    return Point2f(Oc2w1.at<float>(0)*fx/Oc2w1.at<float>(2) + cx, Oc2w1.at<float>(1)*fy/Oc2w1.at<float>(2) + cy);
}

// float a = line.x; float b = line.y; float c = line.z;
// cv::line(plot_cur, Point(0,-c/b), Point(width, -(c+a*width)/b), Scalar(0,0,255));
Point3f Matcher::GetEpipolarLine(Frame* frame1, Frame* frame2, const Point2f& pt1)
{
    Mat F21 = Utils::ComputeF21(frame1->GetPose(), frame2->GetPose(), frame1->mK);
    Mat pt1_ = ( Mat_<float> ( 3,1 ) <<  pt1.x, pt1.y, 1 );
    Mat line = F21 * pt1_;
    return Point3f(line.at<float>(0,0), line.at<float>(1,0), line.at<float>(2,0));
}

vector<Point2f> GetEpipolarSegment(Seed* seed, Frame* frame)
{

    Point3f pose_min = seed->GetWorldPoseMin();
    Point3f pose_max = seed->GetWorldPoseMax();
    
    //to camera frame
    Mat Rcw = frame->GetRotation();
    Mat tcw = frame->GetTranslation();

    Mat pose_min_cam_ = Rcw*Converter::toCvMat(pose_min) + tcw;
    Mat pose_max_cam_ = Rcw*Converter::toCvMat(pose_max) + tcw;

    Point3f pose_min_cam = Converter::toCvPoint3f(pose_min_cam_);
    Point3f pose_max_cam = Converter::toCvPoint3f(pose_max_cam_);
    
    //to pixel frame
    float fx = frame->fx;
    float fy = frame->fy;
    float cx = frame->cx;
    float cy = frame->cy;

    Point2f pose_min_px = Utils::Cam2Px(pose_min_cam, fx, fy, cx, cy);
    Point2f pose_max_px = Utils::Cam2Px(pose_max_cam, fx, fy, cx, cy);
    
    vector<Point2f> Segment;
    Segment.push_back(pose_min_px);
    Segment.push_back(pose_max_px);

    return Segment;
}