#include "localmapping.h"

#include "converter.h"
#include "utils.h"
#include <iostream>
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

LocalMapping::LocalMapping(Map* map)
{
	this->mMap = map;
}

void LocalMapping::Run()
{
	std::vector<Frame*> Frames1_ = CreateFrames1();
	std::vector<Frame*> Frames2_ = CreateFrames2();

    vector<Frame*> Frames1;
    for(int i=Frames1_.size()-1; i>=0; i--)  //Reverse frame
        Frames1.push_back(Frames1_[i]);

    vector<Frame*> Frames2;
    for(int i=Frames2_.size()-1; i>=0; i--)  //Reverse frame
        Frames2.push_back(Frames2_[i]);

    for(int i=26; i<Frames1.size(); i++)
    	this->mMap->AddFrame(Frames1[i]);

    for(int i=28; i<Frames2.size(); i++)
    	this->mMap->AddFrame(Frames2[i]);



	// for(int i=26; i<57; i++)
	// {
	// 	vector<Point2f> pts;
	// 	pts.push_back(Frames1[i]->GetSignLabels()[0]->Vertices[0]);
	// 	pts.push_back(Frames1[i]->GetSignLabels()[0]->Vertices[1]);
	// 	pts.push_back(Frames1[i]->GetSignLabels()[0]->Vertices[2]);
	// 	pts.push_back(Frames1[i]->GetSignLabels()[0]->Vertices[3]);
	// 	PlotPts(Frames1[i], pts);
	// }















}





std::vector<Frame*> LocalMapping::CreateFrames1()
{

    Mat K = cv::Mat(3,3, CV_32F);
    K.at<float>(0,0) = 712.58465576171875000; K.at<float>(0,1) = 0; K.at<float>(0,2) = 613.71289062500000000;
    K.at<float>(1,0) = 0; K.at<float>(1,1) = 713.57849121093750000; K.at<float>(1,2) = 386.50469970703125000;
    K.at<float>(2,0) = 0; K.at<float>(2,1) = 0; K.at<float>(2,2) = 1;

    vector<string> Imagenames; 
    vector<vector<Point2f>> Verticess;
    Utils::ReadVertices("../imgs/line1.csv", Imagenames, Verticess);

    vector<Mat> Twcs;
    Twcs = Utils::ReadCameraPoses("../line1-vins_estimator-camera_pose.csv", Imagenames);

    int n_data = Imagenames.size();
    
    vector<Mat> Imgs;
    for(int i=0; i<n_data; i++)
    {
        Imgs.push_back(imread("../imgs/imgs_line1_undistort/"+Imagenames[i], 1));
    }

    //create frame;
    Ptr<SURF> detector = SURF::create();
    detector->setHessianThreshold(400);

    std::vector<Frame*> Frames_;
    for(int i=0; i<n_data; i++)
    {
        int n_label;

        if(Verticess[i].size() == 4)
            n_label = 1;
        else if (Verticess[i].size() == 8)
            n_label = 2;

        vector<Frame::SignLabel*> Labels;

        for(int j=0; j<n_label; j++)
        {
            Frame::SignLabel* label = new Frame::SignLabel();
            label->Vertices.push_back(Verticess[i][j*4]);
            label->Vertices.push_back(Verticess[i][j*4+1]);
            label->Vertices.push_back(Verticess[i][j*4+2]);
            label->Vertices.push_back(Verticess[i][j*4+3]);

            Labels.push_back(label);
        }

        Frame* frame = new Frame(Imgs[i], K, Labels, detector);
        frame->SetPose(Twcs[i].inv());
        Frames_.push_back(frame);
    }

    return Frames_;
}

std::vector<Frame*> LocalMapping::CreateFrames2()
{

    Mat K = cv::Mat(3,3, CV_32F);
    K.at<float>(0,0) = 712.58465576171875000; K.at<float>(0,1) = 0; K.at<float>(0,2) = 613.71289062500000000;
    K.at<float>(1,0) = 0; K.at<float>(1,1) = 713.57849121093750000; K.at<float>(1,2) = 386.50469970703125000;
    K.at<float>(2,0) = 0; K.at<float>(2,1) = 0; K.at<float>(2,2) = 1;

    vector<string> Imagenames; 
    vector<vector<Point2f>> Verticess;
    Utils::ReadVertices("../imgs/line2.csv", Imagenames, Verticess);

    vector<Mat> Twcs;
    Twcs = Utils::ReadCameraPoses("../line2-vins_estimator-camera_pose.csv", Imagenames);

    int n_data = Imagenames.size();
    
    vector<Mat> Imgs;
    for(int i=0; i<n_data; i++)
    {
        Imgs.push_back(imread("../imgs/imgs_line2_undistort/"+Imagenames[i], 1));
    }

    //create frame;
    Ptr<SURF> detector = SURF::create();
    detector->setHessianThreshold(400);

    std::vector<Frame*> Frames_;
    for(int i=0; i<n_data; i++)
    {
        int n_label;

        if(Verticess[i].size() == 4)
            n_label = 1;
        else if (Verticess[i].size() == 8)
            n_label = 2;

        vector<Frame::SignLabel*> Labels;

        for(int j=0; j<n_label; j++)
        {
            Frame::SignLabel* label = new Frame::SignLabel();
            label->Vertices.push_back(Verticess[i][j*4]);
            label->Vertices.push_back(Verticess[i][j*4+1]);
            label->Vertices.push_back(Verticess[i][j*4+2]);
            label->Vertices.push_back(Verticess[i][j*4+3]);

            Labels.push_back(label);
        }

        Frame* frame = new Frame(Imgs[i], K, Labels, detector);
        frame->SetPose(Twcs[i].inv());
        Frames_.push_back(frame);
    }

    return Frames_;
}




/*
private GT error, reprojection error
*/
// sign1: width_gt = 0.31, height_gt = 0.174, sign1_LT_gt = 3.0;
// sign0: width_gt = 0.31, height_gt = 0.174, sign1_LT_gt = 3.95;
void LocalMapping::GTerror(Sign* sign, const float& width_gt, const float& height_gt, const float& LT_gt)
{
    float sign_width_err_rate = abs((sign->GetAvgWidth() - width_gt))/width_gt*100;
    float sign_height_err_rate = abs((sign->GetAvgHeight() - height_gt))/height_gt*100;
    float sign_LT_err_rate = abs(( norm(sign->GetAllSeeds()[0]->GetWorldPose()) - LT_gt )) / LT_gt*100;

    cout<<"sign_width_err_rate: "<<sign_width_err_rate<<" %"<<endl;
    cout<<"sign_height_err_rate: "<<sign_height_err_rate<<" %"<<endl;
    cout<<"sign_LT_err_rate: "<<sign_LT_err_rate<<" %"<<endl;
}

float LocalMapping::AvgReprojectionError(const vector<Point3f>& pts3D, Frame* frame, const vector<Point2f>& pts)
{
	float avg_error = 0.0;

	for(int i=0; i<pts3D.size(); i++)
	{
    	Mat Pc_ = frame->GetRotation()*Converter::toCvMat(pts3D[i]) + frame->GetTranslation();
    	Point3f Pc = Converter::toCvPoint3f(Pc_);
	    Point2f pt_repro = Utils::Cam2Px(Pc, frame->fx, frame->fy, frame->cx, frame->cy);
	    avg_error += norm(pts[i]-pt_repro);
	}

	return avg_error/=pts3D.size();
}

/*
private plots
*/
void LocalMapping::PlotPts(Frame* frame, const std::vector<cv::Point2f>& pts)
{
    Mat img = frame->GetImg();

    for(int i=0; i<pts.size(); i++)
    	cv::circle(img, pts[i], 2, cv::Scalar(255,0,0), -1);

    imshow("img", img);
    waitKey(0);
}

void LocalMapping::PlotReprojections(const vector<Point3f>& pts3D, Frame* frame, const vector<Point2f>& pts)
{
    Mat img = frame->GetImg();

    for(int i=0; i<pts3D.size(); i++)
    {
        Mat Pc_ = frame->GetRotation()*Converter::toCvMat(pts3D[i]) + frame->GetTranslation();
        Point3f Pc = Converter::toCvPoint3f(Pc_);
        Point2f pt_repro = Utils::Cam2Px(Pc, frame->fx, frame->fy, frame->cx, frame->cy);
   
        cv::circle(img, pt_repro, 2, cv::Scalar(0,0,255), -1);
        cv::circle(img, pts[i], 2, cv::Scalar(255,0,0), -1);
    }

    imshow("img", img);
    waitKey(0);
}