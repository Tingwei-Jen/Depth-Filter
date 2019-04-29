#include "mapping.h"

#include "converter.h"
#include "depth_filter.h"
#include "matcher.h"
#include "optimizer.h"
#include "utils.h"

#include <Eigen/Dense>
#include <iostream>
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Mapping::Mapping()
{
	cout<<"Construct Mapping"<<endl;
    this->map = new Map();
}

void Mapping::CreateMap()
{
    vector<Frame*> Frames_ = CreateFrames();
    vector<Frame*> Frames;
    for(int i=Frames_.size()-1; i>=0; i--)  //Reverse frame
        Frames.push_back(Frames_[i]);

    //Create Sign
    //this->map->AddSign(CreateSign(Frames[0], Frames[0]->GetSignLabels()[0]));    //Init sign0(0-37), sign1(25-56)
    this->map->AddSign(CreateSign(Frames[25], Frames[25]->GetSignLabels()[0]));  //Init sign1(25-56)

    //Sign* sign0 = this->map->GetAllSigns()[0];
    Sign* sign1 = this->map->GetAllSigns()[0];


    PlotSign(sign1);
    PlotSignObs(sign1);


    //Update Sign0 
    // for(int i=1; i<38; i++)
    // {
    //     int index = 0;
    //     if (Frames[i]->GetSignLabels().size()==2)
    //         index = 1;
        
    //     UpdateSign(sign0, Frames[i], Frames[i]->GetSignLabels()[index]);
    // }

    //Update Sign1
    for(int i=26; i<57; i++)
    {      
        UpdateSign(sign1, Frames[i], Frames[i]->GetSignLabels()[0]);
    }

    cout<<"avg_px_error_sign1: "<<AvgReprojectionError(sign1)<<endl; 
    GroundTruthError1(sign1);

    Optimizer::LocalBundleAdjustment(sign1,10);

    cout<<"avg_px_error_sign1: "<<AvgReprojectionError(sign1)<<endl; 
    GroundTruthError1(sign1);

}



void Mapping::CreateMap2()
{

    vector<Frame*> Frames_ = CreateFrames();
    vector<Frame*> Frames;
    for(int i=Frames_.size()-1; i>=0; i--)  //Reverse frame
        Frames.push_back(Frames_[i]);

    Sign* sign = CreateSign(Frames[25], Frames[25]->GetSignLabels()[0]);
    this->map->AddSign(sign);
    
    std::vector<MapPoint*> MapPoints = CreateMapPoints(Frames[25]);
    for(int i=0; i<MapPoints.size(); i++)
        this->map->AddMapPoint(MapPoints[i]);

    //PlotSign(sign);
    //PlotMappoints(MapPoints);

    for(int i=26; i<57; i++)
    {    
        vector<Point3f> pts3D; vector<int> queryIdxs; vector<int> trainIdxs;
        if(!Generate3Dpoints(Frames[25], Frames[i], pts3D, queryIdxs, trainIdxs))
            continue;

        if(pts3D.size()<5)
            continue;

        vector<KeyPoint> KeyPoints1 = Frames[25]->GetKeyPoints();
        vector<KeyPoint> KeyPoints2 = Frames[i]->GetKeyPoints();

        vector<Point2f> pts1, pts2;
        for(int i=0; i<pts3D.size(); i++)
        {
            pts1.push_back(KeyPoints1[ queryIdxs[i] ].pt);
            pts2.push_back(KeyPoints2[ trainIdxs[i] ].pt);
        }

        cout<<i<<endl;
        // cout<<AvgReprojectionError(pts3D, Frames[25], pts1)<<endl;
        // cout<<AvgReprojectionError(pts3D, Frames[i], pts2)<<endl;
        // PlotReprojections(pts3D, Frames[25], pts1);
        // PlotReprojections(pts3D, Frames[i], pts2);

        Optimizer::BundleAdjustmentTwoFrames(Frames[25], Frames[i], pts3D, pts1, pts2);
        
        // cout<<AvgReprojectionError(pts3D, Frames[25], pts1)<<endl;
        // cout<<AvgReprojectionError(pts3D, Frames[i], pts2)<<endl;
        // PlotReprojections(pts3D, Frames[25], pts1);
        // PlotReprojections(pts3D, Frames[i], pts2);
        

        UpdateMapPoints(MapPoints, Frames[i], pts3D, queryIdxs);


    }
}







void Mapping::CreateMap3()
{
    vector<Frame*> Frames_ = CreateFrames();
    vector<Frame*> Frames;
    for(int i=Frames_.size()-1; i>=0; i--)  //Reverse frame
        Frames.push_back(Frames_[i]);

    Sign* sign = CreateSign(Frames[25], Frames[25]->GetSignLabels()[0]);
    this->map->AddSign(sign);
    
    std::vector<MapPoint*> MapPoints = CreateMapPoints(Frames[25]);
    for(int i=0; i<MapPoints.size(); i++)
        this->map->AddMapPoint(MapPoints[i]);



    //frame
    Frame* frame2 = Frames[31];
    //label
    Frame::SignLabel* label = Frames[31]->GetSignLabels()[0];

    //dist_Ows
    Point3f Ow_sign = Converter::toCvPoint3f(sign->GetFirstFrame()->GetCameraCenter());
    Point3f Ow_cur = Converter::toCvPoint3f(frame2->GetCameraCenter());
    float dist_Ows = norm(Ow_cur-Ow_sign);
    cout<<"dist_Ows: "<<dist_Ows<<endl;

    //Tcw
    float fx = frame2->fx;
    float fy = frame2->fy;
    float cx = frame2->cx;
    float cy = frame2->cy;

    Mat Tcw1 = sign->GetFirstFrame()->GetPose();
    Mat Tcw2 = frame2->GetPose();


    //sign
    vector<Point3f> pts3Dsign;
    for(int i=0; i<4; i++)
    {
        Point3f pt1_cam = Utils::Px2Cam(sign->GetAllSeeds()[i]->mpt, fx, fy, cx, cy);
        Point3f pt2_cam = Utils::Px2Cam(label->Vertices[i], fx, fy, cx, cy);
       
        Point3f pt3D;
        bool ret = Utils::Triangulation(Tcw1, Tcw2, pt1_cam, pt2_cam, pt3D);  

        pts3Dsign.push_back(pt3D);
    }

    //mappoints
    vector<KeyPoint> KeyPoints1 = sign->GetFirstFrame()->GetKeyPoints();
    vector<KeyPoint> KeyPoints2 = frame2->GetKeyPoints();

    //feature matching
    vector<DMatch> good_matches;
    Matcher::FindCorresponding(sign->GetFirstFrame(), frame2, good_matches);

    vector<Point3f> pts3Dmappoint;
    vector<int> queryIdxs;
    vector<int> trainIdxs;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        Point2f pt1 = KeyPoints1[ good_matches[i].queryIdx ].pt;
        Point2f pt2 = KeyPoints2[ good_matches[i].trainIdx ].pt;

        Point3f pt1_cam = Utils::Px2Cam(pt1, fx, fy, cx, cy);
        Point3f pt2_cam = Utils::Px2Cam(pt2, fx, fy, cx, cy);

        Point3f pt3D;
        bool ret = Utils::Triangulation(Tcw1, Tcw2, pt1_cam, pt2_cam, pt3D);  

        if(!ret) continue;
        if(ReprojectionError(pt3D, sign->GetFirstFrame(), pt1)>10.0) continue;
        if(ReprojectionError(pt3D, frame2, pt2)>10.0) continue;
    
        pts3Dmappoint.push_back(pt3D);
        queryIdxs.push_back(good_matches[i].queryIdx);
        trainIdxs.push_back(good_matches[i].trainIdx);
    }


    vector<Point2f> pts1, pts2;
    for(int i=0; i<pts3Dmappoint.size(); i++)
    {
        pts1.push_back(KeyPoints1[ queryIdxs[i] ].pt);
        pts2.push_back(KeyPoints2[ trainIdxs[i] ].pt);
    }


    //update add observation
    sign->AddObservation(frame2, label);

    for(int i=0; i<pts3Dmappoint.size(); i++)
    {
        MapPoints[queryIdxs[i]]->AddObservation(frame2, KeyPoints2[ trainIdxs[i] ].pt);
    }


    //update sign
    for(int i=0; i<pts3Dsign.size(); i++)
    {
        cv::Point3f Pw = pts3Dsign[i];
        cv::Mat Pc_ = sign->GetFirstFrame()->GetRotation()*Converter::toCvMat(Pw) + sign->GetFirstFrame()->GetTranslation();
        cv::Point3f Pc = Converter::toCvPoint3f(Pc_);
        sign->GetAllSeeds()[i]->mf = Pc/norm(Pc);
        sign->GetAllSeeds()[i]->mu = 1.0/norm(Pc);
    }

    //update mappoint
    for(int i=0; i<pts3Dmappoint.size(); i++)
    {
        cv::Point3f Pw = pts3Dmappoint[i];
        cv::Mat Pc_ = sign->GetFirstFrame()->GetRotation()*Converter::toCvMat(Pw) + sign->GetFirstFrame()->GetTranslation();
        cv::Point3f Pc = Converter::toCvPoint3f(Pc_);
        MapPoints[queryIdxs[i]]->GetSeed()->mf = Pc/norm(Pc);
        MapPoints[queryIdxs[i]]->GetSeed()->mu = 1.0/norm(Pc);
    }

    //reprojection error
    cout<<"ReprojectionError sign frame1: "<<ReprojectionError(pts3Dsign, sign)<<endl;
    cout<<"ReprojectionError sign frame2: "<<ReprojectionError(pts3Dsign, frame2, label)<<endl;
    PlotReprojections(pts3Dsign, sign);
    PlotReprojections(pts3Dsign, frame2, label);

    cout<<"AvgReprojectionError mappoint frame1: "<<AvgReprojectionError(pts3Dmappoint, sign->GetFirstFrame(), pts1)<<endl;
    cout<<"AvgReprojectionError mappoint frame2: "<<AvgReprojectionError(pts3Dmappoint, frame2, pts2)<<endl;
    PlotReprojections(pts3Dmappoint, sign->GetFirstFrame(), pts1);
    PlotReprojections(pts3Dmappoint, frame2, pts2);
    
    //ground truth error
    GroundTruthError1(pts3Dsign);



    vector<Point3f> pts3d_op;
    vector<Point2f> pts1_op;
    vector<Point2f> pts2_op;

    for(int i=0; i<4; i++)
    {
        pts3d_op.push_back(pts3Dsign[i]);
        pts1_op.push_back(sign->GetAllSeeds()[i]->mpt);
        pts2_op.push_back(label->Vertices[i]);
    }


    for(int i=0; i<pts3Dmappoint.size(); i++)
    {
        pts3d_op.push_back(pts3Dmappoint[i]);
        pts1_op.push_back(pts1[i]);
        pts2_op.push_back(pts2[i]);
    }

    Optimizer::BundleAdjustmentTwoFrames(sign->GetFirstFrame(), frame2, pts3d_op, pts1_op, pts2_op);




    for(int i=0; i<4; i++)
    {
        pts3Dsign[i] = pts3d_op[i];
    }

    for(int i=0; i<pts3Dmappoint.size(); i++)
    {
        pts3Dmappoint[i] = pts3d_op[i+4];
    }



    //update sign
    for(int i=0; i<pts3Dsign.size(); i++)
    {
        cv::Point3f Pw = pts3Dsign[i];
        cv::Mat Pc_ = sign->GetFirstFrame()->GetRotation()*Converter::toCvMat(Pw) + sign->GetFirstFrame()->GetTranslation();
        cv::Point3f Pc = Converter::toCvPoint3f(Pc_);
        sign->GetAllSeeds()[i]->mf = Pc/norm(Pc);
        sign->GetAllSeeds()[i]->mu = 1.0/norm(Pc);
    }

    //update mappoint
    for(int i=0; i<pts3Dmappoint.size(); i++)
    {
        cv::Point3f Pw = pts3Dmappoint[i];
        cv::Mat Pc_ = sign->GetFirstFrame()->GetRotation()*Converter::toCvMat(Pw) + sign->GetFirstFrame()->GetTranslation();
        cv::Point3f Pc = Converter::toCvPoint3f(Pc_);
        MapPoints[queryIdxs[i]]->GetSeed()->mf = Pc/norm(Pc);
        MapPoints[queryIdxs[i]]->GetSeed()->mu = 1.0/norm(Pc);
    }


    cout<<"ReprojectionError sign frame1: "<<ReprojectionError(pts3Dsign, sign)<<endl;
    cout<<"ReprojectionError sign frame2: "<<ReprojectionError(pts3Dsign, frame2, label)<<endl;
    PlotReprojections(pts3Dsign, sign);
    PlotReprojections(pts3Dsign, frame2, label);

    cout<<"AvgReprojectionError mappoint frame1: "<<AvgReprojectionError(pts3Dmappoint, sign->GetFirstFrame(), pts1)<<endl;
    cout<<"AvgReprojectionError mappoint frame2: "<<AvgReprojectionError(pts3Dmappoint, frame2, pts2)<<endl;
    PlotReprojections(pts3Dmappoint, sign->GetFirstFrame(), pts1);
    PlotReprojections(pts3Dmappoint, frame2, pts2);

    GroundTruthError1(pts3Dsign);




}









































std::vector<Frame*> Mapping::CreateFrames()
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



Sign* Mapping::CreateSign(Frame* frame, Frame::SignLabel* label)
{
    std::vector<Seed*> Seeds;
    Seeds.push_back(new Seed(frame, label->Vertices[0], 1.0, 0.1));
    Seeds.push_back(new Seed(frame, label->Vertices[1], 1.0, 0.1));
    Seeds.push_back(new Seed(frame, label->Vertices[2], 1.0, 0.1));
    Seeds.push_back(new Seed(frame, label->Vertices[3], 1.0, 0.1));
    Sign* sign = new Sign(Seeds, frame);
    sign->AddObservation(frame, label);
    return sign;
}


bool Mapping::UpdateSign(Sign* sign, Frame* frame, Frame::SignLabel* label)
{

    Point3f Ow_sign = Converter::toCvPoint3f(sign->GetFirstFrame()->GetCameraCenter());
    Point3f Ow_cur = Converter::toCvPoint3f(frame->GetCameraCenter());

    float dist_Ows = norm(Ow_cur-Ow_sign);

    //ensure long enough
    if(dist_Ows<0.3)
        return false;

    float fx = frame->fx;
    float fy = frame->fy;
    float cx = frame->cx;
    float cy = frame->cy;

    Mat Tcw1 = sign->GetFirstFrame()->GetPose();
    Mat Tcw2 = frame->GetPose();

    //Triangulation //global pose
    vector<Point3f> pts3D;
    for(int i=0; i<4; i++)
    {
        Point3f pt1_cam = Utils::Px2Cam(sign->GetAllSeeds()[i]->mpt, fx, fy, cx, cy);
        Point3f pt2_cam = Utils::Px2Cam(label->Vertices[i], fx, fy, cx, cy);
       
        Point3f pt3D;
        bool ret = Utils::Triangulation(Tcw1, Tcw2, pt1_cam, pt2_cam, pt3D);  

        if(!ret)
            return false;

        pts3D.push_back(pt3D);
    }

    if(ReprojectionError(pts3D, sign)>4.0)
        return false;

    if(ReprojectionError(pts3D, frame, label)>4.0)
        return false;

    //Depth filter
    Mat T12 = Tcw1 * Tcw2.inv(); 
    float px_noise = 1.0;
    float px_error_angle = atan(px_noise/(2.0*fx))*2.0; // law of chord (sehnensatz)
    
    for(int i=0; i<4; i++)
    {
        Mat Rcw1 = sign->GetFirstFrame()->GetRotation();
        Mat tcw1 = sign->GetFirstFrame()->GetTranslation();
        Mat pt3D1_ = Rcw1*Converter::toCvMat(pts3D[i]) + tcw1;
        Point3f pt3D1 = Converter::toCvPoint3f(pt3D1_);     //seed's frame

        float tau;       
        Depth_Filter::ComputeTau(T12, pt3D1, px_error_angle, tau);    //compute tau

        float z = norm(pt3D1);
        float tau_inverse = 0.5 * (1.0/fmax(0.0001, z-tau) - 1.0/(z+tau));
        Depth_Filter::UpdateFilter(1./z, tau_inverse*tau_inverse, sign->GetAllSeeds()[i]);

        //cout<< sign->GetAllSeeds()[i]->mu <<"    "<< sign->GetAllSeeds()[i]->sigma2<<endl;  
    }

    //update
    sign->AddObservation(frame, label);
}


std::vector<MapPoint*> Mapping::CreateMapPoints(Frame* frame)
{
    std::vector<MapPoint*> MapPoints;

    std::vector<cv::KeyPoint> KeyPoints = frame->GetKeyPoints();

    for(int i=0; i<KeyPoints.size(); i++)
    {
        MapPoint* mappoint = new MapPoint(new Seed(frame, KeyPoints[i].pt, 1.0, 0.1), frame);
        mappoint->AddObservation(frame, KeyPoints[i].pt);
        MapPoints.push_back(mappoint);
    }

    return MapPoints;
}

bool Mapping::Generate3Dpoints(Frame* frame1, Frame* frame2, vector<Point3f>& pts3D, vector<int>& queryIdxs, vector<int>& trainIdxs)
{

    pts3D.clear();
    queryIdxs.clear();
    trainIdxs.clear();

    vector<KeyPoint> KeyPoints1 = frame1->GetKeyPoints();
    vector<KeyPoint> KeyPoints2 = frame2->GetKeyPoints();

    //feature matching
    vector<DMatch> good_matches;
    Matcher::FindCorresponding(frame1, frame2, good_matches);

    //ensure long enough
    Point3f Ow1 = Converter::toCvPoint3f(frame1->GetCameraCenter());
    Point3f Ow2 = Converter::toCvPoint3f(frame2->GetCameraCenter());
    
    float dist_Ows = norm(Ow2-Ow1);
    if(dist_Ows<0.3)
        return false;

    //Triangulation, global pose
    float fx = frame1->fx;
    float fy = frame1->fy;
    float cx = frame1->cx;
    float cy = frame1->cy;

    Mat Tcw1 = frame1->GetPose();
    Mat Tcw2 = frame2->GetPose();

    for( int i = 0; i < good_matches.size(); i++ )
    {
        Point2f pt1 = KeyPoints1[ good_matches[i].queryIdx ].pt;
        Point2f pt2 = KeyPoints2[ good_matches[i].trainIdx ].pt;


        Point3f pt1_cam = Utils::Px2Cam(pt1, fx, fy, cx, cy);
        Point3f pt2_cam = Utils::Px2Cam(pt2, fx, fy, cx, cy);

        Point3f pt3D;
        bool ret = Utils::Triangulation(Tcw1, Tcw2, pt1_cam, pt2_cam, pt3D);  

        if(!ret) continue;
        if(ReprojectionError(pt3D, frame1, pt1)>10.0) continue;
        if(ReprojectionError(pt3D, frame2, pt2)>10.0) continue;
    
        pts3D.push_back(pt3D);
        queryIdxs.push_back(good_matches[i].queryIdx);
        trainIdxs.push_back(good_matches[i].trainIdx);
    }
}

void Mapping::UpdateMapPoints(vector<MapPoint*>& MapPoints, Frame* frame_new, const vector<Point3f>& pts3D, const vector<int>& queryIdxs)
{
    
    Mat Tcw1 = MapPoints[0]->GetFirstFrame()->GetPose();
    Mat Tcw2 = frame_new->GetPose();

    Mat T12 = Tcw1 * Tcw2.inv(); 
    float px_noise = 1.0;
    float fx = frame_new->fx;

    float px_error_angle = atan(px_noise/(2.0*fx))*2.0; // law of chord (sehnensatz)


    for(int i=0; i<pts3D.size(); i++)
    {
        Mat Rcw1 = MapPoints[0]->GetFirstFrame()->GetRotation();
        Mat tcw1 = MapPoints[0]->GetFirstFrame()->GetTranslation();
        Mat pt3D1_ = Rcw1*Converter::toCvMat(pts3D[i]) + tcw1;
        Point3f pt3D1 = Converter::toCvPoint3f(pt3D1_);     //seed's frame

        float tau;       
        Depth_Filter::ComputeTau(T12, pt3D1, px_error_angle, tau);    //compute tau

        float z = norm(pt3D1);
        float tau_inverse = 0.5 * (1.0/fmax(0.0001, z-tau) - 1.0/(z+tau));
        Depth_Filter::UpdateFilter(1./z, tau_inverse*tau_inverse, MapPoints[queryIdxs[i]]->GetSeed());

        cout<< MapPoints[queryIdxs[i]]->GetSeed()->mu <<"    "<< MapPoints[queryIdxs[i]]->GetSeed()->sigma2<<endl;  
    }



}



float Mapping::AvgReprojectionError(const std::vector<cv::Point3f>& pts3D, Frame* frame, const std::vector<cv::Point2f>& pts)
{
    float avg_error = 0.0;

    for(int i=0; i<pts3D.size(); i++)
    {
        avg_error += ReprojectionError(pts3D[i], frame, pts[i]);
    }

    return avg_error/=pts3D.size();
}

float Mapping::ReprojectionError(const cv::Point3f& pt3D, Frame* frame, const cv::Point2f& pt)
{
    Mat Rcw = frame->GetRotation();
    Mat tcw = frame->GetTranslation();
    
    Mat Pc_ = Rcw*Converter::toCvMat(pt3D) + tcw;
    Point3f Pc = Converter::toCvPoint3f(Pc_);

    float fx = frame->fx;
    float fy = frame->fy;
    float cx = frame->cx;
    float cy = frame->cy;
    Point2f pt_repro = Utils::Cam2Px(Pc, fx, fy, cx, cy);

    return norm(pt-pt_repro);
}

float Mapping::AvgReprojectionError(Sign* sign)
{

    vector<Point3f> pts3D;
    for(int i=0; i<4; i++)
        pts3D.push_back(sign->GetAllSeeds()[i]->GetWorldPose());

    float avg_px_error_sign = 0.0;
    
    for(int i=0; i<sign->nObs; i++)
    {
        Frame* frame = sign->GetObservationFrames()[i];
        Frame::SignLabel* label = sign->GetObservationLabels()[i];
        avg_px_error_sign += ReprojectionError(pts3D, frame, label);
        PlotReprojections(pts3D, frame, label);
    }

    return avg_px_error_sign /= sign->nObs;
}



float Mapping::ReprojectionError(std::vector<cv::Point3f> pts3D, Sign* sign)
{
    float avg_error = 0.0;

    for(int i=0; i<pts3D.size(); i++)
    {
        Point3f Pw = pts3D[i];
        Mat Pc_ = sign->GetFirstFrame()->GetRotation()*Converter::toCvMat(Pw) + sign->GetFirstFrame()->GetTranslation();
        Point3f Pc = Converter::toCvPoint3f(Pc_);
        
        //to pixel frame
        float fx = sign->GetFirstFrame()->fx;
        float fy = sign->GetFirstFrame()->fy;
        float cx = sign->GetFirstFrame()->cx;
        float cy = sign->GetFirstFrame()->cy;

        Point2f px = Utils::Cam2Px(Pc, fx, fy, cx, cy);
        avg_error+=norm(px-sign->GetAllSeeds()[i]->mpt);
    }

    return avg_error/=pts3D.size();
}



float Mapping::ReprojectionError(std::vector<cv::Point3f> pts3D, Frame* frame, Frame::SignLabel* label)
{
    float avg_error = 0.0;

    for(int i=0; i<pts3D.size(); i++)
    {
        Point3f Pw = pts3D[i];
        Mat Pc_ = frame->GetRotation()*Converter::toCvMat(Pw) + frame->GetTranslation();
        Point3f Pc = Converter::toCvPoint3f(Pc_);
        
        //to pixel frame
        float fx = frame->fx;
        float fy = frame->fy;
        float cx = frame->cx;
        float cy = frame->cy;

        Point2f px = Utils::Cam2Px(Pc, fx, fy, cx, cy);
        avg_error+=norm(px-label->Vertices[i]);
    }

    return avg_error/=pts3D.size();
}




void Mapping::PlotLabel(Frame* frame, Frame::SignLabel* label)
{
    Mat img = frame->GetImg();
    cv::circle(img, label->Vertices[0], 3, Scalar(0,0,255), -1);
    cv::circle(img, label->Vertices[1], 3, Scalar(0,0,255), -1);
    cv::circle(img, label->Vertices[2], 3, Scalar(0,0,255), -1);
    cv::circle(img, label->Vertices[3], 3, Scalar(0,0,255), -1);
    imshow("img", img);
    waitKey(0);
}

void Mapping::PlotSign(Sign* sign)
{
    Mat img =  sign->GetFirstFrame()->GetImg();
    cv::circle(img, sign->GetAllSeeds()[0]->mpt, 2, Scalar(255,0,0), -1);
    cv::circle(img, sign->GetAllSeeds()[1]->mpt, 2, Scalar(255,0,0), -1);
    cv::circle(img, sign->GetAllSeeds()[2]->mpt, 2, Scalar(255,0,0), -1);
    cv::circle(img, sign->GetAllSeeds()[3]->mpt, 2, Scalar(255,0,0), -1);
    imshow("img", img);
    waitKey(0);
}

void Mapping::PlotSignObs(Sign* sign)
{
    for(int i=0; i<sign->nObs; i++)
    {
        Frame* frame = sign->GetObservationFrames()[i];
        Frame::SignLabel* label = sign->GetObservationLabels()[i];
        PlotLabel(frame, label);
    }
}

void Mapping::PlotMappoints(std::vector<MapPoint*> MapPoints)
{
    Mat img =  MapPoints[0]->GetFirstFrame()->GetImg();
    for(int i=0; i<MapPoints.size(); i++)
    {
        cv::circle(img, MapPoints[i]->GetSeed()->mpt, 2, Scalar(255,0,0), -1);
    }
    imshow("img", img);
    waitKey(0);
}

void Mapping::PlotMappointObs(MapPoint* mappoint)
{
    for(int i=0; i<mappoint->nObs; i++)
    {
        Mat img = mappoint->GetObservationFrames()[i]->GetImg();
        Point2f px = mappoint->GetObservationPxs()[i];
        cv::circle(img, px, 2, Scalar(255,0,0), -1);
        imshow("img", img);
        waitKey(0);
    }
}








void Mapping::PlotReprojections(vector<Point3f> pts3D, Sign* sign)
{
    Mat img = sign->GetFirstFrame()->GetImg();

    for(int i=0; i<pts3D.size(); i++)
    {
        Point3f Pw = pts3D[i];
        Mat Pc_ = sign->GetFirstFrame()->GetRotation()*Converter::toCvMat(Pw) + sign->GetFirstFrame()->GetTranslation();
        Point3f Pc = Converter::toCvPoint3f(Pc_);
        
        //to pixel frame
        float fx = sign->GetFirstFrame()->fx;
        float fy = sign->GetFirstFrame()->fy;
        float cx = sign->GetFirstFrame()->cx;
        float cy = sign->GetFirstFrame()->cy;

        Point2f px = Utils::Cam2Px(Pc, fx, fy, cx, cy);

        cv::circle(img, px, 2, Scalar(0,0,255), -1);
        cv::circle(img, sign->GetAllSeeds()[i]->mpt, 2, Scalar(255,0,0), -1);
    }
    imshow("img", img);
    waitKey(0);
}

void Mapping::PlotReprojections(std::vector<cv::Point3f> pts3D, Frame* frame, Frame::SignLabel* label)
{
    Mat img = frame->GetImg();

    for(int i=0; i<pts3D.size(); i++)
    {
        Point3f Pw = pts3D[i];
        Mat Pc_ = frame->GetRotation()*Converter::toCvMat(Pw) + frame->GetTranslation();
        Point3f Pc = Converter::toCvPoint3f(Pc_);
        
        //to pixel frame
        float fx = frame->fx;
        float fy = frame->fy;
        float cx = frame->cx;
        float cy = frame->cy;

        Point2f px = Utils::Cam2Px(Pc, fx, fy, cx, cy);

        cv::circle(img, px, 2, Scalar(0,0,255), -1);
        cv::circle(img, label->Vertices[i], 2, Scalar(255,0,0), -1);
    }
    imshow("img", img);
    waitKey(0);
}

void Mapping::PlotReprojections(const std::vector<cv::Point3f> pts3D, Frame* frame, const std::vector<cv::Point2f> pts)
{

    Mat img = frame->GetImg();

    for(int i=0; i<pts3D.size(); i++)
    {
        Point3f Pw = pts3D[i];
        Mat Pc_ = frame->GetRotation()*Converter::toCvMat(Pw) + frame->GetTranslation();
        Point3f Pc = Converter::toCvPoint3f(Pc_);
        
        float fx = frame->fx;
        float fy = frame->fy;
        float cx = frame->cx;
        float cy = frame->cy;

        Point2f pt_repro = Utils::Cam2Px(Pc, fx, fy, cx, cy);
        
        cv::circle(img, pt_repro, 2, Scalar(0,0,255), -1);
        cv::circle(img, pts[i], 2, Scalar(255,0,0), -1);
    }
    imshow("img", img);
    waitKey(0);
}
