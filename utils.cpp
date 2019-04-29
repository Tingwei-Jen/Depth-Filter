#include "utils.h"

#include <Eigen/Dense>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

cv::Mat Utils::Quaternion2RotM(float x, float y, float z, float w)
{
    float xx = x*x;
    float yy = y*y;
    float zz = z*z;

    float wx = w*x;
    float wy = w*y;
    float wz = w*z;

    float xy = x*y;
    float xz = x*z;
    float yz = y*z;

    Mat R = ( Mat_<float> ( 3,3 ) <<
                ( 1 - 2*yy - 2*zz ), ( 2*xy - 2*wz ), ( 2*xz + 2*wy ),
                ( 2*xy + 2*wz ), ( 1 - 2*xx - 2*zz ), ( 2*yz - 2*wx ),
                ( 2*xz - 2*wy ), ( 2*yz + 2*wx ), ( 1 - 2*xx - 2*yy ));

    return R;
}

void Utils::ReadVertices(const std::string csv_path, std::vector<std::string>& Imagenames, std::vector<std::vector<cv::Point2f>>& Verticess)
{

    Imagenames.clear();
    Verticess.clear();

    ifstream file(csv_path);
    std::string line;
    int row = 0;

    while(std::getline(file, line))
    {
    
        if(row==0)
        {
            row++;
            continue;       
        }

        stringstream ss(line);
        string str;

        int col = 0;
        int LTx0, LTy0, RTx0, RTy0, RBx0, RBy0, LBx0, LBy0;
        int LTx1, LTy1, RTx1, RTy1, RBx1, RBy1, LBx1, LBy1;
        while (getline(ss, str, ','))
        {
            if(col==0)
                Imagenames.push_back(str);
            else
            {
                std::stringstream convertor(str);
                double value;
                convertor >> value;

                if(col==1)
                    LTx0 = value;
                else if (col==2)
                    LTy0 = value;
                else if (col==3)
                    RTx0 = value;
                else if (col==4)
                    RTy0 = value;
                else if (col==5)
                    RBx0 = value;
                else if (col==6)
                    RBy0 = value;
                else if (col==7)
                    LBx0 = value;
                else if (col==8)
                    LBy0 = value;
                else if (col==9)
                    LTx1 = value;
                else if (col==10)
                    LTy1 = value;
                else if (col==11)
                    RTx1 = value;
                else if (col==12)
                    RTy1 = value;
                else if (col==13)
                    RBx1 = value;
                else if (col==14)
                    RBy1 = value;
                else if (col==15)
                    LBx1 = value;
                else if (col==16)
                    LBy1 = value;
            }

            col++;
        }

        std::vector<cv::Point2f> Vertices;
        Vertices.push_back(cv::Point2f(LTx0, LTy0));
        Vertices.push_back(cv::Point2f(RTx0, RTy0));
        Vertices.push_back(cv::Point2f(RBx0, RBy0));
        Vertices.push_back(cv::Point2f(LBx0, LBy0));
        if(LTx1!=-1)
        {
            Vertices.push_back(cv::Point2f(LTx1, LTy1));
            Vertices.push_back(cv::Point2f(RTx1, RTy1));
            Vertices.push_back(cv::Point2f(RBx1, RBy1));
            Vertices.push_back(cv::Point2f(LBx1, LBy1));
        }

        Verticess.push_back(Vertices);

        row++;
    }
}

std::vector<cv::Mat> Utils::ReadCameraPoses(const std::string csv_path, const std::vector<std::string> Imagenames)
{

    std::vector<cv::Mat> Twcs(Imagenames.size(),Mat());
    
    ifstream file(csv_path);
    std::string line;
    int row = 0;
    while(std::getline(file, line))
    {
        if(row==0)
        {
            row++;
            continue;       
        }

        stringstream ss(line);
        string str;
        
        int col = 0;
        double timesec, timensec;
        double posx, posy, posz;
        double qx, qy, qz, qw;
        while (getline(ss, str, ','))
        {
            std::stringstream convertor(str);
            double value;
            convertor >> value;


            if(col==2)
                timesec = value;
            else if(col==3)
                timensec = value;
            else if(col==6)
                posx = value;
            else if (col==7)
                posy = value;
            else if (col==8)
                posz = value;
            else if (col==9)
                qx = value;
            else if (col==10)
                qy = value;
            else if (col==11)
                qz = value;
            else if (col==12)
                qw = value;


            col++;
        }


        timensec = timensec/1000000000;
        timesec = timesec + timensec;
        string name =  std::to_string(timesec).substr(0,12) + ".png";

        for(int i=0; i<Imagenames.size(); i++)
        {
            if(name==Imagenames[i])
            {

                cv::Mat R = Quaternion2RotM(qx, qy, qz, qw);
                Mat T = (Mat_<float> (4,4) <<
                    R.at<float>(0,0), R.at<float>(0,1), R.at<float>(0,2), posx,
                    R.at<float>(1,0), R.at<float>(1,1), R.at<float>(1,2), posy,
                    R.at<float>(2,0), R.at<float>(2,1), R.at<float>(2,2), posz,
                                   0,                0,                0,     1
                );

                Twcs[i] = T;
            }
        }

        row++;        
    }

    return Twcs;
}

Point3f Utils::Px2Cam(Point2f px, float fx, float fy, float cx, float cy)
{
    Point3f p_cam = Point3f((px.x-cx)/fx,(px.y-cy)/fy,1);
    return p_cam;
}    

Point2f Utils::Cam2Px(Point3f p_cam, float fx, float fy, float cx, float cy)
{
    Point2f px = Point2f(p_cam.x*fx/p_cam.z + cx, p_cam.y*fy/p_cam.z + cy);
    return px;
}

Mat Utils::ComputeF21(const Mat& Tcw1, const Mat& Tcw2, const Mat& K)
{
    Mat T21 = Tcw2*Tcw1.inv();

    Mat R21, t21;
    T21.rowRange(0,3).colRange(0,3).copyTo(R21);
    T21.rowRange(0,3).col(3).copyTo(t21);
    
    Mat t21x = ( Mat_<float> ( 3,3 ) <<
                0,                      -t21.at<float> (2),   t21.at<float> (1),
                t21.at<float> (2),     0,                     -t21.at<float> (0),
                -t21.at<float> (1),    t21.at<float> (0),    0);

    Mat E = t21x * R21;

    Mat F = K.inv().t() * E * K.inv();

    //clean up F
    Eigen::MatrixXf f(3,3);
    f(0,0) = F.at<float>(0,0); f(0,1) = F.at<float>(0,1); f(0,2) = F.at<float>(0,2);
    f(1,0) = F.at<float>(1,0); f(1,1) = F.at<float>(1,1); f(1,2) = F.at<float>(1,2);
    f(2,0) = F.at<float>(2,0); f(2,1) = F.at<float>(2,1); f(2,2) = F.at<float>(2,2);

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(f, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
    Eigen::MatrixXf singular_values = svd.singularValues();
    Eigen::MatrixXf left_singular_vectors = svd.matrixU();
    Eigen::MatrixXf right_singular_vectors = svd.matrixV();
    
    Eigen::MatrixXf d(3,3);
    d(0,0) = singular_values(0); d(0,1) = 0;                  d(0,2) = 0;
    d(1,0) = 0;                  d(1,1) = singular_values(1); d(1,2) = 0;
    d(2,0) = 0;                  d(2,1) = 0;                  d(2,2) = 0;

    f = left_singular_vectors*d*right_singular_vectors.transpose();
    F = ( Mat_<float> ( 3,3 ) << f(0,0), f(0,1), f(0,2), f(1,0), f(1,1), f(1,2), f(2,0), f(2,1), f(2,2));

    return F;
}

// |xp2  - p0 |     |0|
// |yp2  - p1 | X = |0| ===> AX = 0
// |x'p2'- p0'|     |0|
// |y'p2'- p1'|     |0|
bool Utils::Triangulation(const Mat& Tcw1, const Mat& Tcw2, const Point3f& pt1_cam, const Point3f& pt2_cam, Point3f& x3Dp)
{

    Mat A(4,4, CV_32F);
    A.row(0) = pt1_cam.x*Tcw1.row(2)-Tcw1.row(0);
    A.row(1) = pt1_cam.y*Tcw1.row(2)-Tcw1.row(1);
    A.row(2) = pt2_cam.x*Tcw2.row(2)-Tcw2.row(0);
    A.row(3) = pt2_cam.y*Tcw2.row(2)-Tcw2.row(1);

    Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    Mat x3D = vt.row(3).t();

    if(x3D.at<float>(3) == 0)
        return false;
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);  

    // check triangulation in front of camera.
    Mat Rcw1, tcw1;
    Tcw1.rowRange(0,3).colRange(0,3).copyTo(Rcw1);
    Tcw1.rowRange(0,3).col(3).copyTo(tcw1);
    
    Mat Rcw2, tcw2;
    Tcw2.colRange(0,3).copyTo(Rcw2);
    Tcw2.col(3).copyTo(tcw2);

    Mat x3Dt = x3D.t();

    float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
    if(z1<=0)
        return false;
        
    float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
    if(z2<=0)
        return false;

    x3Dp = Point3f(x3D.at<float>(0), x3D.at<float>(1), x3D.at<float>(2));

    return true;
}
