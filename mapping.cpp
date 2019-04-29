#include "mapping.h"
#include <Eigen/Dense>
#include <boost/math/distributions/normal.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int Seed::batch_counter = 0;
int Seed::seed_counter = 0;

Seed::Seed(Point2d pt, Point3d f, double depth_mean, double depth_min) :
    batch_id(batch_counter),
    id(seed_counter++),
    pt(pt),
    f(f),
    a(10),
    b(10),
    mu(1.0/depth_mean),
    z_range(1.0/depth_min),
    sigma2(z_range*z_range/36)
{
}


Mapping::Mapping(cv::Mat K)
{

    cout<<"Construct Mapping!!"<<endl;

    this->m_K = K;
    this->m_fx = K.at<double>(0,0);
    this->m_fy = K.at<double>(1,1);
    this->m_cx = K.at<double>(0,2);
    this->m_cy = K.at<double>(1,2);

}


Mapping::~Mapping()
{

    cout<<"Destruct Mapping!!"<<endl;

}


void Mapping::initSeeds(const std::vector<Point2d>& pts, const double& frame_mean_dpeth, const double& frame_min_depth, std::vector<Seed>& seeds)
{

    for(int i=0; i<pts.size(); i++)
    {
        Point3d pt_cam = px2cam(pts[i]);
        seeds.push_back(Seed(pts[i], pt_cam/norm(pt_cam), frame_mean_dpeth, frame_min_depth));
    }

    Seed::batch_counter++;

}


bool Mapping::updateSeed(const Mat& T_w_ref, const Mat& T_w_cur, const Point2d& pt_ref, const Point2d& pt_cur, Seed& seed)
{

    Point3d x3Dp;
    bool ret = triangulation(T_w_ref, T_w_cur, pt_ref, pt_cur, x3Dp);

    if(!ret)
        return false;

    Mat T_ref_cur = T_w_ref.inv() * T_w_cur;
    double px_noise = 1.0;
    double px_error_angle = atan(px_noise/(2.0*this->m_fx))*2.0; // law of chord (sehnensatz)
    double tau;
    computeTau(T_ref_cur, x3Dp, px_error_angle, tau);
    
    double z = norm(x3Dp);
    double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));
    
    updateFilter(1./z, tau_inverse*tau_inverse, seed);

}


void Mapping::reproSeed(const Seed& seed, const Mat& T_w_ref, const Mat& T_w_cur, Point2d& pt_cur_repro)
{

    Point3d p = seed.f * (1/seed.mu);
    Mat p_ = ( Mat_<double> ( 3,1 ) << p.x, p.y, p.z); 
    
    //R_cur_ref, t_cur_ref
    Mat T_cur_ref = T_w_cur.inv() * T_w_ref;
    Mat R_cur_ref, t_cur_ref;
    T_cur_ref.colRange(0,3).rowRange(0,3).copyTo(R_cur_ref);
    T_cur_ref.col(3).rowRange(0,3).copyTo(t_cur_ref);

    Mat p_cur_ = R_cur_ref * p_ + t_cur_ref;
    Point3d p_cur = Point3d(p_cur_.at<double>(0), p_cur_.at<double>(1), p_cur_.at<double>(2));
    pt_cur_repro = cam2px(p_cur);

}

void Mapping::repro3dPoint(const Point3d& pt_w, const Mat& T_c_w, Point2d& pt_c_repro)
{

    Mat pt_w_ = ( Mat_<double> ( 3,1 ) << pt_w.x, pt_w.y, pt_w.z); 
    Mat R_c_w, t_c_w;
    T_c_w.colRange(0,3).rowRange(0,3).copyTo(R_c_w);
    T_c_w.col(3).rowRange(0,3).copyTo(t_c_w);

    Mat pt_c_ = R_c_w * pt_w_ + t_c_w;
    Point3d pt_c = Point3d(pt_c_.at<double>(0), pt_c_.at<double>(1), pt_c_.at<double>(2));
    pt_c_repro = cam2px(pt_c);

}

double Mapping::reproError(const Point2d& pt, const Point2d& pt_repro)
{
    return norm(pt-pt_repro);
}


double Mapping::getCentersDist(const Mat& T_w_ref, const Mat& T_w_cur)
{

    Mat T_ref_cur = T_w_ref.inv() * T_w_cur;

    return sqrt(T_ref_cur.at<double>(0,3)*T_ref_cur.at<double>(0,3) 
                + T_ref_cur.at<double>(1,3)*T_ref_cur.at<double>(1,3) 
                + T_ref_cur.at<double>(2,3)*T_ref_cur.at<double>(2,3));
}


Point3d Mapping::getCurCenter(const Mat& T_w_ref, const Mat& T_w_cur)
{

    Mat T_ref_cur = T_w_ref.inv() * T_w_cur;
    return Point3d(T_ref_cur.at<double>(0,3), T_ref_cur.at<double>(1,3), T_ref_cur.at<double>(2,3));
}


//compute Homography matrix
//x2 = H*x1
//[0   -x1  v2*x1]
//[x1  0   -u2*x1]
Mat Mapping::computeH(const vector<Point2d>& pts_ref, const vector<Point2d>& pts_cur)
{
    
    const int N = pts_ref.size();

    cv::Mat A(2*N,9,CV_64F);

    for(int i=0; i<N; i++)
    {
        const double u1 = pts_ref[i].x;
        const double v1 = pts_ref[i].y;
        const double u2 = pts_cur[i].x;
        const double v2 = pts_cur[i].y;

        A.at<double>(2*i,0) = 0.0;
        A.at<double>(2*i,1) = 0.0;
        A.at<double>(2*i,2) = 0.0;
        A.at<double>(2*i,3) = -u1;
        A.at<double>(2*i,4) = -v1;
        A.at<double>(2*i,5) = -1;
        A.at<double>(2*i,6) = v2*u1;
        A.at<double>(2*i,7) = v2*v1;
        A.at<double>(2*i,8) = v2;

        A.at<double>(2*i+1,0) = u1;
        A.at<double>(2*i+1,1) = v1;
        A.at<double>(2*i+1,2) = 1;
        A.at<double>(2*i+1,3) = 0.0;
        A.at<double>(2*i+1,4) = 0.0;
        A.at<double>(2*i+1,5) = 0.0;
        A.at<double>(2*i+1,6) = -u2*u1;
        A.at<double>(2*i+1,7) = -u2*v1;
        A.at<double>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}


void Mapping::perspectiveTransform(const Mat& H, const std::vector<Point2d>& pts_src, std::vector<Point2d>& pts_dst)
{

    pts_dst.clear();

    for(int i=0; i<pts_src.size(); i++)
    {
    
        Mat pt_src = ( Mat_<double> ( 3,1 ) << pts_src[i].x, pts_src[i].y, 1); 
        Mat pt_dst = H * pt_src;
        pts_dst.push_back(Point2d(pt_dst.at<double>(0)/pt_dst.at<double>(2), pt_dst.at<double>(1)/pt_dst.at<double>(2)));
    
    }

}


void Mapping::findPlane(const std::vector<Point3d>& pts3D, Point3d& normal_vector, double& d)
{
    
    const int N = pts3D.size();
    
    Mat A(N,3,CV_64F);
    Point3d mean = Point3d(0,0,0);

    for(int i=0; i<N; i++)
    {
        A.at<double>(i,0) = pts3D[i].x;
        A.at<double>(i,1) = pts3D[i].y;
        A.at<double>(i,2) = pts3D[i].z;

        mean.x += pts3D[i].x;
        mean.y += pts3D[i].y;
        mean.z += pts3D[i].z;
    }

    mean /= N;

    Mat _mean(N,3,CV_64F);

    for(int i=0; i<N; i++)
    {
        _mean.at<double>(i,0) = mean.x;
        _mean.at<double>(i,1) = mean.y;
        _mean.at<double>(i,2) = mean.z;
    }

    A = A-_mean; 

    Mat sigma = A.t() * A / N;

    cv::Mat u,w,vt;
    cv::SVD::compute(sigma,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

    Point3d u1 = Point3d(u.at<double>(0,0), u.at<double>(1,0), u.at<double>(2,0));
    Point3d u2 = Point3d(u.at<double>(0,1), u.at<double>(1,1), u.at<double>(2,1));

    normal_vector = u1.cross(u2);

    //ax+by+cz+d = 0;
    d = -mean.dot(normal_vector);
}


void Mapping::calPlaneLinesIntersectPoints(const Point3d& camera_center, const std::vector<Point3d>& fs, const Point3d& normal_vector, 
                                                const double& d, std::vector<Point3d>& intersect_points)
{
    intersect_points.clear();

    for(int i=0; i<fs.size(); i++)
    {
     
        double t = -(camera_center.dot(normal_vector)+d)/(fs[i].dot(normal_vector));
        intersect_points.push_back(camera_center + t*fs[i]);
    
    }
}

void Mapping::getEpipole(const Mat& T_w_ref, const Mat& T_w_cur, Point2d& epipole)
{

    Mat T_cur_w = T_w_cur.inv();	
    Mat R_cur_w, t_cur_w;
	T_cur_w.colRange(0,3).rowRange(0,3).copyTo(R_cur_w);
	T_cur_w.col(3).rowRange(0,3).copyTo(t_cur_w);

    // camera center of first image in world frame
    Mat Cw1 = ( Mat_<double> ( 3,1 ) << T_w_ref.at<double>(0,3), T_w_ref.at<double>(1,3), T_w_ref.at<double>(2,3)); 
    // camera center of first image in cur camera frame
    Mat C2 = R_cur_w * Cw1 + t_cur_w;

    epipole = cam2px(Point3d(C2.at<double>(0), C2.at<double>(1), C2.at<double>(2)));

}


void Mapping::getEpipolarLine(const Point2d& pt_ref, const Mat& T_w_ref, const Mat& T_w_cur, Point3d& line_cur)
{

    Mat F;
    computeF(T_w_ref, T_w_cur, F);
	Mat pt_ref_ = ( Mat_<double> ( 3,1 ) <<  pt_ref.x, pt_ref.y, 1 );
	Mat line_cur_ = F * pt_ref_;
    line_cur = Point3d(line_cur_.at<double>(0,0), line_cur_.at<double>(1,0), line_cur_.at<double>(2,0));

}


void Mapping::distPoint2Line(const Point2d& pt_cur, const Point3d& line_cur, double& dist)
{

	dist = abs(pt_cur.x*line_cur.x + pt_cur.y*line_cur.y + line_cur.z)/sqrt(line_cur.x*line_cur.x+line_cur.y*line_cur.y);	

}


bool Mapping::verticalLine(const Point3d& line, const Point2d& point_through, Point3d& verticle_line)
{

    double a = line.x;
    double b = line.y;
    double c = line.z;

    if(a!=0 && b!=0)
    {
        double m1 = -a/b;
        double m2 = -1/m1;

        double y_intercept = point_through.y - m2*point_through.x;

        verticle_line = Point3d(m2,-1,y_intercept);
    }
    else if (a==0)  
    {
        verticle_line = Point3d(1,0,-point_through.x);
    }
    else if (b==0)
    {
        verticle_line = Point3d(0,1,-point_through.y);
    }
}


// void Mapping::epipolarSegment(const Mat& T_w_ref, const Mat& T_w_cur, const Point2d& pt_ref, const double& depth_mu, const double& depth_std, 
//                             std::vector<Point2d>& pts_cur_temp)
// {

//     pts_cur_temp.clear();
    
//     //R_cur_ref, t_cur_ref
//     Mat T_cur_ref = T_w_cur.inv() * T_w_ref;
//     Mat R_cur_ref, t_cur_ref;
//     T_cur_ref.colRange(0,3).rowRange(0,3).copyTo(R_cur_ref);
//     T_cur_ref.col(3).rowRange(0,3).copyTo(t_cur_ref);

//     //pixel frame to camera frame, z = 1
//     Point3d pt_ref_cam = px2cam(pt_ref);

//     //d_min and d_max
//     double d_min = depth_mu - 3 * depth_std;
// 	double d_max = depth_mu + 3 * depth_std;

// 	if ( d_min<0.1 ) 
// 		d_min = 0.1;


//     // different depth
//     Point3d pt_ref_cam_mean = pt_ref_cam * depth_mu;
//     Point3d pt_ref_cam_max = pt_ref_cam * d_max;
//     Point3d pt_ref_cam_min = pt_ref_cam * d_min;
//     Mat pt_ref_cam_mean_ = ( Mat_<double> ( 3,1 ) << pt_ref_cam_mean.x, pt_ref_cam_mean.y, pt_ref_cam_mean.z); 
//     Mat pt_ref_cam_max_ = ( Mat_<double> ( 3,1 ) << pt_ref_cam_max.x, pt_ref_cam_max.y, pt_ref_cam_max.z); 
//     Mat pt_ref_cam_min_ = ( Mat_<double> ( 3,1 ) << pt_ref_cam_min.x, pt_ref_cam_min.y, pt_ref_cam_min.z); 

//     // change to cur camera frame
//     Mat pt_cur_cam_mean_ = R_cur_ref * pt_ref_cam_mean_ + t_cur_ref;
//     Mat pt_cur_cam_max_ = R_cur_ref * pt_ref_cam_max_ + t_cur_ref;
//     Mat pt_cur_cam_min_ = R_cur_ref * pt_ref_cam_min_ + t_cur_ref;
//     Point3d pt_cur_cam_mean = Point3d(pt_cur_cam_mean_.at<double>(0), pt_cur_cam_mean_.at<double>(1), pt_cur_cam_mean_.at<double>(2));
//     Point3d pt_cur_cam_max = Point3d(pt_cur_cam_max_.at<double>(0), pt_cur_cam_max_.at<double>(1), pt_cur_cam_max_.at<double>(2));
//     Point3d pt_cur_cam_min = Point3d(pt_cur_cam_min_.at<double>(0), pt_cur_cam_min_.at<double>(1), pt_cur_cam_min_.at<double>(2));

//     // change to pixel frame 
//     Point2d pt_cur_mean = cam2px(pt_cur_cam_mean);
//     Point2d pt_cur_max = cam2px(pt_cur_cam_max);
//     Point2d pt_cur_min = cam2px(pt_cur_cam_min);

//     //length and direction of epipolar line
//     Point2d epipolar_line = pt_cur_max-pt_cur_min;
//     double half_length = 0.5 * sqrt(epipolar_line.x*epipolar_line.x + epipolar_line.y*epipolar_line.y);   
//     if (half_length>100)
//         half_length = 100;
//     Point2d epipolar_direction = epipolar_line/(2*half_length);     

//     for(double l = -half_length; l<half_length; l+=0.7)
//     {
//         Point2d pt_cur = pt_cur_mean + l*epipolar_direction;
//         if ( !inside(pt_cur) )
//             continue; 

//         pts_cur_temp.push_back(pt_cur);
//     }

// }


// bool Mapping::epipolarSearch(Mat ref, Mat cur, Point2d pt_ref, vector<Point2d> pts_cur_temp, int window_size, Point2d& best_pt_cur)
// {

//     //compute NCC score
//     double best_score = -1.0;
//     Point2d best_pt_cur_;    

//     for(int i=0; i<pts_cur_temp.size(); i++)
//     {
//         double score = NCC(ref, cur, pt_ref, pts_cur_temp[i], window_size);

//         if(best_score<score)
//         {  
//             best_score = score;
//             best_pt_cur_ = pts_cur_temp[i];
//         }
//     }    
//     if(best_score<0.7)
//         return false;

//     best_pt_cur = best_pt_cur_;
//     return true;
// }



/*
private
*/
bool Mapping::triangulation(const Mat& T_w_ref, const Mat& T_w_cur, const Point2d& pt_ref, const Point2d& pt_cur, Point3d& x3Dp)
{

    //pixel frame to camera frame
    Point3d pt_ref_cam = px2cam(pt_ref);
    Point3d pt_cur_cam = px2cam(pt_cur);

    Mat T_ref_ref = Mat::eye(4, 4, CV_64F);    
    Mat T_cur_ref = T_w_cur.inv() * T_w_ref;

    // Linear Triangulation Method
    cv::Mat A(4,4, CV_64F);
    A.row(0) = pt_ref_cam.x*T_ref_ref.row(2)-T_ref_ref.row(0);
    A.row(1) = pt_ref_cam.y*T_ref_ref.row(2)-T_ref_ref.row(1);
    A.row(2) = pt_cur_cam.x*T_cur_ref.row(2)-T_cur_ref.row(0);
    A.row(3) = pt_cur_cam.y*T_cur_ref.row(2)-T_cur_ref.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    cv::Mat x3D = vt.row(3).t();

    if(x3D.at<double>(3) == 0)
        return false;
    x3D = x3D.rowRange(0,3)/x3D.at<double>(3);  
    
    // check triangulation in front of camera.
    cv::Mat R_cur_ref, t_cur_ref;
    T_cur_ref.colRange(0,3).copyTo(R_cur_ref);
    T_cur_ref.col(3).copyTo(t_cur_ref);
    cv::Mat x3Dt = x3D.t();

    double z1 = x3Dt.at<double>(2);
    if(z1<=0)
        return false;
        
    double z2 = R_cur_ref.row(2).dot(x3Dt)+t_cur_ref.at<double>(2);
    if(z2<=0)
        return false;

    x3Dp = Point3d(x3D.at<double>(0), x3D.at<double>(1), x3D.at<double>(2));

}


void Mapping::computeTau(const Mat& T_ref_cur, const Point3d& P, const double& px_error_angle, double& tau)
{

    Point3d t = Point3d(T_ref_cur.at<double>(0,3), T_ref_cur.at<double>(1,3), T_ref_cur.at<double>(2,3));
    Point3d a = P-t;

    double apha = acos( (P.x*t.x + P.y*t.y + P.z*t.z) / (norm(P)*norm(t)) ); 
    double beta = acos( -(a.x*t.x + a.y*t.y + a.z*t.z) / (norm(a)*norm(t)) );
    double beta_plus = beta + px_error_angle;
    double gamma = CV_PI - apha - beta_plus;
    double P_plus_norm = norm(t)*sin(beta_plus)/sin(gamma);

    tau = P_plus_norm - norm(P);

}


void Mapping::updateFilter(const double& x, const double& tau2, Seed& seed)
{

    double norm_scale = sqrt(seed.sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;
    boost::math::normal_distribution<double> nd(seed.mu, norm_scale);
    double s2 = 1./(1./seed.sigma2 + 1./tau2);
    double m = s2*(seed.mu/seed.sigma2 + x/tau2);
    double C1 = seed.a/(seed.a+seed.b) * boost::math::pdf(nd, x);
    double C2 = seed.b/(seed.a+seed.b) * 1./seed.z_range;
    double normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    double f = C1*(seed.a+1.)/(seed.a+seed.b+1.) + C2*seed.a/(seed.a+seed.b+1.);
    double e = C1*(seed.a+1.)*(seed.a+2.)/((seed.a+seed.b+1.)*(seed.a+seed.b+2.))
            + C2*seed.a*(seed.a+1.0f)/((seed.a+seed.b+1.0f)*(seed.a+seed.b+2.0f));
   
    // update parameters
    double mu_new = C1*m+C2*seed.mu;
    seed.sigma2 = C1*(s2 + m*m) + C2*(seed.sigma2 + seed.mu*seed.mu) - mu_new*mu_new;
    seed.mu = mu_new;
    seed.a = (e-f)/(f-e/f);
    seed.b = seed.a*(1.0f-f)/f;

} 


cv::Point3d Mapping::px2cam (cv::Point2d px)
{
    return Point3d((px.x-this->m_cx)/this->m_fx,(px.y-this->m_cy)/this->m_fy,1);
}


cv::Point2d Mapping::cam2px (cv::Point3d p_cam)
{
    return Point2d(p_cam.x*this->m_fx/p_cam.z + this->m_cx, p_cam.y*this->m_fy/p_cam.z + this->m_cy);
}


bool Mapping::inside(const cv::Point2d& px, const int& width, const int& height, const int& border) {

    return px.x >= border && px.y>=border 
        && px.x<(width-border) && px.y < (height-border);
}


void Mapping::computeF(cv::Mat T_w_ref, cv::Mat T_w_cur, cv::Mat& F)
{

    //compute Tranformation from cur to ref
    Mat T_cur_ref = T_w_cur.inv() * T_w_ref;

    Mat R_cur_ref, t_cur_ref;
    T_cur_ref.colRange(0,3).rowRange(0,3).copyTo(R_cur_ref);
    T_cur_ref.col(3).rowRange(0,3).copyTo(t_cur_ref);

    Mat t_cur_ref_x = ( Mat_<double> ( 3,3 ) <<
                0,                            -t_cur_ref.at<double> (2),   t_cur_ref.at<double> (1),
                t_cur_ref.at<double> (2),     0,                           -t_cur_ref.at<double> (0),
                -t_cur_ref.at<double> (1),    t_cur_ref.at<double> (0),    0);

    //essential matrix and fundamental matrix
    Mat E = t_cur_ref_x * R_cur_ref;
    F = this->m_K.inv().t() * E * this->m_K.inv();

    //clean up F
    Eigen::MatrixXf f(3,3);
    f(0,0) = F.at<double>(0,0); f(0,1) = F.at<double>(0,1); f(0,2) = F.at<double>(0,2);
    f(1,0) = F.at<double>(1,0); f(1,1) = F.at<double>(1,1); f(1,2) = F.at<double>(1,2);
    f(2,0) = F.at<double>(2,0); f(2,1) = F.at<double>(2,1); f(2,2) = F.at<double>(2,2);

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(f, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
    Eigen::MatrixXf singular_values = svd.singularValues();
    Eigen::MatrixXf left_singular_vectors = svd.matrixU();
    Eigen::MatrixXf right_singular_vectors = svd.matrixV();
    
    Eigen::MatrixXf d(3,3);
    d(0,0) = singular_values(0); d(0,1) = 0;                  d(0,2) = 0;
    d(1,0) = 0;                  d(1,1) = singular_values(1); d(1,2) = 0;
    d(2,0) = 0;                  d(2,1) = 0;                  d(2,2) = 0;

    f = left_singular_vectors*d*right_singular_vectors.transpose();
    F = ( Mat_<double> ( 3,3 ) << f(0,0), f(0,1), f(0,2), f(1,0), f(1,1), f(1,2), f(2,0), f(2,1), f(2,2));

}


// double Mapping::NCC(cv::Mat ref, cv::Mat cur, cv::Point pt_ref, cv::Point2d pt_cur, int window_size)
// {

//     //get mean
//     double ref_mean = 0;
//     double cur_mean = 0;
//     vector<double> ref_vals;
//     vector<double> cur_vals;

//     for(int i=-window_size; i<window_size; i++)
//     {
//         for(int j=-window_size; j<window_size; j++)
//         {
//             //reference
//             double ref_val = ref.at<uchar>(pt_ref.y+j, pt_ref.x+i)/255.0;
//             ref_mean += ref_val;
//             ref_vals.push_back(ref_val);

//             //current
//             double cur_val = cur.at<uchar>(int(pt_cur.y+j), int(pt_cur.x+i))/255.0;
//             cur_mean += cur_val;
//             cur_vals.push_back(cur_val);
//         }
//     }
//     ref_mean = ref_mean/ref_vals.size();
//     cur_mean = cur_mean/cur_vals.size();
   
//     //zero mean NCC
//     double numerator = 0; 
//     double demoniator1 = 0;
//     double demoniator2 = 0;

//     for( int i=0; i<ref_vals.size(); i++)
//     {
//         double a = (ref_vals[i] - ref_mean) * (cur_vals[i] - cur_mean);
//         numerator += a;
//         demoniator1 += (ref_vals[i] - ref_mean) * (ref_vals[i] - ref_mean);
//         demoniator2 += (cur_vals[i] - cur_mean) * (cur_vals[i] - cur_mean);
//     }
//     return numerator / sqrt( demoniator1*demoniator2+1e-10 );   // 防止分母出现零
// }
