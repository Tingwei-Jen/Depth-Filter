#pragma once
#ifndef MAPPING_H
#define MAPPING_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

struct Seed
{
    static int batch_counter;
    static int seed_counter;

    int batch_id;                //!< Batch id is the id of the frame for which the seed was created.
    int id;                      //!< Seed ID
    Point2d pt;                  //!< feature's pixel location.
    Point3d f;                   //!< unit vector of feature in camera frame
    double a;                    //!< a of Beta distribution: When high, probability of inlier is large.
    double b;                    //!< b of Beta distribution: When high, probability of outlier is large.
    double mu;                   //!< Mean of normal distribution.
    double z_range;              //!< Max range of the possible depth.
    double sigma2;               //!< Variance of normal distribution.
    Seed(Point2d pt, Point3d f, double depth_mean, double depth_min);
};


class Mapping
{
public:
    Mapping(Mat K);
    ~Mapping();

    //init seeds of frame
    void initSeeds(const std::vector<Point2d>& pts, const double& frame_mean_dpeth, const double& frame_min_depth, std::vector<Seed>& seeds);
    //update seed's depth_mean and its covariance
    bool updateSeed(const Mat& T_w_ref, const Mat& T_w_cur, const Point2d& pt_ref, const Point2d& pt_cur, Seed& seed);
    //reproject seed to cur frame
    void reproSeed(const Seed& seed, const Mat& T_w_ref, const Mat& T_w_cur, Point2d& pt_cur_repro);
    //reprojection 3d points from ref to cur
    void repro3dPoint(const Point3d& pt_w, const Mat& T_c_w, Point2d& pt_c_repro);
    //compute reprojection error
    double reproError(const Point2d& pt, const Point2d& pt_repro);
    //compute distance between camera centers
    double getCentersDist(const Mat& T_w_ref, const Mat& T_w_cur);
    //get cur image camera center relative to ref
    Point3d getCurCenter(const Mat& T_w_ref, const Mat& T_w_cur);


    //compute Homography matrix
    Mat computeH(const std::vector<Point2d>& pts_ref, const std::vector<Point2d>& pts_cur);
    void perspectiveTransform(const Mat& H, const std::vector<Point2d>& pts_src, std::vector<Point2d>& pts_dst);
    //use PCA method to find planar
    void findPlane(const std::vector<Point3d>& pts3D, Point3d& normal_vector, double& d);
    //find intersect points between plane and lines from camera center.
    void calPlaneLinesIntersectPoints(const Point3d& camera_center, const std::vector<Point3d>& fs, const Point3d& normal_vector, 
                                        const double& d, std::vector<Point3d>& intersect_points);
    void decomposeH(const Mat& H, Mat& T);




    //compute epipole in current image
    void getEpipole(const Mat& T_w_ref, const Mat& T_w_cur, Point2d& epipole);
    //compute epipolar line by F
    void getEpipolarLine(const Point2d& pt_ref, const Mat& T_w_ref, const Mat& T_w_cur, Point3d& line_cur);
    //compute distant between point and line
    void distPoint2Line(const Point2d& pt_cur, const Point3d& line_cur, double& dist);
    //find vertical line
    bool verticalLine(const Point3d& line, const Point2d& point_through, Point3d& verticle_line);

    // //compute line segment (possible 3D points location) in current frame.
    // void epipolarSegment(const Mat& T_w_ref, const Mat& T_w_cur, const Point2d& pt_ref, const double& depth_mu, const double& depth_std, std::vector<Point2d>& pts_cur_temp);
    // //search best current match points according similarity along epipolar line
    // bool epipolarSearch(Mat ref, Mat cur, Point2d pt_ref, std::vector<Point2d> pts_cur_temp, int window_size, Point2d& best_pt_cur);
    
    bool inside(const cv::Point2d& px, const int& width, const int& height, const int& border); 

private:
    //compute 3D point based on reference camera frame
    bool triangulation(const Mat& T_w_ref, const Mat& T_w_cur, const Point2d& pt_ref, const Point2d& pt_cur, Point3d& x3Dp);
    //compute observed depth std
    void computeTau(const Mat& T_ref_cur, const Point3d& P, const double& px_error_angle, double& tau);
    //gaussian&beta distribution fusion,  (mu, sigma2) combine with (x, tau2)(observation)
    void updateFilter(const double& x, const double& tau2, Seed& seed);   
    
    Point3d px2cam (Point2d px);
    Point2d cam2px (Point3d p_cam);
   
    void computeF(Mat T_w_ref, Mat T_w_cur, Mat& F);
    //double NCC(Mat ref, Mat cur, Point pt_ref, Point2d pt_cur, int window_size);
   
private:
    Mat m_K; 
    double m_fx;
    double m_fy;
    double m_cx;
    double m_cy;

};
#endif //MAPPING_H