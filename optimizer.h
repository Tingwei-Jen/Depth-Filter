#pragma once
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "frame.h"
#include "sign.h"

//opencv
#include <opencv2/opencv.hpp>

//g2o
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"

#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
//#include "g2o/solvers/csparse/linear_solver_csparse.h"

#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"


class Optimizer
{
public:
	static void BundleAdjustment(cv::Mat& Tcw1, cv::Mat& Tcw2, std::vector<cv::Point3f>& pts3d, std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2, int nIterations = 10);
	static void BundleAdjustment(std::vector<Frame*> Frames, std::vector<Sign*> Signs, 
								std::vector<cv::Mat>& New_poses, std::vector<cv::Point3f>& New_pts3D, int nIterations = 10);

	static void LocalBundleAdjustment(Sign* sign, Frame* frame, Frame::SignLabel* label, std::vector<cv::Point3f>& pts3d, int nIterations = 10);
	static void LocalBundleAdjustment(Sign* sign, int nIterations = 10);





	static void BundleAdjustmentTwoFrames(Frame* frame1, Frame* frame2, std::vector<cv::Point3f>& pts3d, 
										const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, int nIterations = 10);

};
#endif //OPTIMIZER_H