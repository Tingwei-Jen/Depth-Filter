#include "optimizer.h"
#include "converter.h"
#include "utils.h"
#include <iostream>

using namespace std;

void Optimizer::BundleAdjustment(cv::Mat& Tcw1, cv::Mat& Tcw2, std::vector<cv::Point3f>& pts3d, std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2, int nIterations)
{

    //步骤1：初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    //linear solver
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    //linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();

    //algorithm, LM
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    optimizer.setAlgorithm(algorithm);


    //步骤2：准备相机参数
    float fx = 712.58465;
    float fy = 713.57849;
    float cx = 613.71289;
    float cy = 386.50469;

    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );


    //步骤3：向优化器添加顶点 set vertices: 
    //camera_pose ---> g2o::VertexSE3Expmap 
    //map_points ---> g2o::VertexSBAPointXYZ

    g2o::VertexSE3Expmap* v1 = new g2o::VertexSE3Expmap();
    v1->setId(0);        
    v1->setFixed(true);   
    v1->setEstimate(Converter::toSE3Quat(Tcw1));
    optimizer.addVertex(v1);
    
    g2o::VertexSE3Expmap* v2 = new g2o::VertexSE3Expmap();
    v2->setId(1);        
    v2->setEstimate(Converter::toSE3Quat(Tcw2));
    optimizer.addVertex(v2);


    for(int i=0; i<pts3d.size(); i++)
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId( 2 + i );
        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(pts3d[i].x,pts3d[i].y,pts3d[i].z));
        optimizer.addVertex( v );        
    }

    //步骤4：向优化器添加投影边边 
    //projection relationship ---> g2o::EdgeProjectXYZ2UV
    vector<g2o::EdgeProjectXYZ2UV*> edges;
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );
        edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }

    // 第二帧
    for ( size_t i=0; i<pts2.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(1)) );
        edge->setMeasurement( Eigen::Vector2d(pts2[i].x, pts2[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0,0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
    
    cout<<"开始优化"<<endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    cout<<"优化完毕"<<endl;


    //我们比较关心两帧之间的变换矩阵
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    g2o::SE3Quat pose = v->estimate();
    Tcw2 = Converter::toCvMat(pose);
    //cout<<"Pose="<<endl<<Tcw2<<endl;

    // 以及所有特征点的位置
    for ( size_t i=0; i<pts3d.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2));
        //cout<<"vertex id "<<i+2<<", pos = ";
        Eigen::Vector3d pos = v->estimate();
        //cout<<pos(0)<<","<<pos(1)<<","<<pos(2)<<endl;
        pts3d[i] = cv::Point3f(pos(0), pos(1), pos(2));
    }
}

void Optimizer::LocalBundleAdjustment(Sign* sign, Frame* frame, Frame::SignLabel* label, std::vector<cv::Point3f>& pts3d, int nIterations)
{

    //步骤1：初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    //linear solver
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    //linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();

    //algorithm, LM
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    optimizer.setAlgorithm(algorithm);

    //步骤2：准备相机参数
    float fx = frame->fx;
    float fy = frame->fy;
    float cx = frame->cx;
    float cy = frame->cy;

    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );

    
    // //步骤3：向优化器添加顶点 set vertices: 
    // //camera_pose ---> g2o::VertexSE3Expmap 
    // //map_points ---> g2o::VertexSBAPointXYZ
    g2o::VertexSE3Expmap* v1 = new g2o::VertexSE3Expmap();
    v1->setId(0);        
    v1->setFixed(true);   
    v1->setEstimate(Converter::toSE3Quat(sign->GetFirstFrame()->GetPose()));
    optimizer.addVertex(v1);
    
    g2o::VertexSE3Expmap* v2 = new g2o::VertexSE3Expmap();
    v2->setId(1);           
    v2->setEstimate(Converter::toSE3Quat(frame->GetPose()));
    optimizer.addVertex(v2);

    for(int i=0; i<pts3d.size(); i++)
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId( 2 + i );
        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(pts3d[i].x, pts3d[i].y, pts3d[i].z));
        optimizer.addVertex( v );        
    }

    // //步骤4：向优化器添加投影边边 
    // //projection relationship ---> g2o::EdgeProjectXYZ2UV
    for ( int i=0; i<4; i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );
        edge->setMeasurement( Eigen::Vector2d(sign->GetAllSeeds()[i]->mpt.x, sign->GetAllSeeds()[i]->mpt.y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
    }

    // 第二帧
    for ( int i=0; i<4; i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(1)) );
        edge->setMeasurement( Eigen::Vector2d(label->Vertices[i].x, label->Vertices[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0,0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
    }

    //步骤5：开始优化
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);


    g2o::VertexSE3Expmap* v1_new = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(0) );
    g2o::SE3Quat pose1 = v1_new->estimate();
    sign->GetFirstFrame()->SetPose(Converter::toCvMat(pose1));

    g2o::VertexSE3Expmap* v2_new = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    g2o::SE3Quat pose2 = v2_new->estimate();
    frame->SetPose(Converter::toCvMat(pose2));

    for ( int i=0; i<pts3d.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2));
        Eigen::Vector3d pos = v->estimate();
        pts3d[i] = cv::Point3f(pos(0), pos(1), pos(2));
    }

}

void Optimizer::LocalBundleAdjustment(Sign* sign, int nIterations)
{

    //necessary inputs
    vector<Frame*> Frames = sign->GetObservationFrames();
    vector<Frame::SignLabel*> Labels = sign->GetObservationLabels();

    vector<cv::Point3f> pts3Dw;
    for(int i=0; i<sign->GetAllSeeds().size(); i++)
        pts3Dw.push_back(sign->GetAllSeeds()[i]->GetWorldPose());

    int nObs = sign->nObs;
    int nSeeds = pts3Dw.size();


    //步骤1：初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    //linear solver
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    //linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();

    //algorithm, LM
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    optimizer.setAlgorithm(algorithm);

    //步骤2：准备相机参数
    float fx = sign->GetFirstFrame()->fx;
    float fy = sign->GetFirstFrame()->fy;
    float cx = sign->GetFirstFrame()->cx;
    float cy = sign->GetFirstFrame()->cy;

    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );


    // //步骤3：向优化器添加顶点 set vertices: 
    // //camera_pose ---> g2o::VertexSE3Expmap 
    // //map_points ---> g2o::VertexSBAPointXYZ
    for(int i=0; i<nObs; i++)
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);          
        v->setEstimate(Converter::toSE3Quat(Frames[i]->GetPose()));
        optimizer.addVertex(v);        
    }

    for(int i=0; i<nSeeds; i++)
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId( nObs + i );
        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(pts3Dw[i].x, pts3Dw[i].y, pts3Dw[i].z));
        optimizer.addVertex( v );        
    }


    // //步骤4：向优化器添加投影边边 
    // //projection relationship ---> g2o::EdgeProjectXYZ2UV
    for(int i=0; i<nObs; i++)
    {
        for(int j=0; j<nSeeds; j++)
        {
            g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
            edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(j+nObs)) );
            edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(i)) );
            edge->setMeasurement( Eigen::Vector2d(Labels[i]->Vertices[j].x, Labels[i]->Vertices[j].y ) );
            edge->setInformation( Eigen::Matrix2d::Identity() );
            edge->setParameterId(0, 0);
            edge->setRobustKernel( new g2o::RobustKernelHuber() );
            optimizer.addEdge( edge );
        }
    }

    //步骤5：开始优化
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    //步骤6：get result
    for(int i=0; i<nObs; i++)
    {
        g2o::VertexSE3Expmap* v_new = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(i) );
        g2o::SE3Quat pose = v_new->estimate();
        sign->GetObservationFrames()[i]->SetPose(Converter::toCvMat(pose));
    }

    for ( int i=0; i<nSeeds; i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+nObs));
        Eigen::Vector3d pos = v->estimate();
        cv::Point3f Pw = cv::Point3f(pos(0), pos(1), pos(2));
        cv::Mat Pc_ = sign->GetFirstFrame()->GetRotation()*Converter::toCvMat(Pw) + sign->GetFirstFrame()->GetTranslation();
        cv::Point3f Pc = Converter::toCvPoint3f(Pc_);
        sign->GetAllSeeds()[i]->mf = Pc/norm(Pc);
        sign->GetAllSeeds()[i]->mu = 1.0/norm(Pc);
    }

}

void Optimizer::BundleAdjustmentTwoFrames(Frame* frame1, Frame* frame2, vector<cv::Point3f>& pts3d, 
                                        const vector<cv::Point2f>& pts1, const vector<cv::Point2f>& pts2, int nIterations)
{

   //步骤1：初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    //linear solver
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    //linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();

    //algorithm, LM
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    optimizer.setAlgorithm(algorithm);

    //步骤2：准备相机参数
    float fx = frame1->fx;
    float fy = frame1->fy;
    float cx = frame1->cx;
    float cy = frame1->cy;

    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );

    //步骤3：向优化器添加顶点 set vertices: 
    //camera_pose ---> g2o::VertexSE3Expmap 
    //map_points ---> g2o::VertexSBAPointXYZ
    g2o::VertexSE3Expmap* v1 = new g2o::VertexSE3Expmap();
    v1->setId(0);        
    v1->setEstimate(Converter::toSE3Quat(frame1->GetPose()));
    optimizer.addVertex(v1);
    
    g2o::VertexSE3Expmap* v2 = new g2o::VertexSE3Expmap();
    v2->setId(1);        
    v2->setEstimate(Converter::toSE3Quat(frame2->GetPose()));
    optimizer.addVertex(v2);


    for(int i=0; i<pts3d.size(); i++)
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId( 2 + i );
        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(pts3d[i].x,pts3d[i].y,pts3d[i].z));
        optimizer.addVertex( v );        
    }


    //步骤4：向优化器添加投影边边 
    //projection relationship ---> g2o::EdgeProjectXYZ2UV
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );
        edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
    }

    // 第二帧
    for ( size_t i=0; i<pts2.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(1)) );
        edge->setMeasurement( Eigen::Vector2d(pts2[i].x, pts2[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0,0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
    }
    
    //步骤5：开始优化
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);


    //步骤6：get result
    g2o::VertexSE3Expmap* v1_new = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(0) );
    g2o::SE3Quat pose1 = v1_new->estimate();
    frame1->SetPose(Converter::toCvMat(pose1));

    g2o::VertexSE3Expmap* v2_new = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    g2o::SE3Quat pose2 = v2_new->estimate();
    frame2->SetPose(Converter::toCvMat(pose2));

    for ( int i=0; i<pts3d.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2));
        Eigen::Vector3d pos = v->estimate();
        pts3d[i] = cv::Point3f(pos(0), pos(1), pos(2));
    }
}
