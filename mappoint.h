#pragma once
#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "frame.h"
#include "seed.h"

class MapPoint
{
public:
	MapPoint(Seed* seed, Frame* frame);

	void AddObservation(Frame* frame, cv::Point2f px);
	
	Seed* GetSeed(){ return mSeed; }
	Frame* GetFirstFrame(){ return mFirstFrame; }
	cv::Point3f GetWorldPose(){ return mSeed->GetWorldPose(); }
	std::vector<Frame*> GetObservationFrames(){ return mObservationFrames; }
	std::vector<cv::Point2f> GetObservationPxs(){ return mObservationPxs; }
 
public:
	static int sign_counter;
	int mId;  
	int nObs;

private:
	Seed* mSeed;
	Frame* mFirstFrame;

private:
	std::vector<Frame*> mObservationFrames;
	std::vector<cv::Point2f> mObservationPxs;



};
#endif //MAPPOINT_H