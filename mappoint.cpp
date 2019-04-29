#include "mappoint.h"

int MapPoint::sign_counter = 0;

MapPoint::MapPoint(Seed* seed, Frame* frame):nObs(0)
{
	this->mSeed = seed;
	this->mFirstFrame = frame;
	this->mId = sign_counter++;
}

void MapPoint::AddObservation(Frame* frame, cv::Point2f px)
{
	this->mObservationFrames.push_back(frame);
	this->mObservationPxs.push_back(px);
	this->nObs++;
}