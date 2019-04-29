#include "sign.h"

int Sign::sign_counter = 0;

Sign::Sign(std::vector<Seed*> Seeds, Frame* frame):nObs(0)
{
	this->mFirstFrame = frame;
	for(int i=0; i<Seeds.size(); i++)
		this->mSeeds.push_back(Seeds[i]);

	this->mId = sign_counter++;
}

float Sign::GetAvgWidth()
{
	cv::Point3f pLT = mSeeds[0]->GetWorldPose();
	cv::Point3f pRT = mSeeds[1]->GetWorldPose();
	cv::Point3f pRB = mSeeds[2]->GetWorldPose();
	cv::Point3f pLB = mSeeds[3]->GetWorldPose();

	cv::Point3f width1 = pLT-pRT;
	cv::Point3f width2 = pLB-pRB;

	return (norm(width1)+norm(width2))/2;
}
	

float Sign::GetAvgHeight()
{
	cv::Point3f pLT = mSeeds[0]->GetWorldPose();
	cv::Point3f pRT = mSeeds[1]->GetWorldPose();
	cv::Point3f pRB = mSeeds[2]->GetWorldPose();
	cv::Point3f pLB = mSeeds[3]->GetWorldPose();

	cv::Point3f height1 = pLT-pLB;
	cv::Point3f height2 = pRT-pRB;

	return (norm(height1)+norm(height2))/2;
}

void Sign::AddObservation(Frame* frame, Frame::SignLabel* label)
{
	this->mObservationFrames.push_back(frame);
	this->mObservationLabels.push_back(label);
    this->nObs++;
}