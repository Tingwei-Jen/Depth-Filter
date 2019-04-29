#pragma once
#ifndef SIGN_H
#define SIGN_H

#include "frame.h"
#include "seed.h"

class Sign
{
public:
	Sign(std::vector<Seed*> Seeds, Frame* frame);
	
	float GetAvgWidth(); 
	float GetAvgHeight();

	void AddObservation(Frame* frame, Frame::SignLabel* label);
	Frame* GetFirstFrame(){ return mFirstFrame; }
	std::vector<Seed*> GetAllSeeds(){ return mSeeds; }
	std::vector<Frame*> GetObservationFrames(){ return mObservationFrames; }
	std::vector<Frame::SignLabel*> GetObservationLabels(){ return mObservationLabels; }

public:
	static int sign_counter;
	int mId;  
	int nObs;

private:
	Frame* mFirstFrame;
	std::vector<Seed*> mSeeds;

private:
	std::vector<Frame*> mObservationFrames;
	std::vector<Frame::SignLabel*> mObservationLabels;
};

#endif //SIGN_H