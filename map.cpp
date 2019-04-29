#include "map.h"

Map::Map()
{
}

void Map::AddFrame(Frame* frame)
{
	this->mFrames.push_back(frame);
}

void Map::AddSign(Sign* sign)
{
	this->mSigns.push_back(sign);
}

void Map::AddMapPoint(MapPoint* mappoint)
{
	this->mMappoints.push_back(mappoint);
}
