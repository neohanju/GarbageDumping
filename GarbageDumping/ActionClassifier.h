#pragma once
#include <opencv2\ml\ml.hpp>
#include <deque>
#include <list>
#include <queue>
#include "haanju_utils.hpp"
#include <map>

namespace jm
{
/////////////////////////////////////////////////////////////////////////
// ALGORITHM PARAMETERS
/////////////////////////////////////////////////////////////////////////
struct stParamAction
{
	stParamAction()
		:nPoseLength(30)
		,nPendingFrame(5)
	{};

	~stParamAction() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
	int nPoseLength;
	int nPendingFrame;
};

// continuous frame pose 
class CPoselet
{
	//----------------------------------------------------------------
	// METHODS
	//---------------------------------------------------------------
public:
	CPoselet()
		:nPendingFrame(0)
	{};
	~CPoselet() {};
	void interpolation() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
public:
	int id;
	int nStartFrame;
	int nEndFrame;
	int nPendingFrame;
	int duration;
	int lastUpdate;
	std::vector<int> needInterpolation;
	std::deque<std::vector<hj::stKeyPoint>> vectorPose;

};

class CActionClassifier
{
public:
	CActionClassifier();
	~CActionClassifier();

	void Initialize(stParamAction &stParam);
	void Finalize();

	void UpdatePoselet(hj::KeyPointsSet _curKeypoints, int frameIdx);


private:
	void Load();
	void Train();
	void Run();
	void Visualize();

//------------------------------------------------
// VARIABLES
//------------------------------------------------
public:
	std::deque<CPoselet*> pendingPoselet;
	std::deque<CPoselet*> activePoselet;
	stParamAction stParam_;


private:

};

}
