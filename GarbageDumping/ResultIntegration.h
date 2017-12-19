#pragma once
#include <opencv2/highgui/highgui.hpp>

#include "ActionClassifier.h"

namespace jm
{

struct stParamResult
{
	stParamResult()
		: bVisualize(true)
	{};

	~stParamResult() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------

	bool bVisualize;
};

struct stDetectResult
{
	stDetectResult() : trackId(0), bActionDetect(false) {}//,bThrowDetect {}
	stDetectResult(unsigned int _trackId, bool _bActionDetect)
		: trackId(_trackId), bActionDetect(_bActionDetect) {} // ,bTrhowDetect(_bThrowDetect){}

	unsigned int trackId;
	bool bActionDetect;
	//bool bThrowDetect;

	cv::Rect2d   box;
	cv::Rect2d   headBox;
	std::vector<hj::stKeyPoint> keyPoint;
};

/////////////////////////////////////////////////////////////////////////
// FINAL RESULT (OF ENTIRE TARGETS)
/////////////////////////////////////////////////////////////////////////
class CDetectResultSet
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	CDetectResultSet() : frameIdx(0), timeStamp(0), procTime(0) {}
	~CDetectResultSet() {}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int frameIdx;
	unsigned int timeStamp;
	time_t procTime;
	std::vector<stDetectResult> detectResults;
};

class CResultIntegration
{
	//----------------------------------------------------------------
	// METHODS
	//---------------------------------------------------------------
public:
	CResultIntegration();
	~CResultIntegration();

	void Initialize(stParamResult &stParams_);
	void Finalize();
	CDetectResultSet Run(hj::CTrackResult *_trackResult, jm::CActionResultSet *_actionResult, cv::Mat _curFrame, int _frameIdx);

private:

	void Integrate(hj::CTrackResult *_trackResult, jm::CActionResultSet *_actionResult);
	void Visualize();

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
public:
	bool bInit_;
	stParamResult stResultParam_;
	unsigned int     nCurrentFrameIdx_;


	/* visualization related */
	bool             bVisualizeResult_;
	std::string      strVisWindowName_;
	cv::Mat          matResult_;


	CDetectResultSet integratedResult_;

	//hj::CTrackResult     curTrackResult_;         //있어야 하나? 없어도 될듯!
	//jm::CActionResultSet curActionResult_;

	/*Detection relate*/
	bool bSVMResult;
	bool bThrowResult;

};
}