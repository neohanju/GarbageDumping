#pragma once
#include <deque>
#include <list>
#include <queue>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\ml.hpp>

#include "haanju_utils.hpp"

namespace jm
{
/////////////////////////////////////////////////////////////////////////
// ALGORITHM PARAMETERS
/////////////////////////////////////////////////////////////////////////
struct stParamAction
{
	stParamAction()
		: nPoseLength(30)
		, nMaxPendingFrame(5)
		, nStepSize(5)
		, bTrained(true)
		, bVisualize(true)
	{};

	~stParamAction() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
	int  nPoseLength;
	int  nMaxPendingFrame;
	int  nStepSize;
	bool bVisualize;
	bool bTrained;
};

//typedef std::vector<hj::stKeyPoint> CPosePoints;
//typedef std::deque<CPosePoints> CAction;
typedef std::deque<hj::CObjectInfo> CAction;   //change this in final implementation
// continuous frame pose 
class CPoselet
{
	//----------------------------------------------------------------
	// METHODS
	//---------------------------------------------------------------
public:
	CPoselet()
		: lastUpdate(-1)
	{};
	~CPoselet() {};
	//void interpolation() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
public:
	int id;
	int nStartFrame;
	int nEndFrame;
	int duration;
	int lastUpdate;      // 30frame씩 묶는것 관련
	//CAction vectorPose;
	CAction vectorObjInfo;       // 이름 다시 생각해보기
	bool bActionDetect;          // TODO: change CAction


};

// typedef std::deque<CAction> ActionSet;

class CActionClassifier
{
public:
	CActionClassifier();
	~CActionClassifier();

	void Initialize(stParamAction &stParam, std::string _strModelPath);
	void Finalize();
	void Run(/*hj::KeyPointsSet _curKeypoints*/ hj::CTrackResult *_curTrackResult, cv::Mat _curFrame, int frameIdx);



private:
	void Detect(std::deque<CPoselet*> _activePoselets, hj::CTrackResult *_curTrackResult);
	void TrainSVM(std::string _saveModelPath);
	void UpdatePoseletUsingTrack(/*hj::CTrackResult _curTrackResult*/);
	void Normalize();
	void Visualize(hj::CTrackResult *_curTrackResult);
	void EliminationStepSize();
	//void ResultPackaging(hj::CTrackResult _curTrackResult);
	//void LoadData(std::deque<CPoselet*> _activePoselets);
	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
public:
	bool             bInit_;
	stParamAction    stParam_;
	unsigned int     nCurrentFrameIdx_;
	hj::CTrackResult curTrackResult_;

	/* visualization related */
	bool             bVisualizeResult_;
	cv::Mat          matDetectResult_;
	std::string      strVisWindowName_;

	/* detection input data related */
	std::list<CPoselet>   listCPoselets_;
	std::deque<CPoselet*> pendingPoselets_;
	std::deque<CPoselet*> activePoselets_;

	std::deque<CAction*>  testActions_;
	std::list<CAction>    listCActions_;

	cv::Ptr<cv::ml::SVM> svm;

private:

};

}
