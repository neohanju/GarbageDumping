#include "ResultIntegration.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace jm
{

CResultIntegration::CResultIntegration()
	:bInit_(false)
{
}


CResultIntegration::~CResultIntegration()
{
	Finalize();
}

void CResultIntegration::Initialize(stParamResult &_stParam)
{
	if (bInit_) { Finalize(); }

	stResultParam_ = _stParam;
	bInit_ = true;


	// visualization related
	bVisualizeResult_ = stResultParam_.bVisualize;
	strVisWindowName_ = "Final result";

}

void CResultIntegration::Finalize()
{
	if (!bInit_) { return; }

	bInit_ = false;

	/* visualize related */
	if (bVisualizeResult_) { cv::destroyWindow(strVisWindowName_); }
}

CDetectResultSet CResultIntegration::Run(hj::CTrackResult *_trackResult, jm::CActionResultSet *_actionResult, cv::Mat _curFrame, int _frameIdx)
{
	nCurrentFrameIdx_ = _frameIdx;
	matResult_ = _curFrame.clone();

	//curTrackResult_ = *_trackResult;
	//curActionResult_ = *_actionResult;

	Integrate(_trackResult, _actionResult);

	if (bVisualizeResult_) { Visualize(); }

	return this->integratedResult_;
}


void CResultIntegration::Integrate(hj::CTrackResult *_trackResult, jm::CActionResultSet *_actionResult)
{
	assert(_trackResult->objectInfos.size() == _actionResult->actionResults.size());
	
	integratedResult_.detectResults.clear();                     // 이부분 다르게 수정할 방법은?
	integratedResult_.frameIdx = this->nCurrentFrameIdx_;

	for (std::vector<hj::CObjectInfo>::iterator objIter = _trackResult->objectInfos.begin();
		objIter != _trackResult->objectInfos.end(); objIter++)
	{

		//stDetectResult *curDetectResult;   //이거 왜 에러나지(?) 생각해보기
		stDetectResult curDetectResult;

		curDetectResult.trackId = objIter->id;
		curDetectResult.keyPoint = objIter->keyPoint;
		curDetectResult.box = objIter->box;
		curDetectResult.headBox = objIter->headBox;

		for (std::vector<stActionResult>::iterator actionIter = _actionResult->actionResults.begin();
			actionIter != _actionResult->actionResults.end(); actionIter++)
		{
			if (objIter->id != actionIter->trackId) { continue; }

			curDetectResult.bActionDetect = actionIter->bActionDetect;
		}

		//TODO: Throw Detect


		integratedResult_.detectResults.push_back(curDetectResult);
	}
	

	
}

void CResultIntegration::Visualize()
{
	/* frame information */
	char strFrameInfo[100];
	sprintf_s(strFrameInfo, "%04d", this->nCurrentFrameIdx_);
	cv::rectangle(matResult_, cv::Rect(5, 2, 100, 22), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(matResult_, strFrameInfo, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));

	//TODO: Track result도 표시되어야 하고, detection result도 표시되어야 한다.( 현재 detection만 출력된다.)
	for (std::vector<stDetectResult>::iterator resultIter = integratedResult_.detectResults.begin();
		resultIter != integratedResult_.detectResults.end(); resultIter++)
	{
		if (!resultIter->bActionDetect) { continue; }


		cv::rectangle(
			matResult_,
			resultIter->box,
			//_curTrackResult->objectInfos.at(index).box,
			cv::Scalar(0, 0, 255), 1);

		char strDetectResult[100];
		sprintf_s(strDetectResult, "Throwing Detected(%d person)", resultIter->trackId);
		cv::putText(matResult_, strDetectResult, cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

	}


	////---------------------------------------------------
	//// RECORD
	////---------------------------------------------------
	//if (bVideoWriterInit_)
	//{
	//	IplImage *currentFrame = new IplImage(matTrackingResult_);
	//	cvWriteFrame(videoWriter_, currentFrame);
	//	delete currentFrame;
	//}

	cv::namedWindow(strVisWindowName_);

	cv::imshow(strVisWindowName_, matResult_);
	cv::waitKey(1);
	matResult_.release();
}

}