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
	if (bVisualizeResult_)
	{
		vecColors_ = GenerateColors(400);
	}

}

void CResultIntegration::Finalize()
{
	if (!bInit_) { return; }

	bInit_ = false;

	/* visualize related */
	if (bVisualizeResult_) { cv::destroyWindow(strVisWindowName_); }
}

CDetectResultSet CResultIntegration::Run(hj::CTrackResult *_trackResult, 
	CThrowDetectorSet *_throwResult,
	jm::CActionResultSet *_actionResult, 
	cv::Mat _curFrame, 
	int _frameIdx)
{
	nCurrentFrameIdx_ = _frameIdx;
	matResult_ = _curFrame.clone();

	/* save result to member variable */
	curTrackResult_ = *_trackResult;
	curActionResult_ = *_actionResult;
	curThrowResult_ = *_throwResult;

	Integrate(/*_trackResult, _actionResult*/);

	if (bVisualizeResult_) { Visualize(); }

	return this->integratedResult_;
}


void CResultIntegration::Integrate(/*hj::CTrackResult *_trackResult, jm::CActionResultSet *_actionResult*/)
{
	assert(curTrackResult_.objectInfos.size() == curActionResult_.actionResults.size());
	//assert(curTrackResult_.objectInfos.size() == curThrowResult_.listThrowResult.size());
	
	integratedResult_.detectResults.clear();                     // �̺κ� �ٸ��� ������ �����?
	integratedResult_.frameIdx = this->nCurrentFrameIdx_;

	
	for (std::vector<hj::CObjectInfo>::iterator objIter = curTrackResult_.objectInfos.begin();
		objIter != curTrackResult_.objectInfos.end(); objIter++)
	{

		//stDetectResult *curDetectResult;   //�̰� �� ��������(?) �����غ���
		stDetectResult curDetectResult;

		curDetectResult.trackId = objIter->id;
		curDetectResult.keyPoint = objIter->keyPoint;
		curDetectResult.box = objIter->box;
		curDetectResult.headBox = objIter->headBox;

		for (std::vector<stActionResult>::iterator actionIter = curActionResult_.actionResults.begin();
			actionIter != curActionResult_.actionResults.end(); actionIter++)
		{
			if (objIter->id != actionIter->trackId) { continue; }

			curDetectResult.bActionDetect = actionIter->bActionDetect;
		}
/*
		for (std::vector<CThrowDetector>::iterator throwIter = curThrowResult_.listThrowResult.begin();
			throwIter != curThrowResult_.listThrowResult.end(); throwIter++)
		{
			if (objIter->id != throwIter->trackId) { continue; }
			
			curDetectResult.bThrowDetect = 
		}
*/
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

	//TODO: Track result�� ǥ�õǾ�� �ϰ�, detection result�� ǥ�õǾ�� �Ѵ�.( ���� detection�� ��µȴ�.)

	/*SVM Result*/
	for (std::vector<stDetectResult>::iterator resultIter = integratedResult_.detectResults.begin();
		resultIter != integratedResult_.detectResults.end(); resultIter++)
	{
		if (!resultIter->bActionDetect) { continue; }


		cv::rectangle(
			matResult_,
			resultIter->box,
			cv::Scalar(0, 0, 255), 3);

		char strDetectResult[100];
		sprintf_s(strDetectResult, "Throwing Detected(%d person)", resultIter->trackId);
		cv::putText(matResult_, strDetectResult, cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

	}

	/*Track Result*/
	for (std::vector<hj::CObjectInfo>::iterator trackIter = curTrackResult_.objectInfos.begin();
		trackIter != curTrackResult_.objectInfos.end(); trackIter++)
	{
		DrawBoxWithID(
			matResult_,
			trackIter->box,
			trackIter->id,
			0,
			0,
			getColorByID(trackIter->id, &vecColors_));

		cv::rectangle(
			matResult_,
			trackIter->headBox,
			cv::Scalar(0, 0, 255), 1);
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

// �� �Լ��� ��� ������... Track���� �޾ƿ��°� ������ �ѵ�..(������)
void CResultIntegration::DrawBoxWithID(
	cv::Mat &imageFrame,
	cv::Rect box,
	unsigned int nID,
	int lineStyle,
	int fontSize,
	cv::Scalar curColor)
{
	// get label length
	unsigned int labelLength = nID > 0 ? 0 : 1;
	unsigned int tempLabel = nID;
	while (tempLabel > 0)
	{
		tempLabel /= 10;
		labelLength++;
	}
	if (0 == fontSize)
	{
		cv::rectangle(imageFrame, box, curColor, 1);
		cv::rectangle(imageFrame, cv::Rect((int)box.x, (int)box.y - 10, 7 * labelLength, 14), curColor, CV_FILLED);
		cv::putText(imageFrame, std::to_string(nID), cv::Point((int)box.x, (int)box.y - 1), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));
	}
	else
	{
		cv::rectangle(imageFrame, box, curColor, 1 + lineStyle);
		cv::putText(imageFrame, std::to_string(nID), cv::Point((int)box.x, (int)box.y + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, curColor);
	}
}


cv::Scalar CResultIntegration::getColorByID(unsigned int nID, std::vector<cv::Scalar> *vecColors)
{
	if (NULL == vecColors) { return cv::Scalar(255, 255, 255); }
	unsigned int colorIdx = nID % vecColors->size();
	return (*vecColors)[colorIdx];
}

std::vector<cv::Scalar> CResultIntegration::GenerateColors(unsigned int numColor)
{
	double golden_ratio_conjugate = 0.618033988749895;
	//double hVal = (double)std::rand()/(INT_MAX);
	double hVal = 0.0;
	std::vector<cv::Scalar> resultColors;
	resultColors.reserve(numColor);
	for (unsigned int colorIdx = 0; colorIdx < numColor; colorIdx++)
	{
		hVal += golden_ratio_conjugate;
		hVal = std::fmod(hVal, 1.0);
		resultColors.push_back(hsv2rgb(hVal, 0.5, 0.95));
	}
	return resultColors;
}


cv::Scalar CResultIntegration::hsv2rgb(double h, double s, double v)
{
	int h_i = (int)(h * 6);
	double f = h * 6 - (double)h_i;
	double p = v * (1 - s);
	double q = v * (1 - f * s);
	double t = v * (1 - (1 - f) * s);
	double r, g, b;
	switch (h_i)
	{
	case 0: r = v; g = t; b = p; break;
	case 1: r = q; g = v; b = p; break;
	case 2: r = p; g = v; b = t; break;
	case 3: r = p; g = q; b = v; break;
	case 4: r = t; g = p; b = v; break;
	case 5: r = v; g = p; b = q; break;
	default:
		break;
	}

	return cv::Scalar((int)(r * 255), (int)(g * 255), (int)(b * 255));
}

}