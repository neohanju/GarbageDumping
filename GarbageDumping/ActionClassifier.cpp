#include "ActionClassifier.h"
#include <opencv2\ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace jm
{

CActionClassifier::CActionClassifier()
	: bInit_(false)
{
}


CActionClassifier::~CActionClassifier()
{
	Finalize();
}

void CActionClassifier::Initialize(stParamAction &_stParam)
{
	if (bInit_) { Finalize(); }

	stParam_ = _stParam;
	bInit_ = true;

	// visualization related
	bVisualizeResult_ = stParam_.bVisualize;
	strVisWindowName_ = "Detection result";
}

void CActionClassifier::Finalize()
{
	if (!bInit_) { return; }

	bInit_ = false;

	/* visualize related */
	if (bVisualizeResult_) { cv::destroyWindow(strVisWindowName_); }
}

void CActionClassifier::UpdatePoseletUsingTrack(hj::CTrackResult _curTrackResult)
{
	//active Poselet update
	std::deque<CPoselet*> newActivePoselets;
	std::deque<CPoselet*> newPendingPoselets;

	//---------------------------------------------------
	// MATCHING STEP 01: active Poselet <-> keypoints
	//---------------------------------------------------
	for (std::vector<hj::CObjectInfo>::iterator objectIter = _curTrackResult.objectInfos.begin();
		objectIter != _curTrackResult.objectInfos.end();)
	{
		bool match = false;
		for (int poseletIdx = 0; poseletIdx < activePoselets_.size(); poseletIdx++)
		{
			CPoselet *curPoselet = activePoselets_[poseletIdx];

			if (curPoselet->id != objectIter->id) { continue; }

			//curPoselet->lastUpdate = this->nCurrentFrameIdx_;
			curPoselet->nEndFrame = this->nCurrentFrameIdx_;         // TODO: EndFrame과 last update은 달라야 한다 interpolation 해야하는 지점들 찾기위함(나중에 둘중 하나는 없앨지도..)
			curPoselet->duration++;
			//curPoselet->vectorPose.push_back(/*objectIter->keyPoint*/);
			curPoselet->vectorObjInfo.push_back(*objectIter);
			newActivePoselets.push_back(curPoselet);

			objectIter = _curTrackResult.objectInfos.erase(objectIter);
			match = true;
			break;
		}

		if (match) { continue; }
		objectIter++;
	}

	// Update pending poselet
	for (std::deque<CPoselet*>::iterator poseIter = activePoselets_.begin();
		poseIter != activePoselets_.end(); poseIter++)
	{
		if ((*poseIter)->nEndFrame /*lastUpdate*/ == this->nCurrentFrameIdx_) { continue; }
		newPendingPoselets.push_back((*poseIter));
	}


	//---------------------------------------------------
	// MATCHING STEP 02: pending Poselet <-> keypoints
	//---------------------------------------------------
	for (std::vector<hj::CObjectInfo>::iterator objectIter = _curTrackResult.objectInfos.begin();
		objectIter != _curTrackResult.objectInfos.end();)
	{
		bool match = false;
		for (int poseletIdx = 0; poseletIdx < pendingPoselets_.size(); poseletIdx++)
		{
			CPoselet *curPoselet = pendingPoselets_[poseletIdx];

			if (curPoselet->id != objectIter->id) { continue; }

			//// inactive에서 active로 업데이트 될때 update된 프레임 -1이 이전에 최근 업뎃과 같지 않으면 interpolation하는 작업
			////interpolation ( TODO: 방법 수정)
			//int pendingTime = this->nCurrentFrameIdx_ - curPoselet->nEndFrame;
			//for (int interpolFrame = 1; interpolFrame < pendingTime; interpolFrame++)
			//{
			//	curPoselet->duration++;

			//	int nKeypoint = 0;
			//	hj::stKeyPoint tmpPoint;
			//	std::vector<hj::stKeyPoint> tmpPoints;
			//	for (std::vector<hj::stKeyPoint>::iterator pointIter = objectIter->keyPoint.begin();
			//		pointIter != objectIter->keyPoint.end(); pointIter++, nKeypoint++)
			//	{
			//		tmpPoint.x = 
			//			//( (pendingTime - interpolFrame) * curPoselet->vectorPose.back()[nKeypoint].x + interpolFrame * pointIter->x) / pendingTime;
			//			((pendingTime - interpolFrame) * curPoselet->vectorObjInfo.back().keyPoint.at(nKeypoint).x + interpolFrame * pointIter->x) / pendingTime;

			//		tmpPoint.y = 
			//			//( (pendingTime - interpolFrame) * curPoselet->vectorPose.back()[nKeypoint].y + interpolFrame * pointIter->y) / pendingTime;
			//			((pendingTime - interpolFrame) * curPoselet->vectorObjInfo.back().keyPoint.at(nKeypoint).y + interpolFrame * pointIter->y) / pendingTime;
			//		tmpPoint.confidence = 0;
			//		tmpPoints.push_back(tmpPoint);
			//	}

			//	curPoselet->vectorPose.push_back(tmpPoints);
			//}

			// curPoselet->lastUpdate = this->nCurrentFrameIdx_;
			curPoselet->nEndFrame  = this->nCurrentFrameIdx_;         // TODO: EndFrame과 last update은 달라야 한다 interpolation 해야하는 지점들 찾기위함(나중에 둘중 하나는 없앨지도..)
			curPoselet->duration++;
			//curPoselet->vectorPose.push_back(objectIter->keyPoint);
			curPoselet->vectorObjInfo.push_back(*objectIter);
			newActivePoselets.push_back(curPoselet);


			objectIter = _curTrackResult.objectInfos.erase(objectIter);
			match = true;
			break;
		}

		if (match) { continue; }
		objectIter++;
	}

	// Update pending poselet
	for (std::deque<CPoselet*>::iterator poseIter = pendingPoselets_.begin();
		poseIter != pendingPoselets_.end(); poseIter++)
	{
		if ((*poseIter)->nEndFrame/*lastUpdate*/ == this->nCurrentFrameIdx_) { continue; }
		newPendingPoselets.push_back((*poseIter));
	}


	//---------------------------------------------------
	// MATCHING STEP 03: Generation Poselet
	//---------------------------------------------------
	for (int idx = 0; idx < _curTrackResult.objectInfos.size(); idx++)
	{
		CPoselet newPoselet;

		newPoselet.id = _curTrackResult.objectInfos[idx].id;
		// newPoselet.lastUpdate = this->nCurrentFrameIdx_;
		newPoselet.nStartFrame = this->nCurrentFrameIdx_;
		newPoselet.nEndFrame = this->nCurrentFrameIdx_;
		newPoselet.duration = 1;
		//newPoselet.vectorPose.push_back(_curTrackResult.objectInfos[idx].keyPoint);
		newPoselet.vectorObjInfo.push_back(_curTrackResult.objectInfos[idx]);

		this->listCPoselets_.push_back(newPoselet);
		newActivePoselets.push_back(&this->listCPoselets_.back());
	}

	//------------------------------------------------
	// MATCHING STEP 04: POSELET TERMINATION
	//------------------------------------------------
	for (std::deque<CPoselet*>::iterator poseIter = newPendingPoselets.begin(); poseIter != newPendingPoselets.end();)
	{
		if ((*poseIter)->nEndFrame/*lastUpdate*/ + stParam_.nMaxPendingFrame < nCurrentFrameIdx_)
		{
			poseIter = newPendingPoselets.erase(poseIter);
			continue;
		}
		poseIter++;
	}

	activePoselets_ = newActivePoselets;
	pendingPoselets_ = newPendingPoselets;

}


// Load python SVM train model 
void CActionClassifier::Detect(std::string _curModelPath, /*std::deque<CAction*>  _testActions,*/ std::deque<CPoselet*> _activePoselets)
{
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm = cv::ml::SVM::load<cv::ml::SVM>(_curModelPath);
	 
	//all active poselet check
	for (std::deque<CPoselet*>::iterator poseletIter = _activePoselets.begin();
		poseletIter != _activePoselets.end(); poseletIter++)
	{
		if ((*poseletIter)->vectorObjInfo.size() < stParam_.nPoseLength) { continue; }
/*
		//float testData[30][36] = {};

		for (int poseIdx = 0; poseIdx < (*poseletIter)->vectorObjInfo.size(); poseIdx++)
		{
			for (int pointIdx = 0; (*poseletIter)->vectorObjInfo.at(0).keyPoint.size(); pointIdx++)
			{
				testData[poseIdx][2 * pointIdx] = (*poseletIter)->vectorObjInfo.at(poseIdx).keyPoint.at(pointIdx).x - (*poseletIter)->vectorObjInfo.at(pointIdx).keyPoint.at(1).x;
				testData[poseIdx][2 * pointIdx + 1] = (*poseletIter)->vectorObjInfo.at(poseIdx).keyPoint.at(pointIdx).y - (*poseletIter)->vectorObjInfo.at(pointIdx).keyPoint.at(1).y;
			}
		}
*/
		cv::Mat sampleMat, tmpMat;
		for (CAction::iterator objIter = (*poseletIter)->vectorObjInfo.begin();
			objIter != (*poseletIter)->vectorObjInfo.end(); objIter++ )
		{
			
			for (std::vector<hj::stKeyPoint>::iterator pointIter = objIter->keyPoint.begin();
				pointIter != objIter->keyPoint.end(); pointIter++)
			{
				sampleMat.push_back(pointIter->x - objIter->keyPoint.at(1).x);
				sampleMat.push_back(pointIter->y - objIter->keyPoint.at(1).y);
			}
		}

		sampleMat.convertTo(tmpMat, CV_32FC1);
		int res = svm->predict(tmpMat.t());
		printf("frame: %d  trackID: %d response: %d\n", nCurrentFrameIdx_, (*poseletIter)->id, res);
		(*poseletIter)->bActionDetect = true;

		////elimination front step size object info.
		//for (int poseIdx = 0; poseIdx < stParam_.nStepSize; poseIdx++)
		//{
		//	(*poseletIter)->vectorObjInfo.pop_front();
		//	(*poseletIter)->nStartFrame++;
		//}
		//(*poseletIter)->bActionDetect = false;

	}

	
}

// Train in opencv SVM model 
void CActionClassifier::TrainSVM(std::string _saveModelPath)
{
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

	//TODO: Train process
	//demand Labeling 

	svm->save(_saveModelPath);

}

void CActionClassifier::EliminationStepSize()
{
	for (std::deque<CPoselet*>::iterator poseletIter = activePoselets_.begin();
		poseletIter != activePoselets_.end(); poseletIter++)
	{
		if ((*poseletIter)->vectorObjInfo.size() < stParam_.nPoseLength) { continue; }

		//elimination front step size object info.
		for (int poseIdx = 0; poseIdx < stParam_.nStepSize; poseIdx++)
		{
			(*poseletIter)->vectorObjInfo.pop_front();
			(*poseletIter)->nStartFrame++;
		}
		(*poseletIter)->bActionDetect = false;
	}

}

// when input action vector 
void CActionClassifier::Run(/*hj::KeyPointsSet _curKeypoints*/ hj::CTrackResult _curTrackResult, cv::Mat _curFrame, int frameIdx, std::string _strModelPath)
{
	nCurrentFrameIdx_ = frameIdx;
	UpdatePoseletUsingTrack(_curTrackResult);
	matDetectResult_ = _curFrame.clone();

	
	if (activePoselets_.size()) 
	{ 
		Detect(_strModelPath, activePoselets_); 
	}
	

	/* visualize */
	if (bVisualizeResult_) { Visualize(); }

	EliminationStepSize();
}


// Normalize pose
void CActionClassifier::Normalize()
{
}


// Result visualization
void CActionClassifier::Visualize()
{
	/* frame information */
	char strFrameInfo[100];
	sprintf_s(strFrameInfo, "%04d", this->nCurrentFrameIdx_);
	cv::rectangle(matDetectResult_, cv::Rect(5, 2, 100, 22), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(matDetectResult_, strFrameInfo, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));


	//activePoselets based visualize
	for (int poseletIdx = 0; poseletIdx < activePoselets_.size(); poseletIdx++)
	{
		CPoselet *curPoselet = activePoselets_[poseletIdx];

		if (curPoselet->vectorObjInfo.size() < stParam_.nPoseLength) { continue; }
		if (!curPoselet->bActionDetect) { continue; }
		if (curPoselet->nEndFrame != this->nCurrentFrameIdx_) { continue; }

		cv::rectangle(
			matDetectResult_,
			curPoselet->vectorObjInfo.back().box,
			cv::Scalar(0, 0, 255), 1);

		char strDetectResult[100];
		sprintf_s(strDetectResult, "Throwing Detected(%d person)", curPoselet->id);
		cv::putText(matDetectResult_, strDetectResult, cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

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
	cv::moveWindow(strVisWindowName_, 200, 10);
	cv::imshow(strVisWindowName_, matDetectResult_);
	cv::waitKey(1);
	matDetectResult_.release();
}


/*
void CActionClassifier::LoadData(std::deque<CPoselet*> _activePoselets)
{
std::deque<CAction*> newActionSets;
//all active poselet check
for (std::deque<CPoselet*>::iterator poseletIter = _activePoselets.begin();
poseletIter != _activePoselets.end(); poseletIter++)
{
if ((*poseletIter)->vectorPose.size() < stParam_.nPoseLength) { continue; }

CAction newAction;
for (int idx = 0; idx < (*poseletIter)->vectorPose.size();idx++)
{
if (idx > stParam_.nPoseLength) { break; }
newAction.push_back((*poseletIter)->vectorPose[idx]);
//
//if (idx < stParam_.nStepSize) {
//	(*poseletIter)->vectorPose.pop_front();
//	(*poseletIter)->nStartFrame++;
//}

}
this->listCActions_.push_back(newAction);
newActionSets.push_back(&this->listCActions_.back());
}
testActions_ = newActionSets;
}
*/

}