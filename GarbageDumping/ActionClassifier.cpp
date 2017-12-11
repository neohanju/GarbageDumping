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

void CActionClassifier::UpdatePoseletUsingTrack(/*hj::CTrackResult _curTrackResult*/)
{
	//active Poselet update
	std::deque<CPoselet*> newActivePoselets;
	std::deque<CPoselet*> newPendingPoselets;

	//---------------------------------------------------
	// MATCHING STEP 01: active Poselet <-> keypoints
	//---------------------------------------------------
	for (std::vector<hj::CObjectInfo>::iterator objectIter = curTrackResult_.objectInfos.begin();
		objectIter != curTrackResult_.objectInfos.end();)
	{
		bool match = false;
		for (int poseletIdx = 0; poseletIdx < activePoselets_.size(); poseletIdx++)
		{
			CPoselet *curPoselet = activePoselets_[poseletIdx];

			if (curPoselet->id != objectIter->id) { continue; }

			curPoselet->nEndFrame = this->nCurrentFrameIdx_;         // TODO: EndFrame�� last update�� �޶�� �Ѵ� interpolation �ؾ��ϴ� ������ ã������(���߿� ���� �ϳ��� ��������..)
			curPoselet->duration++;
			curPoselet->vectorObjInfo.push_back(*objectIter);
			newActivePoselets.push_back(curPoselet);

			objectIter = curTrackResult_.objectInfos.erase(objectIter);
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
		if ((*poseIter)->nEndFrame == this->nCurrentFrameIdx_) { continue; }
		newPendingPoselets.push_back((*poseIter));
	}


	//---------------------------------------------------
	// MATCHING STEP 02: pending Poselet <-> keypoints
	//---------------------------------------------------
	for (std::vector<hj::CObjectInfo>::iterator objectIter = curTrackResult_.objectInfos.begin();
		objectIter != curTrackResult_.objectInfos.end();)
	{
		bool match = false;
		for (int poseletIdx = 0; poseletIdx < pendingPoselets_.size(); poseletIdx++)
		{
			CPoselet *curPoselet = pendingPoselets_[poseletIdx];

			if (curPoselet->id != objectIter->id) { continue; }

			//// inactive���� active�� ������Ʈ �ɶ� update�� ������ -1�� ������ �ֱ� ������ ���� ������ interpolation�ϴ� �۾�
			////interpolation ( TODO: ��� ����)
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
			curPoselet->nEndFrame  = this->nCurrentFrameIdx_;         // TODO: EndFrame�� last update�� �޶�� �Ѵ� interpolation �ؾ��ϴ� ������ ã������(���߿� ���� �ϳ��� ��������..)
			curPoselet->duration++;
			curPoselet->vectorObjInfo.push_back(*objectIter);
			newActivePoselets.push_back(curPoselet);


			objectIter = curTrackResult_.objectInfos.erase(objectIter);
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
		if ((*poseIter)->nEndFrame == this->nCurrentFrameIdx_) { continue; }
		newPendingPoselets.push_back((*poseIter));
	}


	//---------------------------------------------------
	// MATCHING STEP 03: Generation Poselet
	//---------------------------------------------------
	for (int idx = 0; idx < curTrackResult_.objectInfos.size(); idx++)
	{
		CPoselet newPoselet;

		newPoselet.id = curTrackResult_.objectInfos[idx].id;
		newPoselet.nStartFrame = this->nCurrentFrameIdx_;
		newPoselet.nEndFrame = this->nCurrentFrameIdx_;
		newPoselet.duration = 1;
		newPoselet.vectorObjInfo.push_back(curTrackResult_.objectInfos[idx]);

		this->listCPoselets_.push_back(newPoselet);
		newActivePoselets.push_back(&this->listCPoselets_.back());
	}

	//------------------------------------------------
	// MATCHING STEP 04: POSELET TERMINATION
	//------------------------------------------------
	for (std::deque<CPoselet*>::iterator poseIter = newPendingPoselets.begin(); poseIter != newPendingPoselets.end();)
	{
		if ((*poseIter)->nEndFrame + stParam_.nMaxPendingFrame < nCurrentFrameIdx_)
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
void CActionClassifier::Detect(std::string _curModelPath, /*std::deque<CAction*>  _testActions,*/ std::deque<CPoselet*> _activePoselets, hj::CTrackResult *_curTrackResult)
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
		if (res) 
		{
			for (int index = 0; index < _curTrackResult->objectInfos.size(); index++)
			{
				
				if (_curTrackResult->objectInfos.at(index).id != (*poseletIter)->id) { continue; }
				_curTrackResult->objectInfos.at(index).bActionDetect = true;
			}
		}
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
void CActionClassifier::Run(/*hj::KeyPointsSet _curKeypoints*/ hj::CTrackResult *_curTrackResult, cv::Mat _curFrame, int frameIdx, std::string _strModelPath)
{
	nCurrentFrameIdx_ = frameIdx;
	matDetectResult_ = _curFrame.clone();
	curTrackResult_ = *_curTrackResult;
	UpdatePoseletUsingTrack();
	

	
	if (activePoselets_.size()) 
	{ 
		Detect(_strModelPath, activePoselets_, _curTrackResult);
		//ResultPackaging(_curTrackResult);
	}
	

	/* visualize */
	if (bVisualizeResult_) { Visualize(_curTrackResult); }

	EliminationStepSize();
}


// Normalize pose
void CActionClassifier::Normalize()
{
}


// Result visualization
void CActionClassifier::Visualize(hj::CTrackResult *_curTrackResult)
{
	/* frame information */
	char strFrameInfo[100];
	sprintf_s(strFrameInfo, "%04d", this->nCurrentFrameIdx_);
	cv::rectangle(matDetectResult_, cv::Rect(5, 2, 100, 22), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(matDetectResult_, strFrameInfo, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));


	for (int index = 0; index < _curTrackResult->objectInfos.size(); index++)
	{
		hj::CObjectInfo curObjInfo = _curTrackResult->objectInfos.at(index);

		if (!curObjInfo.bActionDetect) { continue; }
		
		cv::rectangle(
			matDetectResult_,
			curObjInfo.box,
			cv::Scalar(0, 0, 255), 1);

		char strDetectResult[100];
		sprintf_s(strDetectResult, "Throwing Detected(%d person)", curObjInfo.id);
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



}