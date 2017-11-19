#include "ActionClassifier.h"

namespace jm
{

CActionClassifier::CActionClassifier()
{
}


CActionClassifier::~CActionClassifier()
{
	Finalize();
}

void CActionClassifier::Initialize(stParamAction &_stParam)
{
	stParam_ = _stParam;
}

void CActionClassifier::Finalize()
{
}

void CActionClassifier::UpdatePoselet(hj::KeyPointsSet _curKeypoints, int _frameIdx)
{
	//active Poselet update
	std::deque<CPoselet*> newActivePoselet;
	std::deque<CPoselet*> newPendingPoselet;

	//---------------------------------------------------
	// MATCHING STEP 01: active Poselet <-> keypoints
	//---------------------------------------------------
	for (hj::KeyPointsSet::iterator keypointIter = _curKeypoints.begin();
		keypointIter != _curKeypoints.end();)
	{
		bool match = false;
		for(int poseletIdx = 0; poseletIdx < activePoselet.size(); poseletIdx++)
		{
			CPoselet *curPoselet = activePoselet[poseletIdx];

			if (curPoselet->id != keypointIter->jsonId) { continue; }

			curPoselet->lastUpdate = keypointIter->nFrame;
			curPoselet->nEndFrame  = keypointIter->nFrame;         // TODO: EndFrame�� last update�� �޶�� �Ѵ� interpolation �ؾ��ϴ� ������ ã������(���߿� ���� �ϳ��� ��������..)
			curPoselet->duration++;
			curPoselet->vectorPose.push_back(keypointIter->points);
			newActivePoselet.push_back(curPoselet);

			keypointIter = _curKeypoints.erase(keypointIter);
			match = true;
			break;
		}

		if (match) { continue; }
		keypointIter++;
	}
	
	// Update pending poselet
	for (std::deque<CPoselet*>::iterator poseIter = activePoselet.begin();
		poseIter != activePoselet.end(); poseIter++)
	{
		if ((*poseIter)->lastUpdate == _frameIdx) { continue; }
		newPendingPoselet.push_back((*poseIter));
	}


	//---------------------------------------------------
	// MATCHING STEP 02: pending Poselet <-> keypoints
	//---------------------------------------------------
	for (hj::KeyPointsSet::iterator keypointIter = _curKeypoints.begin();
		keypointIter != _curKeypoints.end();)
	{
		bool match = false;
		for (int poseletIdx = 0; poseletIdx < pendingPoselet.size(); poseletIdx++)
		{
			CPoselet *curPoselet = pendingPoselet[poseletIdx];

			if (curPoselet->id != keypointIter->jsonId) { continue; }

			curPoselet->lastUpdate = keypointIter->nFrame;
			curPoselet->nEndFrame  = keypointIter->nFrame;         // TODO: EndFrame�� last update�� �޶�� �Ѵ� interpolation �ؾ��ϴ� ������ ã������(���߿� ���� �ϳ��� ��������..)
			curPoselet->duration++;
			curPoselet->vectorPose.push_back(keypointIter->points);
			newActivePoselet.push_back(curPoselet);

			//TODO: interpolation 

			keypointIter = _curKeypoints.erase(keypointIter);
			match = true;
			break;
		}

		if (match) { continue; }
		keypointIter++;
	}

	// Update pending poselet
	for (std::deque<CPoselet*>::iterator poseIter = pendingPoselet.begin();
		poseIter != pendingPoselet.end(); poseIter++)
	{
		if ((*poseIter)->lastUpdate == _frameIdx) { continue; }
		newPendingPoselet.push_back((*poseIter));
	}


	//---------------------------------------------------
	// MATCHING STEP 03: Generation Poselet
	//---------------------------------------------------
	for (int idx = 0; idx < _curKeypoints.size(); idx++)
	{
		CPoselet newPoselet;

		newPoselet.id          = _curKeypoints[idx].jsonId;
		newPoselet.lastUpdate  = _curKeypoints[idx].nFrame;
		newPoselet.nStartFrame = _curKeypoints[idx].nFrame;
		newPoselet.nEndFrame   = _curKeypoints[idx].nFrame;
		newPoselet.lastUpdate  = _curKeypoints[idx].nFrame;
		newPoselet.vectorPose.push_back(_curKeypoints[idx].points);

		newActivePoselet.push_back(&newPoselet);
	}

	// inactive���� active�� ������Ʈ �ɶ� update�� ������ -1�� ������ �ֱ� ������ ���� ������ interpolation�ϴ� �۾�
	//------------------------------------------------
	// POSELET TERMINATION
	//------------------------------------------------
	for (std::deque<CPoselet*>::iterator poseIter = newPendingPoselet.begin(); poseIter != newPendingPoselet.end();)
	{
		if ((*poseIter)->lastUpdate + stParam_.nMaxPendingFrame < _frameIdx)
		{
			poseIter = newPendingPoselet.erase(poseIter);
			continue;
		}
		poseIter++;
	}

	activePoselet  = newActivePoselet;
	pendingPoselet = newPendingPoselet;
	
}

// Load python SVM train model 
void CActionClassifier::Load()
{
}

// Train in opencv SVM model 
void CActionClassifier::Train()
{
}

// when input action vector 
void CActionClassifier::Run()
{
}

// Normalize pose
void CActionClassifier::Normalize()
{
}

// Result visualization
void CActionClassifier::Visualize()
{
}

}