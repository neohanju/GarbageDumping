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
	//bVisualizeResult_ = stParam_.bVisualize;
	strVisWindowName_ = "Final result";

}

void CResultIntegration::Finalize()
{
	if (!bInit_) { return; }

	bInit_ = false;

	/* visualize related */
	if (bVisualizeResult_) { cv::destroyWindow(strVisWindowName_); }
}

void CResultIntegration::Run(hj::CTrackResult _trackResult, jm::CActionResultSet _actionResult)
{
	if (stResultParam_.bVisualize) { Visualize(); }
}

void CResultIntegration::Visualize()
{

}

}