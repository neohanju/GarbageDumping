#include "ResultIntegration.h"
#include <opencv2/imgproc/imgproc.hpp>



ResultIntegration::ResultIntegration()
	:bInit_(false)
{
}


ResultIntegration::~ResultIntegration()
{
	Finalize();
}

void ResultIntegration::Initialize()
{
	if (bInit_) { Finalize(); }

	//stParam_ = _stParam;
	bInit_ = true;

	// visualization related
	//bVisualizeResult_ = stParam_.bVisualize;
	strVisWindowName_ = "Final result";

}

void ResultIntegration::Finalize()
{
	if (!bInit_) { return; }

	bInit_ = false;

	/* visualize related */
	if (bVisualizeResult_) { cv::destroyWindow(strVisWindowName_); }
}
