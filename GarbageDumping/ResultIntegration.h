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
	void Run(hj::CTrackResult _trackResult, jm::CActionResultSet _actionResult);
	void Visualize();

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
public:
	bool bInit_;
	stParamResult stResultParam_;

	/* visualization related */
	bool             bVisualizeResult_;
	std::string      strVisWindowName_;

	/*Detection relate*/
	bool bSVMResult;
	bool bThrowResult;

};
}