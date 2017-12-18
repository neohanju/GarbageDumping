#pragma once
#include <opencv2/highgui/highgui.hpp>

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

	void Initialize();
	void Finalize();
	void Run();

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
public:
	bool bInit_;

	/* visualization related */
	bool             bVisualizeResult_;
	std::string      strVisWindowName_;

	/*Detection relate*/
	bool bSVMResult;
	bool bThrowResult;

};
