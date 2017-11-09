/**************************************************************************
* Title        : Online Multi-Camera Multi-Target Tracking Algorithm
* Author       : Haanju Yoo
* Initial Date : 2013.08.29 (ver. 0.9)
* Version Num. : 1.0 (since 2016.09.06)
* Description  :
*	The implementation of the paper named "Online Scheme for Multiple
*	Camera Multiple Target Tracking Based on Multiple Hypothesis 
*	Tracking" at IEEE transactions on Circuit and Systems for Video 
*	Technology (TCSVT).
***************************************************************************
                                            ....
                                           W$$$$$u
                                           $$$$F**+           .oW$$$eu
                                           ..ueeeWeeo..      e$$$$$$$$$
                                       .eW$$$$$$$$$$$$$$$b- d$$$$$$$$$$W
                           ,,,,,,,uee$$$$$$$$$$$$$$$$$$$$$ H$$$$$$$$$$$~
                        :eoC$$$$$$$$$$$C""?$$$$$$$$$$$$$$$ T$$$$$$$$$$"
                         $$$*$$$$$$$$$$$$$e "$$$$$$$$$$$$$$i$$$$$$$$F"
                         ?f"!?$$$$$$$$$$$$$$ud$$$$$$$$$$$$$$$$$$$$*Co
                         $   o$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                 !!!!m.*eeeW$$$$$$$$$$$f?$$$$$$$$$$$$$$$$$$$$$$$$$$$$$U
                 !!!!!! !$$$$$$$$$$$$$$  T$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                  *!!*.o$$$$$$$$$$$$$$$e,d$$$$$$$$$$$$$$$$$$$$$$$$$$$$$:
                 "eee$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$C
                b ?$$$$$$$$$$$$$$**$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$!
                Tb "$$$$$$$$$$$$$$*uL"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
                 $$o."?$$$$$$$$F" u$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                  $$$$en '''    .e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
                   $$$B*  =*"?.e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$F
                    $$$W"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                     "$$$o#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    R: ?$$$W$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" :!i.
                     !!n.?$???""''.......,''''''"""""""""""''   ...+!!!
                      !* ,+::!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*'
                      "!?!!!!!!!!!!!!!!!!!!~ !!!!!!!!!!!!!!!!!!!~'
                      +!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!?!'
                    .!!!!!!!!!!!!!!!!!!!!!' !!!!!!!!!!!!!!!, !!!!
                   :!!!!!!!!!!!!!!!!!!!!!!' !!!!!!!!!!!!!!!!! '!!:
                .+!!!!!!!!!!!!!!!!!!!!!~~!! !!!!!!!!!!!!!!!!!! !!!.
               :!!!!!!!!!!!!!!!!!!!!!!!!!.':!!!!!!!!!!!!!!!!!:: '!!+
               "~!!!!!!!!!!!!!!!!!!!!!!!!!!.~!!!!!!!!!!!!!!!!!!!!.'!!:
                   ~~!!!!!!!!!!!!!!!!!!!!!!! ;!!!!~' ..eeeeeeo.'+!.!!!!.
                 :..    '+~!!!!!!!!!!!!!!!!! :!;'.e$$$$$$$$$$$$$u .
                 $$$$$$beeeu..  '''''~+~~~~~" ' !$$$$$$$$$$$$$$$$ $b
                 $$$$$$$$$$$$$$$$$$$$$UU$U$$$$$ ~$$$$$$$$$$$$$$$$ $$o
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$. $$$$$$$$$$$$$$$~ $$$u
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$! $$$$$$$$$$$$$$$ 8$$$$.
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$X $$$$$$$$$$$$$$'u$$$$$W
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$! $$$$$$$$$$$$$".$$$$$$$:
                 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  $$$$$$$$$$$$F.$$$$$$$$$
                 ?$$$$$$$$$$$$$$$$$$$$$$$$$$$$f $$$$$$$$$$$$' $$$$$$$$$$.
                  $$$$$$$$$$$$$$$$$$$$$$$$$$$$ $$$$$$$$$$$$$  $$$$$$$$$$!
                  "$$$$$$$$$$$$$$$$$$$$$$$$$$$ ?$$$$$$$$$$$$  $$$$$$$$$$!
                   "$$$$$$$$$$$$$$$$$$$$$$$$Fib ?$$$$$$$$$$$b ?$$$$$$$$$
                     "$$$$$$$$$$$$$$$$$$$$"o$$$b."$$$$$$$$$$$  $$$$$$$$'
                    e. ?$$$$$$$$$$$$$$$$$ d$$$$$$o."?$$$$$$$$H $$$$$$$'
                   $$$W.'?$$$$$$$$$$$$$$$ $$$$$$$$$e. "??$$$f .$$$$$$'
                  d$$$$$$o "?$$$$$$$$$$$$ $$$$$$$$$$$$$eeeeee$$$$$$$"
                  $$$$$$$$$bu "?$$$$$$$$$ 3$$$$$$$$$$$$$$$$$$$$*$$"
                 d$$$$$$$$$$$$$e. "?$$$$$:'$$$$$$$$$$$$$$$$$$$$8
         e$$e.   $$$$$$$$$$$$$$$$$$+  "??f "$$$$$$$$$$$$$$$$$$$$c
        $$$$$$$o $$$$$$$$$$$$$$$F"          '$$$$$$$$$$$$$$$$$$$$b.0
       M$$$$$$$$U$$$$$$$$$$$$$F"              ?$$$$$$$$$$$$$$$$$$$$$u
       ?$$$$$$$$$$$$$$$$$$$$F                   "?$$$$$$$$$$$$$$$$$$$$u
        "$$$$$$$$$$$$$$$$$$"                       ?$$$$$$$$$$$$$$$$$$$$o
          "?$$$$$$$$$$$$$F                            "?$$$$$$$$$$$$$$$$$$
             "??$$$$$$$F                                 ""?3$$$$$$$$$$$$F
                                                       .e$$$$$$$$$$$$$$$$'
                                                      u$$$$$$$$$$$$$$$$$
                                                     '$$$$$$$$$$$$$$$$"
                                                      "$$$$$$$$$$$$F"
                                                        ""?????""

**************************************************************************/

#include <sstream>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "MTTracker.h"

#define KEYPOINTS_BASE_PATH ("D:\\etri_data\\pose_text")
#define VIDEO_BASE_PATH ("D:\\etri_data\\numbering\\numbering")
#define RESULT_PATH ("D:\\etri_data\\GarbageDumpingResult")

#define TARGET_VIDEO ("182")
#define START_FRAME_INDEX (0)
#define END_FRAME_INDEX (-1)


int main(int argc, char** argv)
{	
	// read video info
	std::string strVideoPath = std::string(VIDEO_BASE_PATH) + "\\" + std::string(TARGET_VIDEO) + ".mp4";
	cv::VideoCapture *pVideoCapture = new cv::VideoCapture(strVideoPath);
	int nLastFrameIndex = END_FRAME_INDEX < 0 ? 
		(int)pVideoCapture->get(CV_CAP_PROP_FRAME_COUNT) : 
		std::min((int)END_FRAME_INDEX, (int)pVideoCapture->get(CV_CAP_PROP_FRAME_COUNT));
	int imageWidth = (int)pVideoCapture->get(CV_CAP_PROP_FRAME_WIDTH), 
		imageHeight = (int)pVideoCapture->get(CV_CAP_PROP_FRAME_HEIGHT);


	//---------------------------------------------------
	// TRACKER INITIATION
	//---------------------------------------------------
	hj::CTrackResult trackResult;     // <- The tracking result will be saved here
	hj::stParamTrack trackParams;     // <- Contains whole parameters of tracking module. Using default values is recommended.
	trackParams.nImageWidth = imageWidth;
	trackParams.nImageHeight = imageHeight;
	trackParams.dImageRescale = 1.0;  // <- Heavy influence on the speed of the algorithm.
	trackParams.bVisualize = true;
	trackParams.bVideoRecord = true;  // <- To recoder the result visualization.
	trackParams.strVideoRecordPath = std::string(RESULT_PATH) + "\\" + std::string(TARGET_VIDEO);
	hj::CMTTracker cTracker;      // <- The instance of a multi-target tracker.
	cTracker.Initialize(trackParams);
	

	//---------------------------------------------------
	// MAIN LOOP FOR TRACKING
	//---------------------------------------------------
	std::string strKeypointsBasePath = std::string(KEYPOINTS_BASE_PATH) + "\\" + std::string(TARGET_VIDEO);
	std::string strFilePath;  // <- temporary file path for this and that
	cv::Mat matCurFrame;
	hj::KeyPointsSet curKeyPoints;

	for (int fIdx = START_FRAME_INDEX; fIdx < nLastFrameIndex; fIdx++)
	{
		// Grab frame image
		(*pVideoCapture) >> matCurFrame;

		// Read keypoints
		strFilePath = strKeypointsBasePath + "\\" 
			+ std::string(TARGET_VIDEO) + hj::FormattedString("_%012d_keypoints.txt", fIdx);
		curKeyPoints = hj::ReadKeypoints(strFilePath);

		// Track targets between consecutive frames
		trackResult = cTracker.Track(curKeyPoints, matCurFrame, fIdx);
		/*
		for (std::vector<hj::CObjectInfo>::iterator objectInfoIter = trackResult.objectInfos.begin();
			objectInfoIter != trackResult.objectInfos.end();
			objectInfoIter++)
		{
			printf("%f", &objectInfoIter->keyPoint[0]);
		}
		*/
	}

	return 0;
}


//()()
//('')HAANJU.YOO
