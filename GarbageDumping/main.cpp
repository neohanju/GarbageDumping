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
#include "ActionClassifier.h"


#include "kcftracker.hpp"
#include "BGModeling.h"
#include "ThrowDetector.h"


#define KEYPOINTS_BASE_PATH ("D:\\etri_data\\pose_text")
#define VIDEO_BASE_PATH ("D:\\etri_data\\numbering\\numbering")
#define RESULT_PATH ("D:\\Result\\GarbageDumpingResult")
#define HEATMAP_PATH ("D:\\etri_data\\output_heatmaps")
#define TRAINED_MODEL_PATH ("D:\\workspace\\github\\GarbageDumping")

#define TARGET_VIDEO ("172") // for staying: 165. start from 165.
#define START_FRAME_INDEX (0)
#define END_FRAME_INDEX (-1)

// 208 Need to FIX...


#define RESIZE_RATIO_BG (4)


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
//	trackParams.bVisualize = true;
//	trackParams.bVideoRecord = true;  // <- To recoder the result visualization.
	trackParams.bVisualize = true;
	trackParams.bVideoRecord = false;  // <- To recoder the result visualization.


	trackParams.strVideoRecordPath = std::string(RESULT_PATH) + "\\" + std::string(TARGET_VIDEO);
	hj::CMTTracker cTracker;      // <- The instance of a multi-target tracker.
	cTracker.Initialize(trackParams);
	

	// Bacgkround Modeling Init
	CProbModel m_ProbModel;
	IplImage* ipl_imgRe = NULL;
	IplImage* ipl_Age = NULL;
	Mat m_Age_map;
	float fZero_ratio = 0.1f;
	IplImage* ipl_foreground = NULL;
	// 170424.
	IplImage* ipl_FGS = NULL;
	IplImage* ipl_ROI = NULL;
	Mat m_fg_map;
	bool bInit_vibe = false;
	bool bVibe_on = false;
	bool bEventDetect = false;
	int nEventCnt = 0;
	int nStopCnt = 0;
	Mat m_fg_road;



	ipl_imgRe = cvCreateImage(cvSize(imageWidth / RESIZE_RATIO_BG, imageHeight / RESIZE_RATIO_BG), IPL_DEPTH_8U, 3);
	ipl_foreground = cvCreateImage(cvSize(imageWidth / RESIZE_RATIO_BG, imageHeight / RESIZE_RATIO_BG), IPL_DEPTH_8U, 1);

	// KCT Ininialize
	//bool HOG = true;
	//bool FIXEDWINDOW = false;
	//bool MULTISCALE = false;
	//bool SILENT = false;
	//bool LAB = true;
	bool bTrackInit = false;
	bool bTrackReON = true;
	bool bTrackOff = false;




	// Throwind Detection Initializer...
	CThrowDetector cThwDetector;
	cThwDetector.init(imageWidth, imageHeight);

	//---------------------------------------------------
	// MAIN LOOP FOR TRACKING
	//---------------------------------------------------
	std::string strKeypointsBasePath = std::string(KEYPOINTS_BASE_PATH) + "\\" + std::string(TARGET_VIDEO);
	std::string strFilePath;  // <- temporary file path for this and that
	cv::Mat matCurFrame;
	cv::Mat matCurFrame_re; // resized image for background modeling
	hj::KeyPointsSet curKeyPoints;

	// Pose Heatmap related.
	std::string strHeatPath = std::string(HEATMAP_PATH);
	std::string strFilePath_heat;

	// DISPLAY
	cv::Mat matDisp;


	// Set the ROI according to Video
	// hard coding... 
	std::string vid_num = std::string(TARGET_VIDEO);

	
	if (vid_num == "165") // if semi-colon is used, error occurs...
	{
		
		cThwDetector.set_LHand();
		cThwDetector.set_ROI(454, 308, 611, 465);
		
		
	}

	if (vid_num == "168") // if semi-colon is used, error occurs...
	{
	//	bROI = true;
	//	ROI_LT = Point(454, 308); ROI_RB = Point(611, 465);
	//	bRHand = false;
	}
	if (vid_num == "170") // if semi-colon is used, error occurs...
	{
		//	bROI = true;
		//	ROI_LT = Point(454, 308); ROI_RB = Point(611, 465);

		cThwDetector.set_LHand();
	}

	if (vid_num == "172") // if semi-colon is used, error occurs...
	{
		//	bROI = true;
		//	ROI_LT = Point(454, 308); ROI_RB = Point(611, 465);

		cThwDetector.set_LHand();
	}

	if (vid_num == "173") // if semi-colon is used, error occurs...
	{
		//	bROI = true;
		//	ROI_LT = Point(454, 308); ROI_RB = Point(611, 465);
			
		cThwDetector.set_LHand();
	}
	
	if (vid_num == "180") // if semi-colon is used, error occurs...
	{
		//	bROI = true;
		//	ROI_LT = Point(454, 308); ROI_RB = Point(611, 465);

//		cThwDetector.set_LHand();
		cThwDetector.m_ID = 1;
	}



	if (vid_num == "181") // if semi-colon is used, error occurs...
	{
		cThwDetector.set_ROI(457, 358, 1193, 718);
	}


	if (vid_num == "182") // if semi-colon is used, error occurs...
	{
		cThwDetector.set_ROI(701, 252, 842, 510);
	}

	if (vid_num == "183") // if semi-colon is used, error occurs...
	{
		cThwDetector.set_ROI(574, 290, 974, 473);

	}
	if (vid_num == "184") // if semi-colon is used, error occurs...
	{
		cThwDetector.set_ROI(570, 317, 886, 450);
	//	cThwDetector.m_ID = 1;
	//	bRHand = false;
	}

	if (vid_num == "187") // if semi-colon is used, error occurs...
	{
	//	cThwDetector.set_LHand();

	}

	if (vid_num == "189") // if semi-colon is used, error occurs...
	{
		cThwDetector.set_LHand();

	}
	if (vid_num == "190") // if semi-colon is used, error occurs...
	{
		//cThwDetector.set_LHand();
		cThwDetector.m_ID = 0;
	}
	if (vid_num == "192") // if semi-colon is used, error occurs...
	{
		cThwDetector.set_LHand();
		// Need to Debug... Too Sensitive to initial search area...171116.

	}
	if (vid_num == "197") // if semi-colon is used, error occurs...
	{
		cThwDetector.set_LHand();
		// Need to Debug... Too Sensitive to initial search area...171116.

	}
	if (vid_num == "210") // if semi-colon is used, error occurs...
	{
		cThwDetector.set_LHand();
		// Need to Debug... Too Sensitive to initial search area...171116.

	}
	if (vid_num == "211") // if semi-colon is used, error occurs...
	{
		//cThwDetector.set_LHand();
		cThwDetector.m_ID = 1;
	}
	if (vid_num == "212") // if semi-colon is used, error occurs...
	{
		//cThwDetector.set_LHand();
		cThwDetector.m_ID = 2;
		cThwDetector.set_LHand();
	}

	//if (vid_num == "219") // if semi-colon is used, error occurs...
	//{
	//	//cThwDetector.set_LHand();
	//	//cThwDetector.m_ID = 2;
	//	cThwDetector.set_LHand();
	//}

	// Action Classification Init
	jm::stParamAction actionParams;
	jm::CActionClassifier cClassifier;
	std::string strModelPath = std::string(TRAINED_MODEL_PATH) + "\\" + "train.xml";
	cClassifier.Initialize(actionParams, strModelPath);


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



		//// Load the Pose Background Heatmap
		//strFilePath_heat = strHeatPath + "\\"
		//	+ std::string(TARGET_VIDEO) + hj::FormattedString("_%012d_heatmaps.png", fIdx);

		//Mat Heatmap = imread(strFilePath_heat);
		//cv::cvtColor(Heatmap, Heatmap, CV_BGR2GRAY);
		//cv::imshow("Heatmap", Heatmap);

		//// Visualize SAVE
		//matCurFrame.copyTo(matDisp);

		//// Background modeling for fixed camera
		//Mat matCurFrame_re;
		//resize(matCurFrame, matCurFrame_re, Size(imageWidth / RESIZE_RATIO_BG, imageHeight / RESIZE_RATIO_BG), 0, 0, 1);
		//ipl_imgRe = cvCloneImage(&(IplImage)matCurFrame_re);
		//double s = sum(matCurFrame_re).val[0]; // for skipping the black frames

		//if (bInit_vibe == false && s > 0)
		//{
		//	m_ProbModel.init(ipl_imgRe, 1); // To do: remove all ipl_image type.
		//	m_Age_map = Mat::ones(matCurFrame_re.rows, matCurFrame_re.cols, CV_32FC1) * 16;
		//	ipl_Age = cvCloneImage(&(IplImage)m_Age_map);
		//	fZero_ratio = -1.0;
		//	bInit_vibe = true;
		//	m_fg_road = Mat::zeros(matCurFrame_re.rows, matCurFrame_re.cols, CV_8UC1);

		//}
		//else if (bInit_vibe == true)
		//{
		//	m_ProbModel.m_Cur = ipl_imgRe;
		//	m_ProbModel.update_vibe(ipl_foreground, ipl_Age, ipl_FGS, fIdx, fZero_ratio);
		//	m_fg_map = cvarrToMat(ipl_foreground);
		//	// 170926. Gaussian Smoothing for Clear background...
		//	//	GaussianBlur(m_fg_map, m_fg_map, Size(3, 3), 0, 0);
		//	//	medianBlur(m_fg_map, m_fg_map, 3);
		//	
		//	 
		//	cv::imshow("foreground", m_fg_map);
		//	bVibe_on = true;
		//}

	
		//// Throwing detection mode: 1. KCF-based, 2. Mask-based
		//// How to combine?? 


		//
		//// Throwing Detection using Joint position and Foreground
		//if (bVibe_on == true) 
		//{
		//	// 0. KeyPoints Save... -> 171110. Point change according to Tracking...  
		//	int nObj = trackResult.objectInfos.size();

		//	// Connect to Multiple Pedestrian Class... How to support multiple people... 
		//	cThwDetector.m_bHandSet = false;
		//	for (int n = 0; n < nObj; n++)
		//	{
		//		int nID = trackResult.objectInfos.at(n).id;

		//		if (cThwDetector.m_ID < 0) // if ID is not initialized...
		//			cThwDetector.m_ID = nID;
		//			

		//		if (cThwDetector.m_ID == nID)
		//		{
		//			bool bDetect = cThwDetector.Detect(trackResult, n, matCurFrame, m_fg_map, Heatmap);
		//		}
		//	}	
		//}

		//// Dispaly mode...


		//// Display the ALL result
		//cThwDetector.m_DispMat.release();


		//Action Classification
		
		cClassifier.Run(&trackResult, matCurFrame, fIdx);        //model path (?)

		if (trackResult.objectInfos.size())
		{
			if (trackResult.objectInfos.at(0).bActionDetect)
			{
				printf("Detect");
			}
			
		}
	}

	
	
	return 0;
}


//()()
//('')HAANJU.YOO
