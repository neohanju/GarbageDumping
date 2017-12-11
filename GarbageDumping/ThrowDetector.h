
/************************************************************************/
/* Basic Includes                                                       */
/************************************************************************/
#pragma once

#include <vector>
#include <queue>
#include <list>


#include	<iostream>
#include	<cstdlib>
#include	<cstring>
#include	<vector>
#include	<algorithm>

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "haanju_utils.hpp"

#include "kcftracker.hpp"


/************************************************************************/
/*  Necessary includes for this Algorithm                               */
/************************************************************************/

#include "params.h"
using namespace cv;


class CThrowDetector 
{

public: 
	float m_fDist_mean[18];
	float m_fDist_std[18];
	float m_fDist_mean_obs[18];
	float m_fDist_std_obs[18];

	Point2d m_ROI_LT;
	Point2d m_ROI_RB;

	bool m_bROI = false;
	bool m_bRHand = true; 

	// For visualize matrix...
	Mat m_DispMat;

//	bool m_bTrackInit;

	// Image realted Mat
	Mat m_curFrame;
	Mat m_fore_up, m_heatmap_up;

	// Road region estimation...
	Mat m_fg_accum;
	Mat m_heat_accum;
	Mat m_mov_ped_accum;
	Mat m_mov_ped_accum_tmp;
	Mat m_prev_mov_ped;

	// ID for multiple object
	int m_ID;
	bool m_bHandSet;
	int m_ID_index;

	// Input Keypoints..
	float _patch_w;
	float _patch_h;
	float _LH_x, _LH_y, _LH_c;
	float _min_fore_area;



	// KCF Tracker.
	KCFTracker tracker;
	Rect KCF_result;
	bool m_bTrackInit;

	// Decision
	bool m_bROI_warning;
	bool m_bThw_warning;
	bool m_bThw_warning2;	
	bool m_bFirstStart;


	int imageWidth;
	int imageHeight;

	bool m_bKCFRectInit;
	Rect m_KCF_rect_init;
	Rect m_L_hand;
	int m_nReCnt;

public:
	CThrowDetector(void);
	~CThrowDetector(void);
	void uninit(void);
	void init(int width, int height);
	void ReInit(void);
	bool run_proposal(Mat curFrame, Mat foreground, Mat Heatmap);
	bool run_decision(Rect track_box, hj::CTrackResult trackResult);

	void set_ROI(int LT_x, int LT_y, int RB_x, int RB_y);
	void set_LHand(void);
	void Visualize_ROI(void);
	void carryingObjectProposal();
	void FindCarryingObject(Rect region_rect, int min_fore_area, float th_heat_m);
	float MeasureJointBgProb(Rect region_rect);
	bool KCF_ReInit(Rect track_box);
	void Visualize(bool bROI_, bool bThw1, bool bThw2);

	void set_keypoints(hj::CTrackResult trackResult, int idx);

	bool Detect(hj::CTrackResult trackResult, int idx, Mat matCurFrame, Mat m_fg_map, Mat Heatmap);

	void region_accumulate(Mat mat_fg, Mat mat_heat);


};
