//#include "stdafx.h"
#include "ThrowDetector.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



CThrowDetector::CThrowDetector(void)
{



}

CThrowDetector::~CThrowDetector()
{
	uninit();
}

void CThrowDetector::uninit(void)
{

}

void CThrowDetector::init(int width, int height)
{

	memset(m_fDist_mean, 0, sizeof(float) * 18);
	memset(m_fDist_std, 10, sizeof(float) * 18);
	memset(m_fDist_mean_obs, 0, sizeof(float) * 18);
	memset(m_fDist_std_obs, 10, sizeof(float) * 18);
	



	m_ROI_LT = Point(0, 0);
	m_ROI_RB = Point(0, 0);
	m_bROI = false;
	m_bRHand = true;
	
	imageWidth = width;
	imageHeight = height;

	// KCF related...
	m_bKCFRectInit = false;
	m_bTrackInit = false;



	m_nReCnt = 0;

	// Tracking ID... 
	m_ID = -1; // 171110.
	m_bHandSet = false;
	m_ID_index = -1; 


	// Decision
	m_bROI_warning = false;
	m_bThw_warning = false;
	m_bThw_warning2 = false;
	m_bFirstStart = false;




}
void CThrowDetector::ReInit(void)
{

	memset(m_fDist_mean, 0, sizeof(float) * 18);
	memset(m_fDist_std, 10, sizeof(float) * 18);
	m_nReCnt = 0;

}




bool CThrowDetector::run_proposal(Mat curFrame, Mat foreground,  Mat Heatmap)
{
	m_bKCFRectInit = false;
	
	if (m_nReCnt > 100)
		ReInit();


	m_curFrame = curFrame;
	curFrame.copyTo(m_DispMat); // copy to member variable
 //	Visualize_ROI();

	// Match the foreground / heatmap size to input image Size.
	resize(foreground, m_fore_up, Size(imageWidth, imageHeight), 0, 0, INTER_CUBIC);
	resize(Heatmap, m_heatmap_up, Size(imageWidth, imageHeight), 0, 0, INTER_CUBIC);

	// Extract Initial Track Region using Mask & Heatmap info.
	if (m_bHandSet)
		carryingObjectProposal();

		 


	// Visualize... 
//	if (m_bKCFRectInit == true)
//		rectangle(m_DispMat, Point(m_KCF_rect_init.x, m_KCF_rect_init.y), Point(m_KCF_rect_init.x + m_KCF_rect_init.width, m_KCF_rect_init.y + m_KCF_rect_init.height), Scalar(0, 0, 255), 1, LINE_AA);

	if (m_bKCFRectInit == false)
		m_nReCnt++;



	return m_bKCFRectInit;
	

}

void CThrowDetector::set_ROI(int LT_x, int LT_y, int RB_x, int RB_y)
{
	m_ROI_LT = Point(LT_x, LT_y);
	m_ROI_RB = Point(RB_x, RB_y);

	m_bROI = true;

}

void CThrowDetector::set_LHand(void)
{
	m_bRHand = false;

}

void CThrowDetector::Visualize_ROI()
{
	if (m_bROI)
	{
		rectangle(m_DispMat, m_ROI_LT, m_ROI_RB, Scalar(255, 255, 255), 1, LINE_AA);
	}
}


void CThrowDetector::carryingObjectProposal()
{
	float xMin = _LH_x;
	float yMin = _LH_y;
	float conf = _LH_c;

	if (xMin + _patch_w < imageWidth && yMin + _patch_h < imageHeight && xMin > 0 && yMin > 0)
	{

	


		// Visualize: Initial Rectangle Drawing around Hand...
		rectangle(m_DispMat, Point(xMin, yMin), Point(xMin + _patch_w, yMin + _patch_h), Scalar(255, 0, 0), 1, LINE_AA);
		
		
		Mat fore_patch = m_fore_up(Rect(xMin, yMin, _patch_w, _patch_h));
		Mat heatmap_patch = m_heatmap_up(Rect(xMin, yMin, _patch_w, _patch_h));

		
		//cv::namedWindow("patch_fore", WINDOW_NORMAL);
		//cv::resizeWindow("patch_fore", _patch_w * 2, _patch_h * 2);
		//cv::imshow("patch_fore", fore_patch);

		//cv::namedWindow("patch_heatmap", WINDOW_NORMAL);
		//cv::resizeWindow("patch_heatmap", _patch_w * 2, _patch_h * 2);
		//cv::imshow("patch_heatmap", heatmap_patch);
		
	
		float heat_m = MeasureJointBgProb(Rect(xMin, yMin, _patch_w, _patch_h));

		if (conf > 0.5 && heat_m > 0.5)
		{
			FindCarryingObject(Rect(xMin, yMin, _patch_w, _patch_h), _min_fore_area, 0.5); // 171108. Simply high threshold can solve the prob?? Tradeoff between missing event??
		}


	}
	



	


}


void CThrowDetector::FindCarryingObject(Rect region_rect, int min_fore_area, float th_heat_m)
{

	Mat fore_patch = m_fore_up(region_rect);

	double threshold = 200; // 0 - 255 binarize....
	Mat fg_map_bin;
	Mat heatmap_bin;

	compare(fore_patch, threshold, fg_map_bin, CMP_GT);

	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	findContours(fg_map_bin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0)); // Error in debug mode...

	double max_moment = -1.0;
	int max_cc = -1;

	// get the moment
	std::vector<Moments> mu(contours.size()); // moments for each contour
	std::vector<Rect> boundRect(contours.size());
	std::vector<std::vector<Point> > contours_poly(contours.size());

	if (contours.size() != 0)
	{
		// contour display
		RNG rng(12345);
		Mat drawing;
		fore_patch.copyTo(drawing);
		bool bPincnt = false;

		int contour_num = contours.size();
		double tmp_moment = 0.0;

		for (int cc = 0; cc < contour_num; cc++)
		{
			tmp_moment = moments(contours[cc], false).m00;
			if (tmp_moment > min_fore_area && tmp_moment > max_moment)
			{
				// extract region 
				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

				approxPolyDP(Mat(contours[cc]), contours_poly[cc], 3, true);
				boundRect[cc] = boundingRect(Mat(contours_poly[cc]));
				//				drawContours(drawing, contours_poly, cc, color, 2, 8, hierarchy, 0, Point());
				//				rectangle(drawing, boundRect[cc].tl(), boundRect[cc].br(), color, 2, 8, 0);

				//				namedWindow("contour", WINDOW_NORMAL);
				//				resizeWindow("contour", frame_re.cols * 2, frame_re.rows * 2);
				//				imshow("contour", drawing);

				max_cc = cc;
				max_moment = tmp_moment;


			}

		}

	}

	if (max_cc >= 0)
	{

		m_KCF_rect_init = Rect(region_rect.x + boundRect[max_cc].x, region_rect.y + boundRect[max_cc].y, boundRect[max_cc].width, boundRect[max_cc].height);

		// Visualize... 
		Mat heat_ex = m_heatmap_up(m_KCF_rect_init);

		//namedWindow("contour2", WINDOW_NORMAL);
		//resizeWindow("contour2", m_KCF_rect_init.width * 2, m_KCF_rect_init.height * 2);
		//cv::imshow("contour2", heat_ex);


		float jBGprob = 0.0;
		jBGprob = MeasureJointBgProb(m_KCF_rect_init);
		if (jBGprob > th_heat_m)
		{
			// Visualize... 		
			rectangle(m_DispMat, Point(m_KCF_rect_init.x, m_KCF_rect_init.y), Point(m_KCF_rect_init.x + m_KCF_rect_init.width, m_KCF_rect_init.y + m_KCF_rect_init.height), Scalar(0, 0, 255), 1, LINE_AA);


			namedWindow("contour", WINDOW_NORMAL);
			resizeWindow("contour", m_L_hand.width * 2, m_L_hand.height * 2);
			cv::imshow("contour", m_DispMat(m_L_hand));
		//	cv::waitKey();
			m_bKCFRectInit = true;
		}
			

	}


	



}

float CThrowDetector::MeasureJointBgProb(Rect region_rect)
{
	Mat heatmap_patch = m_heatmap_up(region_rect);

	// Visualize: 
//	cv::imshow("patch_heatmap", heatmap_patch);



	Mat heat_ex = m_heatmap_up(region_rect);
	double heatmap_th = 190; // 0 - 255 binarize....
	Mat heat_map_bin;
	compare(heat_ex, heatmap_th, heat_map_bin, CMP_GT);
//	cv::imshow("patch_heatmap_bin", heat_map_bin);

	return mean(heat_map_bin / 255).val[0];
}

bool CThrowDetector::run_decision(Rect track_box, hj::CTrackResult trackResult)
{
//	Rect overlap = track_box & m_L_hand;
	FILE* fp;
	fp = fopen("debug_.txt", "a");

	bool bWriteTxt = false;

	memset(m_fDist_mean_obs, 0, sizeof(float) * 18);
	memset(m_fDist_std_obs, 10, sizeof(float) * 18);


	if (m_bHandSet == true)
	{
		// All joint distance keep and Decision is done by voting. 
		int nTotalJoint = 0;
		int nDesicionCnt = 0;

		// TEST. 
		for (int jj = 0; jj < 18; jj++)
		{

			float joint_x = trackResult.objectInfos.at(m_ID_index).keyPoint.at(jj).x;
			float joint_y = trackResult.objectInfos.at(m_ID_index).keyPoint.at(jj).y;
			float joint_c = trackResult.objectInfos.at(m_ID_index).keyPoint.at(jj).confidence;


			if (joint_c > 0.3) // confidence 0.3? 0.5? 
			{


				float KCF_cx = track_box.x + (track_box.width) / 2;
				float KCF_cy = track_box.y + (track_box.height) / 2;


				line(m_DispMat, Point(KCF_cx, KCF_cy), Point(joint_x, joint_y), Scalar(128, 255, 128));
				m_fDist_mean_obs[jj] = sqrt((joint_x - KCF_cx)*(joint_x - KCF_cx) + (joint_y - KCF_cy)*(joint_y - KCF_cy));
				m_fDist_std_obs[jj] = MAX(m_fDist_mean_obs[jj] - m_fDist_mean[jj], 0);

				//fprintf(fp, "%f %f %f %f %f ", KCF_cx, joint_x, KCF_cy, joint_y, m_fDist_mean_obs[jj]);
				
				// Init the Joint Configuration...
				if (m_fDist_mean[jj] == 0 && joint_c > 0.5)
				{
					m_fDist_mean[jj] = m_fDist_mean_obs[jj];
					m_fDist_std[jj] = m_fDist_mean_obs[jj] * 0.05;

					if (bWriteTxt)
						fprintf(fp, "%f %f %f %f -1.0 ", m_fDist_mean_obs[jj], m_fDist_std_obs[jj], m_fDist_mean[jj], m_fDist_std[jj]);
				}
				// If initialized, 
				else if (m_fDist_mean[jj] != 0)
				{
					if (m_fDist_std_obs[jj] > m_fDist_std[jj] * 2)
					{
						nDesicionCnt++;
						if (bWriteTxt)
							fprintf(fp, "%f %f %f %f 1.0 ", m_fDist_mean_obs[jj], m_fDist_std_obs[jj], m_fDist_mean[jj], m_fDist_std[jj]);
					}
					else
					{
						if (bWriteTxt)
							fprintf(fp, "%f %f %f %f 0.0 ", m_fDist_mean_obs[jj], m_fDist_std_obs[jj], m_fDist_mean[jj], m_fDist_std[jj]);
					}
						

					nTotalJoint++;
				}


				
			}
			else
			{
				if (bWriteTxt)
					fprintf(fp, "-1.0 -1.0 -1.0 -1.0 -1.0 ");
			}
				
		}

		if (bWriteTxt)
			fprintf(fp, "\n");  fclose(fp);

		// Decision...
		std::cout << nDesicionCnt << ", " << nTotalJoint << std::endl;
		if (nDesicionCnt * 3 > nTotalJoint && nTotalJoint > 8) // set the minimum joint number.
		{
			//putText(matDispKCF, "Throwing Detected", Point(10, 100), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255))
			return true;
		}
		else
		{
			// Update
			for (int jj = 0; jj < 18; jj++)
			{
				if (m_fDist_mean[jj] != 0)
				{
					float learning_rate = 0.95;
					m_fDist_mean[jj] = m_fDist_mean[jj] * learning_rate + m_fDist_mean_obs[jj] * (1.0 - learning_rate); // To Do: 

					if (m_fDist_std_obs[jj] > 0)
						m_fDist_std[jj] = m_fDist_std[jj] * learning_rate + m_fDist_std_obs[jj] * (1.0 - learning_rate); // To Do: 
				}
				
			}
			return false;

		}
	}
	return false;
	





}

void CThrowDetector::Visualize(bool bROI_, bool bThw1, bool bThw2)
{
	Visualize_ROI();
	

	if (bROI_)
		putText(m_DispMat, "Suspicious Event Detected", Point(10, 50), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0));

	if (bThw1)
		putText(m_DispMat, "Throwing Detected (KCF)", Point(10, 100), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));

	if (bThw2)
		putText(m_DispMat, "Throwing Detected (MASK)", Point(10, 150), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));

	// EVENT VISUALIZE...
	imshow("Throwing Detection", m_DispMat);

}

bool CThrowDetector::KCF_ReInit(Rect track_box)
{

	Rect overlap = track_box & m_L_hand;
	if (overlap.area() == 0)
	{
		float KCF_cx = track_box.x + (track_box.width) / 2;
		float KCF_cy = track_box.y + (track_box.height) / 2;

		float mean_obs = sqrt((m_L_hand.x - KCF_cx)*(m_L_hand.x - KCF_cx) + (m_L_hand.y - KCF_cy)*(m_L_hand.y - KCF_cy));

		if (mean_obs > m_L_hand.width * 2)
		{
			return true;
		}
		
	}
	return false;

	


}

void CThrowDetector::set_keypoints(hj::CTrackResult trackResult, int idx)
{

	

	_patch_w = trackResult.objectInfos.at(idx).box.width;
	_patch_h = trackResult.objectInfos.at(idx).box.height / 2;
	_min_fore_area = MAX(0.01 * _patch_w * _patch_h, 100);
		

	if (m_bRHand)
	{
		_LH_x = trackResult.objectInfos.at(idx).keyPoint.at(4).x - _patch_w / 5; // 171101. avoid human region & mask the object region.
		//LH_x = curKeyPoints.at(0).points.at(4).x; // 171101. avoid human region & mask the object region.
		_LH_y = trackResult.objectInfos.at(idx).keyPoint.at(4).y;
		_LH_c = trackResult.objectInfos.at(idx).keyPoint.at(4).confidence; // 171025. Left hnad - 7, Right hand - 4

	}
	else
	{
		_LH_x = trackResult.objectInfos.at(idx).keyPoint.at(7).x - _patch_w / 2;
		_LH_y = trackResult.objectInfos.at(idx).keyPoint.at(7).y;
		_LH_c = trackResult.objectInfos.at(idx).keyPoint.at(7).confidence; //
	}


	m_L_hand = Rect(_LH_x, _LH_y, _patch_w, _patch_h);



}

void  CThrowDetector::region_accumulate(Mat mat_fg, Mat mat_heat)
{
	float lr_decay = 1.0;

	// 171106. Maybe not used...
	// Foreground Accumulation Map... 
	//unsigned char* fg_data = (unsigned char*)(mat_fg.data);
	//unsigned char* fg_accum_data = (unsigned char*)(m_fg_accum.data);



	//// Caution: foreground map size and heatmap size are different...
	//if (m_fg_accum.cols <= 0)
	//	m_fg_accum = Mat::zeros(mat_fg.rows, mat_fg.cols, CV_8UC1);
	//else
	//{
	//	addWeighted(m_fg_accum, lr_decay, mat_fg, lr_decay, 0, m_fg_accum);
	//	//for (int j = 0; j < mat_fg.rows; j++) 
	//	//{
	//	//	for (int i = 0; i < mat_fg.cols;i++) 
	//	//	{
	//	//		if (fg_data[i + j*(mat_fg.cols)] > 0)
	//	//		{
	//	//			//	fg_accum_data[i + j*(mat_fg.cols)] = (1.0 - lr_decay) * fg_accum_data[i + j*(mat_fg.cols)] + lr_decay * fg_data[i + j*(mat_fg.cols)];
	//	//			fg_accum_data[i + j*(mat_fg.cols)] = 255;
	//	//		}
	//	//		else
	//	//			fg_accum_data[i + j*(mat_fg.cols)] = lr_decay * fg_accum_data[i + j*(mat_fg.cols)];
	//	//	}
	//	//}


	//}


	// Heatmap Accumulation Map...
	//unsigned char* heat_data = (unsigned char*)(mat_heat.data);
	//unsigned char* heat_accum_data = (unsigned char*)(m_heat_accum.data);

	//if (m_heat_accum.cols <= 0)
	//{
	//	mat_heat.copyTo(m_heat_accum);
	//	m_heat_accum = 255 - m_heat_accum;
	//}
	//	
	//else
	//{
	//	addWeighted(m_heat_accum, lr_decay, 255 - mat_heat, lr_decay, 0, m_heat_accum);
	//	//for (int j = 0; j < mat_heat.rows; j++)
	//	//{
	//	//	for (int i = 0; i < mat_heat.cols; i++)
	//	//	{
	//	//		if (255 - heat_data[i + j*(mat_heat.cols)] > 128)
	//	//		{
	//	//			heat_accum_data[i + j*(mat_heat.cols)] = (1.0 - lr_decay) * heat_accum_data[i + j*(mat_heat.cols)] + lr_decay * (255 - heat_data[i + j*(mat_heat.cols)]);
	//	//			//heat_accum_data[i + j*(mat_heat.cols)] = 255;
	//	//		}
	//	//		else
	//	//			heat_accum_data[i + j*(mat_heat.cols)] = lr_decay * heat_accum_data[i + j*(mat_fg.cols)];
	//	//	}
	//	//}



	//}


	//cv::imshow("Road Region FG", m_fg_accum);
	//cv::imshow("Road Region Heat", m_heat_accum);


	// Moving X Pedestrian...
	Mat heat_th;
	resize(255 - mat_heat, heat_th, cv::Size(mat_fg.cols, mat_fg.rows), 0, 0, 1);
	threshold(heat_th, heat_th, 50, 255, 0); // binary... threshold function: if 0, 255 set, if 3, the value is preserved...
//	cv::imshow("Heatmap threshold", heat_th);
//	cv::imshow("FG inverted", 255 - mat_fg);

	

	Mat mov_ped;
	cv::multiply(mat_fg, heat_th, mov_ped, 1.0 / 255.0);
	
	

	if (m_mov_ped_accum.cols <= 0)
	{
		m_mov_ped_accum = Mat::zeros(mat_fg.rows, mat_fg.cols, CV_8UC1);
		m_prev_mov_ped = Mat::zeros(mat_fg.rows, mat_fg.cols, CV_8UC1);
		m_mov_ped_accum_tmp = Mat::zeros(mat_fg.rows, mat_fg.cols, CV_8UC1);
		mov_ped = Mat::zeros(mat_fg.rows, mat_fg.cols, CV_8UC1);
	}
	else
	{
//		addWeighted(m_mov_ped_accum, lr_decay, mov_ped, lr_decay, 0, m_mov_ped_accum);
		
		unsigned char* p_mov_ped = (unsigned char*)(mov_ped.data);
		unsigned char* p_m_mov_ped_accum = (unsigned char*)(m_mov_ped_accum.data);
		unsigned char* p_m_mov_ped_accum_tmp = (unsigned char*)(m_mov_ped_accum_tmp.data);
		unsigned char* p_m_prev_mov_ped = (unsigned char*)(m_prev_mov_ped.data);
	

		cv::imshow("Accumulumation Map2", mov_ped);
		cv::imshow("Accumulumation Map3", m_prev_mov_ped);

		for (int j = 0; j < mat_fg.rows; j++)
		{
			for (int i = 0; i < mat_fg.cols; i++)
			{
				if (p_mov_ped[i + j*(mat_fg.cols)] > 100 && p_m_prev_mov_ped[i + j*(mat_fg.cols)] > 100)
				{
					p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)]++; // Count the staying pixel by consecutive foreground...
					if (p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)] > p_m_mov_ped_accum[i + j*(mat_fg.cols)]) // Compare pre-stored staying pixels
					{
						p_m_mov_ped_accum[i + j*(mat_fg.cols)] = p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)]; // Swap the staying pixels...
					}
				
				}
				else
				{
					//if (p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)] > p_m_mov_ped_accum[i + j*(mat_fg.cols)]) // Compare pre-stored staying pixels
					//{
					//	p_m_mov_ped_accum[i + j*(mat_fg.cols)] = p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)]; // Swap the staying pixels...
					//}
					p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)] = 0; // re-initialized the count...
				}
					
			}
		}
	}



	cv::imshow("Accumulumation Map", m_mov_ped_accum);

	mov_ped.copyTo(m_prev_mov_ped);

	
//	pAge_tmp[i + j*modelWidth] = MIN(pAge_tmp[i + j*modelWidth] * exp(-VAR_DEC_RATIO*MAX(0.0, pVar_tmp[i + j*modelWidth] - VAR_MIN_NOISE_T)), MAX_BG_AGE);
	
	
}

bool  CThrowDetector::Detect(hj::CTrackResult trackResult, int idx, Mat matCurFrame, Mat m_fg_map, Mat Heatmap)
{
	set_keypoints(trackResult, idx);
	m_bHandSet = true;
	m_ID_index = idx;

	if (m_bTrackInit == false)
	{
		bool bTrackPos = run_proposal(matCurFrame, m_fg_map, Heatmap);
		if (bTrackPos == true)
		{
			tracker.init(m_KCF_rect_init, matCurFrame);
			m_bTrackInit = true;
			//cv::waitKey();

		}
	}
	else if (m_bTrackInit == true)
	{
		matCurFrame.copyTo(m_DispMat);

		// 2. Track RUN. 
		KCF_result = tracker.update(matCurFrame, m_fore_up);

		// 2.2 Mask info...
		bool bTrackPos = run_proposal(matCurFrame, m_fg_map, Heatmap);


		// 3. Throwing Action Decision using Voting concept...
		m_bThw_warning = run_decision(KCF_result, trackResult);

		// 3.2. Throwing Action Decision using MASK.
		if (bTrackPos)
			m_bThw_warning2 = run_decision(m_KCF_rect_init, trackResult);
		else
			m_bThw_warning2 = false;



		// 4. IF Track fail, Tracker re initialize...
		float peak_Val = tracker.mPeak_value; // from tracker...
		//std::cout << peak_Val << std::endl;
		if (peak_Val < 0.15)
		{
			m_bTrackInit = false;
			m_bThw_warning = false;
		}
		// KCF tracker far away with hand... 171102. Problem??
		if (KCF_ReInit(KCF_result))
		{
			m_bTrackInit = false;
			m_bThw_warning = false;
		}

		// KCF visualize...
		rectangle(m_DispMat, Point(KCF_result.x, KCF_result.y), Point(KCF_result.x + KCF_result.width, KCF_result.y + KCF_result.height), Scalar(0, 255, 255), 1, LINE_AA);
		rectangle(m_DispMat, Point(m_KCF_rect_init.x, m_KCF_rect_init.y), Point(m_KCF_rect_init.x + m_KCF_rect_init.width, m_KCF_rect_init.y + m_KCF_rect_init.height), Scalar(100, 100, 255), 1, LINE_AA);

	}


	// 5. Combine with Prior ROI.
	// Human & ROI intersectiong exist, the alarm occurs...
	if (m_bROI)
	{
	
		trackResult.objectInfos.at(idx).box.width;
		Rect person_rect = Rect(trackResult.objectInfos.at(idx).box.x, trackResult.objectInfos.at(idx).box.y, trackResult.objectInfos.at(idx).box.width, trackResult.objectInfos.at(idx).box.height);
		Rect overlap_ROI_person = Rect(m_ROI_LT.x, m_ROI_LT.y, m_ROI_RB.x - m_ROI_LT.x, m_ROI_RB.y - m_ROI_LT.y) & person_rect;

		if (overlap_ROI_person.area() > 20)
		{
			m_bROI_warning = true;
		}
		else
			m_bROI_warning = false;

		// 5.2. Tracker disappear in the ROI, tkhe occlusion message appears. (ressonable?)
	}
	else
		m_bROI_warning = false;

	// EVENT VISUALIZE....


	// 6. Road Region Estimation using Attenuated Foreground map...
	// moving average by foreground map.
	//	GaussianBlur(m_fg_map, m_fg_map, Size(10, 10), 0, 0);
	region_accumulate(m_fg_map, Heatmap);



	Visualize(m_bROI_warning, m_bThw_warning, m_bThw_warning2);






	return true;



}