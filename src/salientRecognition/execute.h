/*
 * main.cpp
 *
 *  Created on: Jan 19, 2015
 *      Author: fc
 */

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
#include "rc/main.h"
//#include "segmentation.h"
#include "rc/quantize.h"
//#include "cut.h"
#include "rc/cut2.h"
#include "rc/rc.h"
#include "segmentation/image.h"
#include "segmentation/misc.h"
#include "segmentation/pnmfile.h"
#include "segmentation/segment-image.h"


using namespace cv;
using namespace std;

//t1 = 0.1, t2 = 0.9
Mat CutObjs(Mat &_img3f, Mat &_sal1f, float t1, float t2)
{
	// 1) clean border
	Mat _border1u = Mat();
	int wkSize = 20;
	Mat border1u = _border1u;
	if (border1u.data == NULL || border1u.size != _img3f.size){
		int bW = cvRound(0.02 * _img3f.cols), bH = cvRound(0.02 * _img3f.rows);
		border1u.create(_img3f.rows, _img3f.cols, CV_8U);
		border1u = 255;
		border1u(Rect(bW, bH, _img3f.cols - 2*bW, _img3f.rows - 2*bH)) = 0;
	}
	Mat sal1f, wkMask;
	_sal1f.copyTo(sal1f);
	sal1f.setTo(0, border1u);
	// 2) binarization
	cv::Rect rect(0, 0, _img3f.cols, _img3f.rows);
	if (wkSize > 0){
		//set low number pixel to 0
		threshold(sal1f, sal1f, t1, 1, THRESH_TOZERO);
		sal1f.convertTo(wkMask, CV_8U, 255);
		//set pixel whose number is less than 70 as 0
		threshold(wkMask, wkMask, 70, 255, THRESH_TOZERO);
		//check region
//		wkMask = CmSalCut::GetNZRegionsLS(wkMask, 0.005);
//		if (wkMask.data == NULL)
//			return Mat();
//		rect = CmSalCut::GetMaskRange(wkMask, wkSize);
//		sal1f(rect) = 255;
//		border1u = border1u(rect);
//		wkMask = wkMask(rect);
	}
//	namedWindow("binarization");
//	imshow("binarization", wkMask);
	return wkMask;
//	Mat img3f = _img3f(rect);
//	Mat fMask;
//	CmSalCut salCut(img3f);
//	salCut.initialize(sal1f, t1, t2);
//	const int outerIter = 4;
////	salCut.showMedialResults("Ini");
//	for (int j = 0; j < outerIter; j++)	{
//		salCut.fitGMMs();
//		int changed = 1000, times = 8;
//		cout << "A" << endl;
//		while (changed > 50 && times--) {
//			//salCut.showMedialResults("Medial results");
//			changed = salCut.refineOnce();
//			//waitKey(1);
//		}
//		cout << "A" << endl;
//		//salCut.showMedialResults(format("It%d", j));
//		//waitKey(0);
//		salCut.drawResult(fMask);
//		cout << "A" << endl;
//		fMask = CmSalCut::GetNZRegionsLS(fMask);
//		if (fMask.data == NULL)
//			return Mat();
//
//		if (j == outerIter - 1 || CmSalCut::ExpandMask(fMask, wkMask, border1u, 5) < 10)
//			break;
//
//		salCut.initialize(wkMask);
//		fMask.copyTo(wkMask);
//		cout << "A" << endl;
//	}
//
//	Mat resMask = Mat::zeros(_img3f.size(), CV_8U);
////	fMask.copyTo(resMask(rect));
//	return resMask;
}

void salient(const char *inputPath, const char *segPath, const char *rcPath){
	Mat image = imread(inputPath, 1);
	Mat seg,regionIdxImage1i;
	const clock_t begin_time = clock();
	GraphSegmentation segmentation(1.2, 200, 1000, true);
	int regNum = segmentation.segment_image(image, regionIdxImage1i);
	seg = segmentation.getRealSeg();
	RegionContrastSalient rcs;
	Mat mat1 = rcs.getRC(image, regionIdxImage1i, regNum, 0.4, false);

	imwrite(segPath, seg);
	RegionCut rc(0.1f, 0.9f);
	mat1 = rc.cut(mat1);
	Mat mat2 = convertToVisibleMat<float>(mat1);
	imwrite(rcPath, mat2);
	cout << inputPath << ",size:[" << image.rows << "*" << image.cols << "]" << image.rows * image.cols << "/" << float(clock() - begin_time) / 1000 << "ms" << endl;

}

void salientDebug(const char *inputPath){
	Mat image = imread(inputPath, 1);
	Mat seg,regionIdxImage1i;
	const clock_t begin_time = clock();
	GraphSegmentation segmentation(1.2, 200, 1000, true);
	int regNum = segmentation.segment_image(image, regionIdxImage1i);
	seg = segmentation.getRealSeg();
	RegionContrastSalient rcs;
	Mat mat1 = rcs.getRC(image, regionIdxImage1i, regNum, 0.4, true);

	normalize(mat1, mat1, 0, 1, NORM_MINMAX);
	namedWindow("Seg");
	imshow("Seg", seg);
//	namedWindow("Final");
//	imshow("Final", mat1);

	Mat cutMat;
	RegionCut rc(0.1f, 0.9f, true);
	cutMat = rc.cut(mat1);
	namedWindow("Final2");
	imshow("Final2", cutMat);
	cout << inputPath << ",size:[" << image.rows << "*" << image.cols << "]" << image.rows * image.cols << "/" << float(clock() - begin_time) / 1000 << "ms" << endl;
}



void wholeTest(){
	string input("test/SalientRec/input/"), seg("test/SalientRec/seg/"), output("test/SalientRec/output/");
	vector<string> inputCases = listFiles(input);
	size_t caseLen = inputCases.size();
	for (size_t i = 0; i < caseLen; ++i) {
		cout << inputCases[i] << ",";
		salient((input + inputCases[i]).c_str(), (seg + inputCases[i]).c_str(), (output + inputCases[i]).c_str());
	}
}

void emptyTest(){
	string input("test/SalientRec/output/");
	vector<string> inputCases = listFiles(input);
	for(int i = 0,len = (int)inputCases.size(); i < len; ++i){
		Mat mat = imread((input + inputCases[i]).c_str());
		bool empty = true;
		for(int j = 0; j < mat.rows; ++j){
			for(int k = 0; k < mat.cols; ++k){
				Vec3b color = mat.at<Vec3b>(j,k);
				if(color[0] > 0 || color[1] > 0 || color[2] > 0){
					empty = false;
					break;
				}
			}
			if(empty == false){
				break;
			}
		}
		cout << inputCases[i] << " empty:" << empty << endl;
	}
}

int main1(int argc, char** argv) {
//	salientDebug("test/input/imaget3.png");
//	salient("test/input/book3.jpg","test/seg/book3.jpg","test/output/book3.jpg");
	wholeTest();
//	emptyTest();

//	image = imread(imageFile, 1);
//	namedWindow("Origin");
//	imshow("Origin", image);
//	Mat seg;
//	Mat mat1 = getRC(image, seg, 0.4, 100, 1000, 1.5);

//	createTrackbar("sigma：", "Segmentation",&sigma,100, update );
//	createTrackbar("c：", "Segmentation",&c,100, update );
//	createTrackbar("min_size：", "Segmentation",&min_size,1000, update );

//	GaussianBlur(mat1, mat1, Size(9,9), 0);
//	normalize(mat1, mat1, 0, 1, NORM_MINMAX);
//	namedWindow("Segmentation");
//	imshow("Segmentation", mat1);
//	Mat cutMat;
//	float t = 0.9f;
//	int maxIt = 4;
//	while(cutMat.empty() && maxIt--){
//		cutMat = CutObjs(image, mat1, 0.1f, t);
//	}
//	namedWindow("Final");
//	imshow("Final", cutMat);

	waitKey(0);

	return 0;
}

