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

class SalientRec{
public:
	void salient(const char *inputPath, const char *segPath = NULL, const char *rcPath = NULL);
	void salient(Mat &input, Mat &output);
	void salientDebug(const char *inputPath);
	void salientDebug(Mat &input, Mat &output);
	void wholeTest();
	void emptyTest();
	bool isResultUseful(Mat &input);
};

void SalientRec::salientDebug(Mat &input, Mat &output){
	Mat seg,regionIdxImage1i;
	const clock_t begin_time = clock();
	GraphSegmentation segmentation(1.2, 200, 1000, true);
	int regNum = segmentation.segment_image(input, regionIdxImage1i);
	seg = segmentation.getRealSeg();
	RegionContrastSalient rcs;
	Mat mat1 = rcs.getRC(input, regionIdxImage1i, regNum, 0.4, true);
	normalize(mat1, mat1, 0, 1, NORM_MINMAX);
	namedWindow("Seg");
	imshow("Seg", seg);
	RegionCut rc(0.1f, 0.9f, true);
	mat1 = rc.cut(mat1);
	output = convertToVisibleMat<float>(mat1);
	namedWindow("Final2");
	imshow("Final2", output);
	cout << "size:[" << input.rows << "*" << input.cols << "]" << input.rows * input.cols << "/" << float(clock() - begin_time) / 1000 << "ms" << endl;
}

bool SalientRec::isResultUseful(Mat &input){
	bool empty = true;
	for(int y = 0; y < input.rows; ++y){
		Vec3b *row = input.ptr<Vec3b>(y);
		for(int x = 0; x < input.cols; ++x, ++row){
			if((*row)[0] > 0 || (*row)[1] > 0 || (*row)[2] > 0){
				return true;
			}
		}
	}
	return false;
}

void SalientRec::salient(Mat &input, Mat &output){
	Mat regionIdxImage1i;
	//	const clock_t begin_time = clock();
	GraphSegmentation segmentation(1.2, 200, 1000, true);
	int regNum = segmentation.segment_image(input, regionIdxImage1i);
	RegionContrastSalient rcs;
	Mat mat1 = rcs.getRC(input, regionIdxImage1i, regNum, 0.4, false);
	RegionCut rc(0.1f, 0.9f);
	mat1 = rc.cut(mat1);
	output = convertToVisibleMat<float>(mat1);
}

void SalientRec::salient(const char *inputPath, const char *segPath, const char *rcPath){
	Mat image = imread(inputPath, 1);
	Mat seg,regionIdxImage1i;
//	const clock_t begin_time = clock();
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
//	cout << inputPath << ",size:[" << image.rows << "*" << image.cols << "]" << image.rows * image.cols << "/" << float(clock() - begin_time) / 1000 << "ms" << endl;

}

void SalientRec::salientDebug(const char *inputPath){
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



void SalientRec::wholeTest(){
	string input("test/SalientRec/input/"), seg("test/SalientRec/seg/"), output("test/SalientRec/output/");
	vector<string> inputCases = listFiles(input);
	size_t caseLen = inputCases.size();
	for (size_t i = 0; i < caseLen; ++i) {
		cout << inputCases[i] << ",";
		salient((input + inputCases[i]).c_str(), (seg + inputCases[i]).c_str(), (output + inputCases[i]).c_str());
	}
}

void SalientRec::emptyTest(){
	string input("test/SalientRec/output/");
	vector<string> inputCases = listFiles(input);
	for(int i = 0,len = (int)inputCases.size(); i < len; ++i){
		Mat mat = imread((input + inputCases[i]).c_str());
		cout << inputCases[i] << " empty:" << isResultUseful(mat) << endl;
	}
}

int main1(int argc, char** argv) {
//	salientDebug("test/input/imaget3.png");
//	salient("test/input/book3.jpg","test/seg/book3.jpg","test/output/book3.jpg");
//	wholeTest();
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

