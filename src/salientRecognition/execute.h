/*
 * main.cpp
 *
 *  Created on: Jan 19, 2015
 *      Author: fc
 */

#ifndef EXECUTE_H_
#define EXECUTE_H_

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
#include "rc/quantize.h"
#include "rc/cut2.h"
#include "rc/rc.h"
#include "segmentation/image.h"
#include "segmentation/misc.h"
#include "segmentation/pnmfile.h"
#include "segmentation/segment-image.h"
#include "pyramid/pyramid.h"
#include "../util/general.h"

using namespace cv;
using namespace std;

class SalientRec{
public:
	SalientRec(bool debug = false);
	~SalientRec();
	/**
	 * which is not thread-safe
	 */
	void salient(const char *inputPath, const char *segPath = NULL, const char *rcPath = NULL);
	/**
	 * which is not thread-safe
	 */
	void salient(Mat &input, Mat &output, Mat &seg);
	void wholeTest();
	void emptyTest();
	bool isResultUseful(Mat &input);
private:
	void debugStart();
	/**
	 * type: 1 means highContrast, 2 means lowContrast
	 */
	void debugEnd(Mat &input, Mat &output, GraphSegmentation *selection);
private:
	bool _debug;
	GraphSegmentation *highContrastSeg;
	GraphSegmentation *lowContrastSeg;
	clock_t tic;
	RegionContrastSalient *rcs;
	RegionCut *rc;
};

SalientRec::SalientRec(bool debug):
	_debug(debug){
	tic = clock();
	highContrastSeg = new GraphSegmentation(1.2, 200, 500, debug);
	lowContrastSeg = new GraphSegmentation(0.95, 200, 500, debug);
	rcs = new RegionContrastSalient(0.4, 2, 0.01, debug);
	rc = new RegionCut(0.1f, 0.9f, debug);
}

SalientRec::~SalientRec(){
	delete highContrastSeg;
	delete lowContrastSeg;
	delete rcs;
	delete rc;
}

void SalientRec::debugStart(){
	tic = clock();
}

void SalientRec::debugEnd(Mat &input, Mat &output, GraphSegmentation *selection){
	Mat seg = selection->getRealSeg();
	namedWindow("Seg");
	imshow("Seg", seg);
	namedWindow("Final2");
	imshow("Final2", output);
	cout << "size:[" << input.rows << "*" << input.cols << "]" << input.rows * input.cols << "/" << float(clock() - tic) / 1000 << "ms" << endl;
}

bool SalientRec::isResultUseful(Mat &input){
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

void SalientRec::salient(Mat &input, Mat &output, Mat &seg){
	Mat regionIdxImage1i;
	if(_debug){
		debugStart();
	}
	Pyramid pyramid(input);
	Mat scaledInput = pyramid.scale();
	General g(scaledInput);
	pair<Vec3b,double> p = g.meanVariance();
	if(_debug){
		cout << "mean variance:" << p.first << "," << p.second << endl;
	}
	int regNum;
	GraphSegmentation *selection;
	selection = p.second > 100 ? highContrastSeg : lowContrastSeg;
	regNum = selection->segment_image(scaledInput, regionIdxImage1i);
	seg = selection->getRealSeg();
	Mat mat1 = rcs->getRC(scaledInput, regionIdxImage1i, regNum, 0.4, false);
	mat1 = rc->cut(mat1, regionIdxImage1i);
	// if still not found, we can choose the largest one as salient.
	output = convertToVisibleMat<float>(mat1);
	output = pyramid.reScale(output);
	if(_debug){
		debugEnd(input, output, selection);
	}
}

void SalientRec::salient(const char *inputPath, const char *segPath, const char *rcPath){
	Mat input = imread(inputPath, 1);
	Mat seg,output;
	salient(input, output, seg);
	if(segPath != NULL){
		imwrite(segPath, seg);
	}
	if(rcPath != NULL){
		imwrite(rcPath, output);
	}
}

void SalientRec::wholeTest(){
	string input("test/SalientRec/input/"), seg("test/SalientRec/seg/"), output("test/SalientRec/output/");
	vector<string> inputCases = listFiles(input);
	size_t caseLen = inputCases.size();
	for (size_t i = 0; i < caseLen; ++i) {
		cout << inputCases[i] << ":";
		clock_t tic = clock();
		salient((input + inputCases[i]).c_str(), (seg + inputCases[i]).c_str(), (output + inputCases[i]).c_str());
		cout << float(clock() - tic) / 1000 << endl;
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
	waitKey(0);
	return 0;
}

#endif
