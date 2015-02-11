/*
 * textExtraction.h
 *
 *  Created on: Feb 10, 2015
 *      Author: fc
 */

#ifndef TEXTEXTRACTION_H_
#define TEXTEXTRACTION_H_

#include <iostream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

#include "RobustTextDetection.h"
#include "ConnectedComponent.h"
#include "lineFormation.h"
#include "../salientRecognition/pyramid/pyramid.h"

using namespace cv;
using namespace std;

class TextExtraction{
public:
	vector<Rect> textExtract(Mat &mat);
	void debug(Mat &originalImg, vector<Rect> regions, char* title);
	vector<Mat> findRegions(Mat &originalImg, vector<Rect> regions);
private:
	bool _debug;
};

vector<Rect> TextExtraction::textExtract(Mat &mat){

	Mat image1 = mat.clone();
	Pyramid pyramid(image1);
	Mat image = pyramid.scale(true);

	/* Quite a handful or params */
	RobustTextParam param;
	param.minMSERArea        = 10;
	param.maxMSERArea        = 2000;
	param.cannyThresh1       = 20;
	param.cannyThresh2       = 100;

	param.maxConnCompCount   = 10000;
	param.minConnCompArea    = 15;// origin 75
	param.maxConnCompArea    = 800;

	param.minEccentricity    = 0.1;
	param.maxEccentricity    = 0.995;
	param.minSolidity        = 0.4;
	param.maxStdDevMeanRatio = 0.5;

	/* Apply Robust Text Detection */
	/* ... remove this temp output path if you don't want it to write temp image files */
	string temp_output_path = ".";
	RobustTextDetection detector(param );
	pair<Mat, Rect> result = detector.apply( image );

	LineFormation lf;
	vector<Rect> rects = lf.findLines(result.first);
	debug(image, rects, "scaledResult");
	for(unsigned int i = 0, len = rects.size(); i < len; ++i){
		Rect r = rects[i];
		rects[i] = pyramid.reScale(r);
	}
	return rects;
}

void TextExtraction::debug(Mat &originalImg, vector<Rect> regions, char * title){
	for(unsigned int i = 0, len = regions.size(); i < len; ++i){
		Rect r = regions[i];
		rectangle( originalImg, r, Scalar(0, 0, 255), 2);
	}
	namedWindow(title);
	imshow(title, originalImg);
}

vector<Mat> TextExtraction::findRegions(Mat &originalImg, vector<Rect> regions){
	vector<Mat> textRegions(regions.size());
	for(unsigned int i = 0, len = regions.size(); i < len; ++i){
		textRegions[i] = Mat(originalImg, regions[i]);
	}
	return textRegions;
}

#endif /* TEXTEXTRACTION_H_ */
