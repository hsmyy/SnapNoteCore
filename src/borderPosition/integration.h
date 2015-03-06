/*
 * integration.h
 *
 *  Created on: Feb 3, 2015
 *      Author: fc
 */

#ifndef BORDERPOSITION_INTEGRATION_H_
#define BORDERPOSITION_INTEGRATION_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/*
 * test how many percent the border and salient intersect
 */
pair<float,float> coverage(vector<Point2f> borderPoints, Mat salientImg1f);

pair<float,float> coverage(vector<Point2f> borderPoints, Mat salientImg1f){

	if(countNonZero(salientImg1f)<0.1*salientImg1f.cols*salientImg1f.rows)
		return make_pair(0,0);

	Mat borderImg = Mat::zeros(salientImg1f.size(), CV_8UC1);
	for (unsigned int j = 0, len = borderPoints.size(); j < len; j++) {
		line(borderImg, borderPoints[j], borderPoints[(j + 1) % len], Scalar(255), 3, 8);
	}
	vector<vector<Point> > contours;
	findContours(borderImg, contours, RETR_EXTERNAL,
				CHAIN_APPROX_SIMPLE);
	int intersectNum = 0;
	int salientNum = 0;
	int borderNum = 0;
	int rowNum = salientImg1f.rows;
	int colNum = salientImg1f.cols;

	for(int y = 0; y < rowNum; ++y){
		float *row = salientImg1f.ptr<float>(y);
		for(int x = 0; x < colNum; ++x){
			if(pointPolygonTest(contours[0],
				Point2f(x, y), false) > 0){
				++borderNum;
				if(row[x] > 0){
					++intersectNum;
					++salientNum;
				}
			}else{
				if(row[x] > 0){
					++salientNum;
				}
			}
		}
	}
	//cout<<"PR: "<<salientNum<<" "<<borderNum<<" "<<intersectNum<<endl;
	float precision = borderNum > 0 ? intersectNum / (float)borderNum : 0;
	float recall = salientNum > 0 ? intersectNum / (float)salientNum : 0;
	return make_pair( precision, recall);
}



#endif /* BORDERPOSITION_INTEGRATION_H_ */
