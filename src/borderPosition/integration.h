/*
 * integration.h
 *
 *  Created on: Feb 3, 2015
 *      Author: fc
 */

#ifndef BORDERPOSITION_INTEGRATION_H_
#define BORDERPOSITION_INTEGRATION_H_

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/*
 * test how many percent the border and salient intersect
 */
float coverage(vector<Point2f> borderPoints, Mat salientImg1f);

float coverage(vector<Point2f> borderPoints, Mat salientImg1f){
	Mat borderImg = Mat::zeros(salientImg1f.size(), CV_8UC1);
	for (unsigned int j = 0, len = borderPoints.size(); j < len; j++) {
		line(borderImg, borderPoints[j], borderPoints[(j + 1) % len], Scalar(255), 3, 8);
	}
	vector<vector<Point> > contours;
	findContours(borderImg, contours, RETR_EXTERNAL,
				CHAIN_APPROX_SIMPLE);
	int coverNum = 0;
	int rowNum = salientImg1f.rows;
	int colNum = salientImg1f.cols;

	for(int y = 0; y < rowNum; ++y){
		float *row = salientImg1f.ptr<float>(y);
		for(int x = 0; x < colNum; ++x){
			if(row[x] > 0){
				if(pointPolygonTest(contours[0],
					Point2f(x, y), false) > 0){
					++coverNum;
				}
			}
		}
	}
	cout << "coverNum:" << coverNum << endl;
	cout << "total:" << (rowNum * colNum) << endl;
	return coverNum / (float)(rowNum * colNum);
}



#endif /* BORDERPOSITION_INTEGRATION_H_ */
