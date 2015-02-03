/*
 * pyramid.h
 *
 *  Created on: Feb 3, 2015
 *      Author: fc
 */

#ifndef SALIENTRECOGNITION_PYRAMID_PYRAMID_H_
#define SALIENTRECOGNITION_PYRAMID_PYRAMID_H_

#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

const int THRESHOLD_WIDTH = 600;
const int THRESHOLD_HEIGHT = 600;

class Pyramid{
public:
	Pyramid(Mat origin);
	Mat scale();
	Mat reScale(Mat mat);
	Mat getOrigin();
private:
	Mat _origin;
	Mat _final;
	int _scale;
};

Pyramid::Pyramid(Mat origin):
		_origin(origin),_scale(0){

}

Mat Pyramid::scale(){
	_scale = 0;
	Mat mid = _origin;
	while(mid.rows > THRESHOLD_HEIGHT || mid.cols > THRESHOLD_WIDTH){
		++_scale;
		pyrDown(mid, _final, Size(mid.cols / 2, mid.rows / 2));
		mid = _final;
	}
	return _final;
}

Mat Pyramid::getOrigin(){
	return _origin;
}

Mat Pyramid::reScale(Mat mat){
	Mat res, mid = mat;
	for(int i = 0; i < _scale; ++i){
		pyrUp(mid, res, Size(mid.cols * 2, mid.rows * 2));
		mid = res;
	}
	return res;
}

#endif /* SALIENTRECOGNITION_PYRAMID_PYRAMID_H_ */
