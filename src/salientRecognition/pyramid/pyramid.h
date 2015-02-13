/*
 * pyramid.h
 *
 *  Created on: Feb 3, 2015
 *      Author: fc
 */

#ifndef SALIENTRECOGNITION_PYRAMID_PYRAMID_H_
#define SALIENTRECOGNITION_PYRAMID_PYRAMID_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

const int THRESHOLD_WIDTH = 800;
const int THRESHOLD_HEIGHT = 800;

class Pyramid{
public:
	Pyramid(Mat origin);
	Mat scale(bool resharp = false);
	Mat reScale(Mat mat);
	Rect reScale(Rect &rect);
	Mat getOrigin();
private:
	Mat _origin;
	Mat _final;
	int _scale;
};

Pyramid::Pyramid(Mat origin):
		_origin(origin),_scale(0){

}

Mat Pyramid::scale(bool resharp){
	_scale = 0;
	Mat mid = _origin;
	Mat smooth;
	while(mid.rows > THRESHOLD_HEIGHT || mid.cols > THRESHOLD_WIDTH){
		++_scale;
		pyrDown(mid, _final, Size(mid.cols / 2, mid.rows / 2));
		if(resharp){
			GaussianBlur(_final, smooth, cv::Size(0, 0), 3);
			addWeighted(_final, 1.5, smooth, -0.5, 0, _final);
		}
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

Rect Pyramid::reScale(Rect &rect){
	rect.x <<= _scale;
	rect.y <<= _scale;
	rect.width <<= _scale;
	rect.height <<= _scale;
	return rect;
}

#endif /* SALIENTRECOGNITION_PYRAMID_PYRAMID_H_ */
