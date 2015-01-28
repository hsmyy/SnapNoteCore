/*
 * main.h
 *
 *  Created on: Jan 19, 2015
 *      Author: fc
 */

#ifndef MAIN_H_
#define MAIN_H_

#include <opencv2/opencv.hpp>

#define THRESHOLD(size, c) (c/size)

const double EPS = 1e-200;
double const SQRT2 = sqrt(2.0);

using namespace std;
using namespace cv;

typedef vector<double> vecD;
typedef vector<int> vecI;
typedef unsigned char byte;
typedef const Mat CMat;
typedef const string CStr;
#define _S(str) ((str).c_str())

typedef vector<float> vecF;
inline float sqr(float x) { return x * x; }

#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)

template<class T, int D> inline T vecSqrDist(const Vec<T, D> &v1, const Vec<T, D> &v2) {T s = 0; for (int i=0; i<D; i++) s += sqr(v1[i] - v2[i]); return s;} // out of range risk for T = byte, ...

template<class T, int D> inline T vecDist(const Vec<T, D> &v1, const Vec<T, D> &v2) { return sqrt(vecSqrDist(v1, v2)); }

template<class T> inline T pntSqrDist(const Point_<T> &p1, const Point_<T> &p2) {return sqr(p1.x - p2.x) + sqr(p1.y - p2.y);}

#define ForPoints2(pnt, xS, yS, xE, yE)	for (Point pnt(0, (yS)); pnt.y != (yE); pnt.y++) for (pnt.x = (xS); pnt.x != (xE); pnt.x++)

typedef pair<float, int> CostfIdx;

struct Region{
		Region() { pixNum = 0; ad2c = Point2d(0, 0);}
		int pixNum;  // Number of pixels
		vector<CostfIdx> freIdx;  // Frequency of each color and its index
		Point2d centroid;
		Point2d ad2c; // Average distance to image center
	};

Point const DIRECTION8[9] = {
	Point(1,  0), //Direction 0
	Point(1,  1), //Direction 1
	Point(0,  1), //Direction 2
	Point(-1, 1), //Direction 3
	Point(-1, 0), //Direction 4
	Point(-1,-1), //Direction 5
	Point(0, -1), //Direction 6
	Point(1, -1),  //Direction 7
	Point(0, 0),
};  //format: {dx, dy}

template<class T> void printMat(Mat &mat){
	for(int y = 0; y < mat.rows; ++y){
		T *row = mat.ptr<T>(y);
		for(int x = 0; x < mat.cols; ++x){
			cout << row[x] << ", ";
		}
		cout << endl;
	}
}

template<class T> void minmaxNorm(Mat &mat){
	T min = mat.at<T>(0,0),max = mat.at<T>(0,0);
	for(int y = 0; y < mat.rows; ++y){
		T *row = mat.ptr<T>(y);
		for(int x = 0; x < mat.cols; ++x){
			if(min > row[x]){
				min = row[x];
			}else if(max < row[x]){
				max = row[x];
			}
		}
	}
	T range = max - min;
	for(int y = 0; y < mat.rows; ++y){
		T * row = mat.ptr<T>(y);
		for(int x = 0; x < mat.cols; ++x){
			row[x] = (row[x] - min) / range;
		}
	}
}

template<class T> Mat convertToVisibleMat(Mat &mat){
	Mat result(mat.rows, mat.cols, CV_8UC3);
	for(int i = 0; i < mat.rows; ++i){
		Vec3b * row = result.ptr<Vec3b>(i);
		float * oRow = mat.ptr<T>(i);
		for(int j = 0; j < mat.cols; ++j){
			row[j][0] = oRow[j] * 255;
			row[j][1] = oRow[j] * 255;
			row[j][2] = oRow[j] * 255;
		}
	}
	return result;
}

#endif /* MAIN_H_ */
