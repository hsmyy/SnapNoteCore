//  Author:  www.icvpr.com
//  Blog  :  http://blog.csdn.net/icvpr
#include "../connectedComponentAnalysis/CCA.h"

#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;


vector<vector<Point> > blobs;
Mat filteredMat;
char win_name[] = "ccs";
void areaChange(int minArea, void* userData) {
	int kernelSize = *((int*) userData);
	filteredMat = Mat::zeros(filteredMat.size(), filteredMat.type());
	//cout<<minArea<<endl;
	for (int i = 0; i < blobs.size(); i++) {
		if (blobs[i].size() < minArea)
			continue;
		vector<Point>& points = blobs[i];
		for (int j = 0; j < points.size(); j++) {
			filteredMat.ptr<uchar>(points[j].x)[points[j].y] = 255;
		}
	}
	Mat dilated, eroded;

	Mat elem = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * kernelSize + 1, 2 * kernelSize + 1),
			Point(kernelSize, kernelSize));
	dilate(filteredMat, dilated, elem, Point(kernelSize, kernelSize));
	//erode(dilated, eroded, elem, Point(kernelSize, kernelSize));
	imshow(win_name, dilated);
}

void iterationChange(int iterations, void* userData) {
	int kernelSize = *((int*) userData);
//	filteredMat = Mat::zeros(filteredMat.size(), filteredMat.type());
//	//cout<<minArea<<endl;
//	for (int i = 0; i < blobs.size(); i++) {
//		if (blobs[i].size() < minArea)
//			continue;
//		vector<Point>& points = blobs[i];
//		for (int j = 0; j < points.size(); j++) {
//			filteredMat.ptr<uchar>(points[j].x)[points[j].y] = 255;
//		}
//	}
	Mat dilated, eroded;

	Mat elem = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * kernelSize + 1, 2 * kernelSize + 1),
			Point(kernelSize, kernelSize));
	dilate(filteredMat, dilated, elem, Point(kernelSize, kernelSize),
			iterations);
//	erode(dilated, eroded, elem, Point(kernelSize, kernelSize),
//			iterations);
	imshow(win_name, dilated);
}

void kernelChange(int kernelSize, void* userData) {
	int iterations = *((int*) userData);
//	filteredMat = Mat::zeros(filteredMat.size(), filteredMat.type());
//	//cout<<minArea<<endl;
//	for (int i = 0; i < blobs.size(); i++) {
//		if (blobs[i].size() < minArea)
//			continue;
//		vector<Point>& points = blobs[i];
//		for (int j = 0; j < points.size(); j++) {
//			filteredMat.ptr<uchar>(points[j].x)[points[j].y] = 255;
//		}
//	}
	Mat dilated, eroded;

	Mat elem = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * kernelSize + 1, 2 * kernelSize + 1),
			Point(kernelSize, kernelSize));
	dilate(filteredMat, dilated, elem, Point(kernelSize, kernelSize),
			iterations);
//	erode(dilated, eroded, elem, Point(kernelSize, kernelSize),
//			iterations);
	imshow(win_name, dilated);

}

int main_CCA(int argc, char** argv) {
	Mat binImage = imread("tmp/noise.png", IMREAD_GRAYSCALE);
	binImage = 255 - binImage;
	Mat A1 = (Mat_<uchar>(3, 3) << 0, 255, 0, 255, 255, 0, 0, 0, 255);
	threshold(binImage, binImage, 128, 1, THRESH_BINARY);

	// connected component labeling
	Mat labelImg;
	CCA::labelByTwoPass(binImage, labelImg);

	CCA::findBlobs(labelImg, blobs);

	filteredMat = Mat::zeros(labelImg.size(), CV_8UC1);

	namedWindow(win_name);
	int minArea = 0;
	int kernelSize = 0;
	int iterations = 1;
	createTrackbar("area", win_name, &minArea, 10, areaChange, &kernelSize);
	areaChange(minArea, &kernelSize);

	createTrackbar("kernel", win_name, &kernelSize, 10, kernelChange, &iterations);
	kernelChange(kernelSize, &iterations);

	createTrackbar("iterations", win_name, &iterations, 10, iterationChange, &kernelSize);
	iterationChange(iterations, &kernelSize);

	waitKey(0);

	return 0;

}
