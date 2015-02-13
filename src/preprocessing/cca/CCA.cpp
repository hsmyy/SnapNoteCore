//  Author:  www.icvpr.com
//  Blog  :  http://blog.csdn.net/icvpr

#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "CCA.h"
#include "../binarize/binarize.h"
#include "../utils/FileUtil.h"

using namespace cv;
using namespace std;

vector<Blob> blobs;
Mat filteredMat;
char win_name[] = "cca";
void areaChange(int minArea, void* userData) {
	filteredMat = Mat::zeros(filteredMat.size(), filteredMat.type());
	ostringstream os;

	//cout<<minArea<<endl;
	for (int i = 0; i < blobs.size(); i++) {
//		if ((blobs[i].area() < minArea) || (blobs[i].aspectRatio() > 5.0)
//				|| (blobs[i].aspectRatio() < (1.0 / 5))
//				|| (blobs[i].contentRatio() < (1.0 / 8)))
//			continue;
		cout << (blobs[i].aspectRatio() < 0.33) << " "
				<< blobs[i].contentRatio() << endl;
		vector<Point>& points = blobs[i].points;
		Vec3b rv(255 * (rand() / (1.0 + RAND_MAX)),
				255 * (rand() / (1.0 + RAND_MAX)),
				255 * (rand() / (1.0 + RAND_MAX)));
		for (int j = 0; j < points.size(); j++) {
			//Vec3b color = filteredMat.at<Vec3b>(points[j]);

			filteredMat.at<Vec3b>(Point(points[j].y, points[j].x)) = rv;
//			cout<<filteredMat.at<Vec3b>(points[j])<<endl;
//			cout<<rv<<endl;
		}
		os << blobs[i].area() << " : " << blobs[i].aspectRatio() << " : "
				<< blobs[i].contentRatio() << endl;

	}

	imshow(win_name, filteredMat);
	imwrite("ret.jpg", filteredMat);
	FileUtil::writeToFile(os.str(), "ret.txt");

}

//void ratioChange(int ratio, void* userData) {
//	if (ratio <= 0)
//		return;
//	double ratiod = ratio;
//	if (ratio < 1)
//		ratiod = 1 / ratiod;
//
//	filteredMat = Mat::zeros(filteredMat.size(), filteredMat.type());
//	//cout<<minArea<<endl;
//	for (int i = 0; i < blobs.size(); i++) {
//		if (blobs[i].aspectRatio() > ratiod
//				|| blobs[i].aspectRatio() < 1 / ratiod)
//			continue;
//		vector<Point>& points = blobs[i].points;
//		Vec3b rv(255 * (rand() / (1.0 + RAND_MAX)),
//				255 * (rand() / (1.0 + RAND_MAX)),
//				255 * (rand() / (1.0 + RAND_MAX)));
//		for (int j = 0; j < points.size(); j++) {
//			//Vec3b color = filteredMat.at<Vec3b>(points[j]);
//
//			filteredMat.at<Vec3b>(points[j]) = rv;
//		}
//	}
//
//	imshow(win_name, filteredMat);
//}

bool blobCom(const Blob b1, const Blob b2) {
	return (b1.area() < b2.area());
}

int main_cca(int argc, char** argv) {
	Mat binImage = imread("xing_14.jpg", IMREAD_GRAYSCALE);
	namedWindow(win_name, WINDOW_AUTOSIZE);
	Binarize::binarize(binImage, binImage);
	//adaptiveThreshold(binImage, binImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 41, 0);
	binImage = 255 - binImage;
	//cout<<binImage(Rect(100, 100, 50 ,50))<<endl;

	// connected component labeling
	Mat labelImg, colorImg;
	CCA::labelByTwoPass(binImage, labelImg);
	CCA::labelColor(labelImg, colorImg);
	namedWindow("color", WINDOW_AUTOSIZE);
	imshow("color", colorImg);

//	normalize(labelImg, labelImg, 0, 255, NORM_MINMAX);
//	imshow("label", labelImg);
	//cout<<labelImg(Rect(100, 100, 50 ,50))<<endl;

	CCA::findBlobs(labelImg, blobs);
	//sort(blobs.begin(), blobs.end(), blobCom);

	filteredMat = Mat::zeros(labelImg.rows, labelImg.cols, CV_8UC1);

	ostringstream os;

	for (int i = 0; i < blobs.size(); i++) {

		if(CCA::isGarbageBlob(blobs[i], filteredMat.cols, filteredMat.rows, blobs.size()))
			continue;
		vector<Point>& points = blobs[i].points;
		for (int j = 0; j < points.size(); j++) {

			filteredMat.ptr<uchar>(points[j].x)[points[j].y] = 255;
		}
		os << blobs[i].area() << " : " << blobs[i].aspectRatio() << " : "
				<< blobs[i].contentRatio() << endl;

	}

	imshow(win_name, filteredMat);
	imwrite("ret.jpg", filteredMat);
	FileUtil::writeToFile(os.str(), "ret.txt");

//	namedWindow(win_name);
//	int minArea = 1;
//	int ratio = 2;
//
//	createTrackbar("area", win_name, &minArea, 10000, areaChange, NULL);
//	areaChange(minArea, NULL);

//	createTrackbar("ratio", win_name, &ratio, 30, ratioChange, NULL);
//	ratioChange(ratio, NULL);

	waitKey(0);

	return 0;

}
